"""Independent fixed-schedule validation for the final formulation.

The optimization classes are deliberately not imported here.  Given an
assignment/schedule returned by any method, this module reconstructs arrivals,
service starts, completions, and the best feasible tolerance credits directly
from the mathematical definitions.  The resulting objective is therefore a
method-independent audit value suitable for reported comparisons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class FixedScheduleValidation:
    status: str
    feasible: bool
    objective: float = math.inf
    max_violation: float = math.inf
    assignment: Optional[np.ndarray] = None
    alpha: Optional[np.ndarray] = None
    beta: Optional[np.ndarray] = None
    ready: Optional[np.ndarray] = None
    start: Optional[np.ndarray] = None
    completion: Optional[np.ndarray] = None


def _first_and_previous(c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h_count, patient_count = c.shape
    first = np.zeros_like(c, dtype=int)
    previous = np.full((h_count, patient_count), -1, dtype=int)
    for j in range(patient_count):
        attended = [h for h in range(h_count) if int(c[h, j]) == 1]
        if attended:
            first[attended[0], j] = 1
        for index in range(1, len(attended)):
            previous[attended[index], j] = attended[index - 1]
    return first, previous


def _assignment_from_x(x: np.ndarray, tol: float) -> Tuple[Optional[np.ndarray], float]:
    h_count, position_count, patient_count = x.shape
    assignment = np.argmax(x, axis=2).astype(int)
    violation = 0.0
    for h in range(h_count):
        violation = max(
            violation,
            float(np.max(np.abs(np.sum(x[h, :, :], axis=1) - 1.0))),
            float(np.max(np.abs(np.sum(x[h, :, :], axis=0) - 1.0))),
        )
        if sorted(int(j) for j in assignment[h, :]) != list(range(patient_count)):
            return None, max(violation, 1.0)
        for i in range(position_count):
            chosen = int(assignment[h, i])
            violation = max(violation, abs(float(x[h, i, chosen]) - 1.0))
    return (assignment if violation <= tol else None), violation


def validate_fixed_schedule(
    inst: Dict,
    x: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-6,
) -> FixedScheduleValidation:
    """Audit a fixed schedule and independently recompute its objective."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    c = np.asarray(inst["c"], dtype=int)
    s = np.asarray(inst["s"], dtype=float)
    tau = np.asarray(inst["tau"], dtype=float)
    psi = np.asarray(inst["psi"], dtype=float)
    p = np.asarray(inst["p"], dtype=float)
    r_d = np.asarray(inst["r_d"], dtype=float)
    r_s = np.asarray(inst["r_s"], dtype=float)
    iv = np.asarray(inst["Iv"], dtype=int)
    limits = np.asarray(inst["L"], dtype=float)

    h_count, patient_count = c.shape
    position_count = patient_count
    scenario_count = int(s.shape[0])
    if x.shape != (h_count, position_count, patient_count):
        return FixedScheduleValidation("INVALID_X_SHAPE", False)
    if y.shape != (h_count, position_count):
        return FixedScheduleValidation("INVALID_Y_SHAPE", False)
    if p.shape != (scenario_count,) or not np.isclose(np.sum(p), 1.0, atol=1e-8):
        return FixedScheduleValidation("INVALID_PROBABILITIES", False)

    assignment, max_violation = _assignment_from_x(x, max(tol, 1e-5))
    if assignment is None:
        return FixedScheduleValidation(
            "INVALID_ASSIGNMENT", False, max_violation=max_violation
        )

    position: Dict[Tuple[int, int], int] = {}
    for h in range(h_count):
        for i in range(position_count):
            j = int(assignment[h, i])
            position[h, j] = i
            expected_active = int(i >= iv[h])
            max_violation = max(max_violation, abs(int(c[h, j]) - expected_active))
            if i < iv[h]:
                max_violation = max(max_violation, abs(float(y[h, i])))
        active_y = y[h, iv[h] :]
        if active_y.size:
            max_violation = max(
                max_violation,
                max(0.0, -float(np.min(active_y))),
                max(0.0, float(np.max(active_y)) - float(limits[h])),
            )
        if active_y.size > 1:
            max_violation = max(
                max_violation, max(0.0, -float(np.min(np.diff(active_y))))
            )
    if max_violation > max(tol, 1e-5):
        return FixedScheduleValidation(
            "STRUCTURAL_VIOLATION",
            False,
            max_violation=max_violation,
            assignment=assignment,
        )

    first, previous = _first_and_previous(c)
    arrival = np.zeros((h_count, position_count, scenario_count), dtype=float)
    ready = np.zeros_like(arrival)
    start = np.zeros_like(arrival)
    completion = np.zeros_like(arrival)
    alpha = np.zeros((h_count, position_count), dtype=float)
    beta = np.zeros((h_count, position_count), dtype=float)

    for h in range(h_count):
        for i in range(iv[h], position_count):
            j = int(assignment[h, i])
            tardiness = tau[:, j] * first[h, j] + psi[:, j] * (1 - first[h, j])
            arrival[h, i, :] = float(y[h, i]) + tardiness
            predecessor = np.zeros(scenario_count, dtype=float)
            previous_h = int(previous[h, j])
            if previous_h >= 0:
                previous_i = int(position[previous_h, j])
                predecessor = completion[previous_h, previous_i, :]
            ready[h, i, :] = np.maximum(arrival[h, i, :], predecessor)
            start_inputs = [ready[h, i, :]]
            if i > iv[h]:
                start_inputs.append(completion[h, i - 1, :])
            start[h, i, :] = np.maximum.reduce(start_inputs)
            completion[h, i, :] = start[h, i, :] + s[:, h, i, j]

    satisfaction = 0.0
    for h in range(h_count):
        for i in range(iv[h], position_count):
            j = int(assignment[h, i])
            direct = np.maximum(0.0, start[h, i, :] - ready[h, i, :])
            direct_excess = float(np.dot(p, direct) - r_d[j])
            max_violation = max(max_violation, direct_excess)
            if direct_excess > tol:
                return FixedScheduleValidation(
                    "DIRECT_TOLERANCE_VIOLATION",
                    False,
                    max_violation=max_violation,
                    assignment=assignment,
                    alpha=alpha,
                    beta=beta,
                )
            alpha[h, i] = min(r_d[j], max(0.0, r_d[j] - float(np.dot(p, direct))))
            satisfaction += float(alpha[h, i])

            previous_h = int(previous[h, j])
            if previous_h < 0:
                beta[h, i] = float(r_s[j])
                satisfaction += float(beta[h, i])
                continue
            previous_i = int(position[previous_h, j])
            raw = arrival[h, i, :] - completion[previous_h, previous_i, :]
            tolerance = float(r_s[j])

            def required(beta_value: float) -> float:
                return float(np.dot(p, np.maximum(0.0, raw + beta_value)))

            indirect_excess = required(0.0) - tolerance
            max_violation = max(max_violation, indirect_excess)
            if indirect_excess > tol:
                return FixedScheduleValidation(
                    "INDIRECT_TOLERANCE_VIOLATION",
                    False,
                    max_violation=max_violation,
                    assignment=assignment,
                    alpha=alpha,
                    beta=beta,
                )
            if required(tolerance) <= tolerance + tol:
                beta[h, i] = tolerance
            else:
                lower, upper = 0.0, tolerance
                for _ in range(60):
                    middle = 0.5 * (lower + upper)
                    if required(middle) <= tolerance:
                        lower = middle
                    else:
                        upper = middle
                beta[h, i] = lower
            satisfaction += float(beta[h, i])

    base = float(
        sum(
            r_d[j] + r_s[j]
            for h in range(h_count)
            for j in range(patient_count)
            if int(c[h, j]) == 1
        )
    )
    objective = float(base - satisfaction)
    return FixedScheduleValidation(
        "PASS",
        True,
        objective=objective,
        max_violation=max(0.0, float(max_violation)),
        assignment=assignment,
        alpha=alpha,
        beta=beta,
        ready=ready,
        start=start,
        completion=completion,
    )


__all__ = ["FixedScheduleValidation", "validate_fixed_schedule"]
