"""Path-consistent primal construction for the exact LBBD/CBBD algorithms.

The repair is deterministic and contains no K-BRKGA component.  From four
transparent pathway-order rules and a fixed appointment grid, it synchronously
swaps two patients with the same clinical pathway at every attended stage.
Each trial is evaluated by the exact full-scenario recourse evaluator.  The
returned schedule is only a feasible primal incumbent; it supplies neither cuts
nor lower bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from de_milp import natural_assignment
from fixed_schedule_validation import validate_fixed_schedule


@dataclass
class PathwisePrimalStart:
    assignment: np.ndarray
    y: np.ndarray
    objective: float
    evaluations: int
    accepted_swaps: int
    accepted_timing_moves: int
    runtime_sec: float


def _assignment_to_x(assignment: np.ndarray) -> np.ndarray:
    h_count, position_count = assignment.shape
    x = np.zeros((h_count, position_count, position_count), dtype=float)
    for h in range(h_count):
        x[h, np.arange(position_count), assignment[h, :]] = 1.0
    return x


def _even_grid(inst: Dict) -> np.ndarray:
    y = np.zeros((int(inst["H"]), int(inst["I"])), dtype=float)
    for h in range(int(inst["H"])):
        first = int(inst["Iv"][h])
        count = int(inst["I"]) - first
        if count > 1:
            y[h, first:] = np.linspace(0.0, float(inst["L"][h]), count)
    return y


def _objective(inst: Dict, assignment: np.ndarray, y: np.ndarray) -> float:
    audit = validate_fixed_schedule(
        inst, _assignment_to_x(assignment), y, tol=1e-6
    )
    return float(audit.objective) if audit.feasible else math.inf


def _pathway_groups(c: np.ndarray) -> List[List[int]]:
    groups: Dict[Tuple[int, ...], List[int]] = {}
    for j in range(c.shape[1]):
        groups.setdefault(tuple(int(value) for value in c[:, j]), []).append(j)
    return list(groups.values())


def _starting_assignment(inst: Dict, mode: str) -> np.ndarray:
    """Build one deterministic pathway-block order without random search."""
    base, _ = natural_assignment(inst["c"], inst["Iv"])
    base = np.asarray(base, dtype=int)
    groups = _pathway_groups(np.asarray(inst["c"], dtype=int))
    service = np.zeros(int(inst["J"]), dtype=float)
    lateness = np.zeros(int(inst["J"]), dtype=float)
    for j in range(int(inst["J"])):
        for h in range(int(inst["H"])):
            if int(inst["c"][h, j]) == 1:
                service[j] += float(
                    np.mean(inst["s"][:, h, int(inst["Iv"][h]) :, j])
                )
        lateness[j] = float(
            np.mean(inst["tau"][:, j]) + np.mean(inst["psi"][:, j])
        )

    group_id: Dict[int, int] = {}
    rank: Dict[int, int] = {}
    for group_index, group in enumerate(groups):
        if mode == "natural":
            order = list(group)
        elif mode == "reverse":
            order = list(reversed(group))
        elif mode == "service_up":
            order = sorted(group, key=lambda j: (service[j], j))
        elif mode == "lateness_up":
            order = sorted(group, key=lambda j: (lateness[j], j))
        else:
            raise ValueError(f"Unknown deterministic start mode {mode!r}.")
        for position, patient in enumerate(order):
            group_id[int(patient)] = int(group_index)
            rank[int(patient)] = int(position)

    assignment = base.copy()
    for h in range(int(inst["H"])):
        first = int(inst["Iv"][h])
        active = [int(j) for j in assignment[h, first:]]
        active.sort(key=lambda j: (group_id[j], rank[j], j))
        assignment[h, first:] = np.asarray(active, dtype=int)
    return assignment


def _local_descent(
    inst: Dict,
    assignment: np.ndarray,
    y: np.ndarray,
    groups: List[List[int]],
    *,
    max_passes: int,
    deadline: float,
    improvement_tol: float,
) -> Tuple[np.ndarray, float, int, int]:
    best_objective = _objective(inst, assignment, y)
    evaluations = 1
    accepted_swaps = 0
    if not np.isfinite(best_objective):
        return assignment, math.inf, evaluations, accepted_swaps
    for _ in range(max(0, int(max_passes))):
        if time.time() >= deadline:
            break
        pass_objective = best_objective
        pass_assignment: Optional[np.ndarray] = None
        for group in groups:
            for left_index, patient_left in enumerate(group[:-1]):
                for patient_right in group[left_index + 1 :]:
                    if time.time() >= deadline:
                        break
                    trial = assignment.copy()
                    for h in range(int(inst["H"])):
                        left_position = int(np.flatnonzero(trial[h] == patient_left)[0])
                        right_position = int(np.flatnonzero(trial[h] == patient_right)[0])
                        trial[h, left_position], trial[h, right_position] = (
                            trial[h, right_position], trial[h, left_position]
                        )
                    objective = _objective(inst, trial, y)
                    evaluations += 1
                    if objective < pass_objective - float(improvement_tol):
                        pass_objective = objective
                        pass_assignment = trial
                if time.time() >= deadline:
                    break
            if time.time() >= deadline:
                break
        if pass_assignment is None:
            break
        assignment = pass_assignment
        best_objective = pass_objective
        accepted_swaps += 1
    return assignment, best_objective, evaluations, accepted_swaps


def _timing_polish(
    inst: Dict,
    assignment: np.ndarray,
    y: np.ndarray,
    *,
    deadline: float,
    max_passes: int = 3,
    improvement_tol: float = 1e-9,
) -> Tuple[np.ndarray, float, int, int]:
    """Deterministically polish appointment times by bounded coordinates."""
    best_objective = _objective(inst, assignment, y)
    evaluations = 1
    accepted_moves = 0
    fractions = (0.2, 0.4, 0.6, 0.8)
    for _ in range(max(0, int(max_passes))):
        if time.time() >= deadline:
            break
        improved = False
        for h in range(int(inst["H"])):
            first = int(inst["Iv"][h])
            for i in range(first, int(inst["I"])):
                if time.time() >= deadline:
                    break
                lower = 0.0 if i == first else float(y[h, i - 1])
                upper = (
                    float(inst["L"][h])
                    if i == int(inst["I"]) - 1
                    else float(y[h, i + 1])
                )
                current = float(y[h, i])
                local_best = best_objective
                best_value = current
                for fraction in fractions:
                    if time.time() >= deadline:
                        break
                    value = lower + fraction * (upper - lower)
                    if abs(value - current) <= 1e-10:
                        continue
                    trial_y = y.copy()
                    trial_y[h, i] = value
                    objective = _objective(inst, assignment, trial_y)
                    evaluations += 1
                    if objective < local_best - float(improvement_tol):
                        local_best = objective
                        best_value = value
                if abs(best_value - current) > 1e-10:
                    y[h, i] = best_value
                    best_objective = local_best
                    accepted_moves += 1
                    improved = True
            if time.time() >= deadline:
                break
        if not improved:
            break
    return y, best_objective, evaluations, accepted_moves


def build_pathwise_primal_start(
    inst: Dict,
    *,
    max_passes: int = 8,
    time_limit: float = 5.0,
    improvement_tol: float = 1e-9,
) -> Optional[PathwisePrimalStart]:
    """Run deterministic best-improvement pathway-synchronous interchanges."""
    started = time.time()
    deadline = started + max(0.0, float(time_limit))
    y = _even_grid(inst)
    groups = _pathway_groups(np.asarray(inst["c"], dtype=int))
    best_assignment: Optional[np.ndarray] = None
    best_objective = math.inf
    evaluations = 0
    accepted_swaps = 0
    for mode in ("natural", "reverse", "service_up", "lateness_up"):
        if time.time() >= deadline:
            break
        assignment, objective, used, swaps = _local_descent(
            inst,
            _starting_assignment(inst, mode),
            y,
            groups,
            max_passes=max_passes,
            deadline=deadline,
            improvement_tol=improvement_tol,
        )
        evaluations += used
        accepted_swaps += swaps
        if objective < best_objective - float(improvement_tol):
            best_assignment = assignment
            best_objective = objective

    if best_assignment is None or not np.isfinite(best_objective):
        return None

    polished_y, polished_objective, timing_evaluations, timing_moves = (
        _timing_polish(
            inst,
            best_assignment,
            y.copy(),
            deadline=deadline,
            max_passes=3,
            improvement_tol=improvement_tol,
        )
    )
    evaluations += timing_evaluations
    if polished_objective < best_objective - float(improvement_tol):
        y = polished_y
        best_objective = polished_objective
    else:
        timing_moves = 0

    return PathwisePrimalStart(
        assignment=best_assignment,
        y=y,
        objective=float(best_objective),
        evaluations=int(evaluations),
        accepted_swaps=int(accepted_swaps),
        accepted_timing_moves=int(timing_moves),
        runtime_sec=float(time.time() - started),
    )


__all__ = ["PathwisePrimalStart", "build_pathwise_primal_start"]
