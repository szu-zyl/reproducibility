"""Shared instance-generation and evaluation utilities for the experiments."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from scenario_generation import generate_all_scenarios_independent
from model_utils import compute_data_driven_bigM, compute_first_follow


DEFAULT_MEAN_SERVICE = (2.0, 2.0, 2.0)
DEFAULT_SD_SERVICE = (0.0, 1.0 / 3.0, 2.0 / 3.0)


def parse_ints(text: str) -> List[int]:
    """Parse a comma-separated integer list."""
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def coupled_pathway(patient_count: int) -> np.ndarray:
    """Build the 50%/25%/25% three-stage pathway mix."""
    if patient_count < 4 or patient_count % 4 != 0:
        raise ValueError("patient_count must be a positive multiple of four.")
    all_stages = patient_count // 2
    early_path = patient_count // 4
    pathway = np.zeros((3, patient_count), dtype=int)
    pathway[:, :all_stages] = 1
    pathway[0:2, all_stages : all_stages + early_path] = 1
    pathway[1:3, all_stages + early_path :] = 1
    return pathway


def coupled_tolerances(patient_count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Assign pathway-specific direct and inter-stage waiting tolerances."""
    all_stages = patient_count // 2
    early_path = patient_count // 4
    direct = np.empty(patient_count, dtype=float)
    inter_stage = np.empty(patient_count, dtype=float)
    direct[:all_stages], inter_stage[:all_stages] = 2.0, 3.2
    direct[all_stages : all_stages + early_path] = 2.2
    inter_stage[all_stages : all_stages + early_path] = 3.5
    direct[all_stages + early_path :] = 2.4
    inter_stage[all_stages + early_path :] = 4.0
    return direct, inter_stage


def stable_windows(
    pathway: np.ndarray,
    buffer_z: float = 0.5,
    terminal_shortfall: float = 0.6,
    mean_service_by_stage: Tuple[float, float, float] = DEFAULT_MEAN_SERVICE,
    sd_service_by_stage: Tuple[float, float, float] = DEFAULT_SD_SERVICE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct distribution-calibrated time windows for each stage."""
    if buffer_z < 0.0:
        raise ValueError("buffer_z must be nonnegative.")
    if terminal_shortfall < 0.0:
        raise ValueError("terminal_shortfall must be nonnegative.")
    spacing = np.asarray(mean_service_by_stage, dtype=float) + float(buffer_z) * np.asarray(
        sd_service_by_stage, dtype=float
    )
    windows = np.zeros(pathway.shape[0], dtype=float)
    for stage in range(pathway.shape[0]):
        active_count = int(np.sum(pathway[stage, :] == 1))
        windows[stage] = max(
            0.5,
            max(0, active_count - 1) * float(spacing[stage])
            - float(terminal_shortfall),
        )
    return windows, spacing


def build_scale_stable_instance(
    J: int,
    K: int,
    seed_service: int,
    seed_lateness: int,
    buffer_z: float = 0.5,
    terminal_shortfall: float = 0.6,
) -> Dict:
    """Generate one patient-level instance used in the algorithm comparison."""
    H, I = 3, J
    pathway = coupled_pathway(J)
    direct_tolerance, inter_stage_tolerance = coupled_tolerances(J)
    virtual_count = [int(J - np.sum(pathway[h, :])) for h in range(H)]
    service, initial_lateness, carried_lateness = generate_all_scenarios_independent(
        numSamples=K,
        numStages=H,
        numPositions=I,
        numPatients=J,
        c=pathway,
        seed_service=seed_service,
        seed_lateness=seed_lateness,
    )
    windows, spacing = stable_windows(pathway, buffer_z, terminal_shortfall)
    probabilities = np.full(K, 1.0 / K, dtype=float)
    big_m = compute_data_driven_bigM(
        s=service,
        tau=initial_lateness,
        psi=carried_lateness,
        c=pathway,
        L=windows,
        Iv=virtual_count,
    )
    return {
        "H": H,
        "I": I,
        "J": J,
        "K": K,
        "c": pathway,
        "r_d": direct_tolerance,
        "r_s": inter_stage_tolerance,
        "L": windows,
        "spacing": spacing,
        "Iv": virtual_count,
        "physical": [int(np.sum(pathway[h, :])) for h in range(H)],
        "s": service,
        "tau": initial_lateness,
        "psi": carried_lateness,
        "p": probabilities,
        "bigM": big_m,
        "seed_service": int(seed_service),
        "seed_lateness": int(seed_lateness),
        "pattern": "coupled_50_25_25",
        "formulation": "compact_patient_completion",
        "window_mode": "distribution_buffer_fixed_shortfall",
        "buffer_z": float(buffer_z),
        "terminal_shortfall": float(terminal_shortfall),
    }


def independent_wait_metrics(
    inst: Dict,
    x: np.ndarray,
    y: np.ndarray,
    test_K: int,
    seed_service: int,
    seed_lateness: int,
    enforce_cross_stage_precedence: bool = False,
) -> Dict[str, float]:
    """Evaluate a fixed schedule on newly generated scenarios."""
    service, initial_lateness, carried_lateness = generate_all_scenarios_independent(
        numSamples=int(test_K),
        numStages=inst["H"],
        numPositions=inst["I"],
        numPatients=inst["J"],
        c=inst["c"],
        seed_service=int(seed_service),
        seed_lateness=int(seed_lateness),
    )
    first, follow = compute_first_follow(inst["c"])
    assignment = np.argmax(np.asarray(x), axis=2)
    direct_values: List[float] = []
    indirect_values: List[float] = []
    direct_exceed = 0
    indirect_exceed = 0

    for scenario in range(test_K):
        completion_by_patient = np.full(
            (inst["H"], inst["J"]), np.nan, dtype=float
        )
        for stage in range(inst["H"]):
            department_available = 0.0
            for position in range(inst["Iv"][stage], inst["I"]):
                patient = int(assignment[stage, position])
                deviation = float(
                    initial_lateness[scenario, patient] * first[stage, patient]
                    + carried_lateness[scenario, patient] * follow[stage, patient]
                )
                arrival = float(y[stage, position]) + deviation
                previous_stages = [
                    earlier
                    for earlier in range(stage)
                    if int(inst["c"][earlier, patient]) == 1
                ]
                previous_completion = 0.0
                if previous_stages:
                    previous_completion = float(
                        completion_by_patient[previous_stages[-1], patient]
                    )
                    if not np.isfinite(previous_completion):
                        previous_completion = 0.0
                readiness = max(
                    arrival,
                    previous_completion if enforce_cross_stage_precedence else 0.0,
                )
                start = max(readiness, department_available)
                direct_wait = max(0.0, start - readiness)
                duration = max(
                    0.0,
                    float(service[scenario, stage, position, patient]),
                )
                completion = start + duration
                completion_by_patient[stage, patient] = completion
                department_available = completion

                direct_values.append(direct_wait)
                direct_exceed += int(
                    direct_wait > float(inst["r_d"][patient]) + 1e-9
                )

                if previous_stages:
                    previous = previous_stages[-1]
                    upstream_completion = completion_by_patient[previous, patient]
                    inter_stage_wait = (
                        0.0
                        if not np.isfinite(upstream_completion)
                        else max(0.0, arrival - upstream_completion)
                    )
                    indirect_values.append(inter_stage_wait)
                    indirect_exceed += int(
                        inter_stage_wait > float(inst["r_s"][patient]) + 1e-9
                    )

    direct = np.asarray(direct_values, dtype=float)
    indirect = np.asarray(indirect_values, dtype=float)
    return {
        "test_mean_direct": float(np.mean(direct)) if direct.size else 0.0,
        "test_p95_direct": float(np.quantile(direct, 0.95)) if direct.size else 0.0,
        "test_direct_exceed_rate": float(direct_exceed / max(1, direct.size)),
        "test_mean_indirect": float(np.mean(indirect)) if indirect.size else 0.0,
        "test_p95_indirect": float(np.quantile(indirect, 0.95)) if indirect.size else 0.0,
        "test_indirect_exceed_rate": float(
            indirect_exceed / max(1, indirect.size)
        ),
    }
