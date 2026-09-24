"""Patient-level K-BRKGA benchmark for the final formulation.

This module adapts the knowledge-based biased random-key idea used by Yang
et al. (2025) to the three-stage M-TAD instances.  It is deliberately kept
solver-free inside the evolutionary loop:

* patient keys determine the order inside each clinical-pathway class;
* a pathway-aware decoder preserves the relative block structure needed for
  feasible cross-stage flow without aggregating patients;
* the paper baseline assigns patients to the common scale-stable appointment
  slots, matching the fixed-slot allocation role of the cited K-BRKGA; and
* every chromosome is scored with the same finite-scenario waiting
  definitions used by the independent fixed-schedule validator.

The result is an adapted heuristic benchmark, not an exact method and not a
source of bounds.  Its final schedule must still be passed through
``validate_fixed_schedule`` by the experiment runner.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from fixed_schedule_validation import (
    FixedScheduleValidation,
    validate_fixed_schedule,
)


@dataclass
class KBRKGAResult:
    status: str
    obj: float = math.inf
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    alpha: Optional[np.ndarray] = None
    beta: Optional[np.ndarray] = None
    best_gen: int = -1
    evaluations: int = 0
    runtime_sec: float = math.nan
    feasible_evaluations: int = 0


class KBRKGASolver:
    """Knowledge-based BRKGA adapted to the patient-level instances."""

    def __init__(
        self,
        inst: Dict,
        pop_size: int = 100,
        elite_size: int = 20,
        mutant_size: int = 10,
        inheritance_prob: float = 0.70,
        max_generations: int = 100,
        time_limit: float = 120.0,
        stall_generations: int = 25,
        optimize_timing: bool = False,
        timing_dispersion: float = 0.70,
        minimum_window_fraction: float = 0.70,
        random_seed: int = 2025,
        verbose: bool = False,
    ) -> None:
        self.inst = dict(inst)
        self.c = np.asarray(inst["c"], dtype=int)
        self.iv = np.asarray(inst["Iv"], dtype=int)
        self.limits = np.asarray(inst["L"], dtype=float)
        self.H, self.J = self.c.shape
        self.I = self.J

        if int(inst["I"]) != self.I or int(inst["J"]) != self.J:
            raise ValueError("K-BRKGA requires I=J patient-level instances.")
        if len(self.iv) != self.H or self.limits.shape != (self.H,):
            raise ValueError("Invalid Iv or L dimensions.")

        self.pop_size = int(pop_size)
        self.elite_size = int(elite_size)
        self.mutant_size = int(mutant_size)
        self.inheritance_prob = float(inheritance_prob)
        self.max_generations = int(max_generations)
        self.time_limit = float(time_limit)
        self.stall_generations = int(max(1, stall_generations))
        self.optimize_timing = bool(optimize_timing)
        self.timing_dispersion = float(max(0.0, timing_dispersion))
        self.minimum_window_fraction = float(min(1.0, max(0.0, minimum_window_fraction)))
        self.verbose = bool(verbose)

        if not 0 < self.elite_size < self.pop_size:
            raise ValueError("elite_size must lie in (0, pop_size).")
        if not 0 < self.mutant_size < self.pop_size - self.elite_size:
            raise ValueError("mutant_size must leave room for offspring.")
        if not 0.5 <= self.inheritance_prob <= 1.0:
            raise ValueError("inheritance_prob must lie in [0.5,1].")

        self.rng = np.random.default_rng(int(random_seed))
        self._pathway_groups = self._build_pathway_groups()
        self._interval_counts = [max(0, self.I - int(self.iv[h]) - 1) for h in range(self.H)]
        self._timing_offsets: List[Tuple[int, int]] = []
        if self.optimize_timing:
            offset = self.J + self.H
            for count in self._interval_counts:
                self._timing_offsets.append((offset, offset + count))
                offset += count
            self._n_keys = offset
        else:
            # The cited K-BRKGA allocates patients to predefined appointment
            # slots.  The paper baseline therefore searches patient allocation
            # only and uses the common scale-stable slot template below.
            self._n_keys = self.J

        self._evaluations = 0
        self._feasible_evaluations = 0
        self._start_time = 0.0

    def _build_pathway_groups(self) -> List[np.ndarray]:
        """Group individual patients by their attended-stage pattern.

        Group order follows first appearance in the patient list.  This is a
        decoder restriction, not aggregation: every patient retains a separate
        key and a separate physical position.
        """
        groups: Dict[Tuple[int, ...], List[int]] = {}
        order: List[Tuple[int, ...]] = []
        for j in range(self.J):
            signature = tuple(int(v) for v in self.c[:, j])
            if signature not in groups:
                groups[signature] = []
                order.append(signature)
            groups[signature].append(j)
        return [np.asarray(groups[signature], dtype=int) for signature in order]

    def _decode_assignment(self, keys: np.ndarray) -> np.ndarray:
        patient_keys = np.asarray(keys[: self.J], dtype=float)
        within_group: Dict[int, List[int]] = {}
        for group_index, group in enumerate(self._pathway_groups):
            within_group[group_index] = sorted(
                (int(j) for j in group), key=lambda j: (float(patient_keys[j]), j)
            )

        assignment = np.full((self.H, self.I), -1, dtype=int)
        for h in range(self.H):
            inactive: List[int] = []
            active: List[int] = []
            for group_index, group in enumerate(self._pathway_groups):
                ordered = within_group[group_index]
                if int(self.c[h, int(group[0])]) == 1:
                    active.extend(ordered)
                else:
                    inactive.extend(ordered)
            if len(inactive) != int(self.iv[h]) or len(inactive) + len(active) != self.I:
                raise RuntimeError("Pathway-aware decoder produced an invalid stage permutation.")
            assignment[h, :] = np.asarray(inactive + active, dtype=int)
        return assignment

    def _decode_times(self, keys: np.ndarray) -> np.ndarray:
        y = np.zeros((self.H, self.I), dtype=float)
        if not self.optimize_timing:
            for h in range(self.H):
                first = int(self.iv[h])
                active_count = self.I - first
                if active_count <= 1:
                    continue
                for rank, i in enumerate(range(first, self.I)):
                    y[h, i] = float(self.limits[h]) * rank / (active_count - 1)
            return y
        horizon_keys = np.asarray(keys[self.J : self.J + self.H], dtype=float)
        for h in range(self.H):
            first = int(self.iv[h])
            active_count = self.I - first
            if active_count <= 1:
                continue
            used_fraction = self.minimum_window_fraction + (
                1.0 - self.minimum_window_fraction
            ) * float(horizon_keys[h])
            used_horizon = float(self.limits[h]) * used_fraction
            begin, end = self._timing_offsets[h]
            raw = np.asarray(keys[begin:end], dtype=float)
            # Exponential weights keep every interval positive and allow a
            # controlled departure from equal spacing.
            weights = np.exp(self.timing_dispersion * (2.0 * raw - 1.0))
            weights /= float(np.sum(weights))
            cumulative = np.concatenate(([0.0], np.cumsum(weights)))
            y[h, first : self.I] = used_horizon * cumulative
        return y

    def _assignment_to_x(self, assignment: np.ndarray) -> np.ndarray:
        x = np.zeros((self.H, self.I, self.J), dtype=float)
        for h in range(self.H):
            x[h, np.arange(self.I), assignment[h, :]] = 1.0
        return x

    def _knowledge_seed(self) -> np.ndarray:
        """Natural within-pathway order with equal full-window spacing."""
        keys = np.full(self._n_keys, 0.5, dtype=float)
        for group in self._pathway_groups:
            denominator = max(1, len(group) - 1)
            for rank, j in enumerate(group):
                keys[int(j)] = rank / denominator
        if self.optimize_timing:
            keys[self.J : self.J + self.H] = 1.0
        return keys

    def _initial_population(self) -> np.ndarray:
        population = self.rng.random((self.pop_size, self._n_keys))
        population[0, :] = self._knowledge_seed()
        # Additional structured seeds preserve the same feasible decoder while
        # exploring reversed and moderately compressed templates.
        if self.pop_size > 1:
            reversed_seed = self._knowledge_seed()
            reversed_seed[: self.J] = 1.0 - reversed_seed[: self.J]
            population[1, :] = reversed_seed
        if self.pop_size > 2 and self.optimize_timing:
            compressed_seed = self._knowledge_seed()
            compressed_seed[self.J : self.J + self.H] = 0.5
            population[2, :] = compressed_seed
        return population

    def _evaluate(self, keys: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, FixedScheduleValidation]:
        assignment = self._decode_assignment(keys)
        y = self._decode_times(keys)
        x = self._assignment_to_x(assignment)
        validation = validate_fixed_schedule(self.inst, x, y, tol=1e-6)
        self._evaluations += 1
        if validation.feasible:
            self._feasible_evaluations += 1
            return float(validation.objective), x, y, validation
        # Feasibility-first ranking.  The violation term still provides a
        # gradient among infeasible timing templates.
        violation = float(validation.max_violation)
        if not np.isfinite(violation):
            violation = 1e6
        return 1e6 + 1e4 * max(0.0, violation), x, y, validation

    def _time_exhausted(self) -> bool:
        return (time.time() - self._start_time) >= self.time_limit

    def solve(self) -> KBRKGAResult:
        self._start_time = time.time()
        population = self._initial_population()
        fitness = np.full(self.pop_size, math.inf, dtype=float)
        decoded: List[Optional[Tuple[np.ndarray, np.ndarray, FixedScheduleValidation]]] = [
            None
        ] * self.pop_size

        best_fit = math.inf
        best_data: Optional[Tuple[np.ndarray, np.ndarray, FixedScheduleValidation]] = None
        best_gen = -1

        for index in range(self.pop_size):
            if self._time_exhausted():
                break
            fit, x, y, validation = self._evaluate(population[index, :])
            fitness[index] = fit
            decoded[index] = (x, y, validation)
            if validation.feasible and fit < best_fit - 1e-10:
                best_fit, best_data, best_gen = fit, decoded[index], 0

        stall = 0
        for generation in range(1, self.max_generations + 1):
            if self._time_exhausted():
                break
            order = np.argsort(fitness)
            elite_indices = order[: self.elite_size]
            nonelite_indices = order[self.elite_size :]

            next_population = np.zeros_like(population)
            next_fitness = np.full_like(fitness, math.inf)
            next_decoded: List[Optional[Tuple[np.ndarray, np.ndarray, FixedScheduleValidation]]] = [
                None
            ] * self.pop_size
            for target, source in enumerate(elite_indices):
                next_population[target, :] = population[source, :]
                next_fitness[target] = fitness[source]
                next_decoded[target] = decoded[source]

            cursor = self.elite_size
            for _ in range(self.mutant_size):
                next_population[cursor, :] = self.rng.random(self._n_keys)
                cursor += 1
            while cursor < self.pop_size:
                elite_parent = population[int(self.rng.choice(elite_indices)), :]
                other_parent = population[int(self.rng.choice(nonelite_indices)), :]
                mask = self.rng.random(self._n_keys) < self.inheritance_prob
                child = np.where(mask, elite_parent, other_parent)
                next_population[cursor, :] = child
                cursor += 1

            improved = False
            for index in range(self.elite_size, self.pop_size):
                if self._time_exhausted():
                    break
                fit, x, y, validation = self._evaluate(next_population[index, :])
                next_fitness[index] = fit
                next_decoded[index] = (x, y, validation)
                if validation.feasible and fit < best_fit - 1e-10:
                    best_fit = fit
                    best_data = next_decoded[index]
                    best_gen = generation
                    improved = True

            population, fitness, decoded = next_population, next_fitness, next_decoded
            stall = 0 if improved else stall + 1
            if self.verbose:
                print(
                    f"[K-BRKGA] gen={generation:03d} best={best_fit:.6f} "
                    f"feasible_eval={self._feasible_evaluations} stall={stall}"
                )
            if stall >= self.stall_generations:
                break

        runtime = float(time.time() - self._start_time)
        if best_data is None:
            return KBRKGAResult(
                status="NO_FEASIBLE_SCHEDULE",
                best_gen=best_gen,
                evaluations=self._evaluations,
                runtime_sec=runtime,
                feasible_evaluations=self._feasible_evaluations,
            )
        x, y, validation = best_data
        return KBRKGAResult(
            status="FEASIBLE_HEURISTIC",
            obj=float(validation.objective),
            x=np.asarray(x, dtype=float),
            y=np.asarray(y, dtype=float),
            alpha=None if validation.alpha is None else np.asarray(validation.alpha, dtype=float),
            beta=None if validation.beta is None else np.asarray(validation.beta, dtype=float),
            best_gen=int(best_gen),
            evaluations=int(self._evaluations),
            runtime_sec=runtime,
            feasible_evaluations=int(self._feasible_evaluations),
        )

    run = solve


__all__ = ["KBRKGAResult", "KBRKGASolver"]
