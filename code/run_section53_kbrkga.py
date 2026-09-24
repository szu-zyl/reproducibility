"""Run the independent adapted K-BRKGA benchmark.

The exact-method CSV files are left untouched.  This runner regenerates the
same instances from the documented seeds, applies the common 120-second cap,
and validates every returned schedule with the independent evaluator.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Dict, Iterable, List, Tuple

import numpy as np

from kbrkga import KBRKGASolver
from experiment_utils import build_scale_stable_instance, independent_wait_metrics
from fixed_schedule_validation import validate_fixed_schedule


FIELDS = [
    "protocol_version",
    "method",
    "J",
    "K",
    "rep",
    "seed_service",
    "seed_lateness",
    "algorithm_seed",
    "budget_sec",
    "pop_size",
    "elite_size",
    "mutant_size",
    "inheritance_prob",
    "max_generations",
    "stall_generations",
    "timing_mode",
    "status",
    "train_obj",
    "validated_obj",
    "validation_status",
    "validation_max_violation",
    "runtime_sec",
    "best_gen",
    "evaluations",
    "feasible_evaluations",
    "test_K",
    "test_mean_direct",
    "test_p95_direct",
    "test_direct_exceed_rate",
    "test_mean_indirect",
    "test_p95_indirect",
    "test_indirect_exceed_rate",
]


def parse_cells(text: str) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        left, right = token.split(":", 1)
        cell = (int(left), int(right))
        if cell not in cells:
            cells.append(cell)
    return cells


def existing_keys(path: str) -> set[Tuple[int, int, int]]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        return {
            (int(row["J"]), int(row["K"]), int(row["rep"]))
            for row in csv.DictReader(handle)
        }


def append_row(path: str, row: Dict) -> None:
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})
        handle.flush()


def mean(values: Iterable[float]) -> float:
    data = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(data)) if data else math.nan


def build_summary(raw_path: str, summary_path: str) -> None:
    with open(raw_path, "r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    grouped: Dict[Tuple[int, int], List[Dict]] = {}
    for row in rows:
        grouped.setdefault((int(row["J"]), int(row["K"])), []).append(row)
    fields = [
        "J",
        "K",
        "n",
        "validated_mean",
        "validated_std",
        "validated_ci95_half",
        "runtime_mean",
        "runtime_std",
        "feasible_runs",
        "evaluations_mean",
        "feasible_evaluations_mean",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (J, K), group in sorted(grouped.items()):
            objectives = np.asarray(
                [float(row["validated_obj"]) for row in group if row["validated_obj"]],
                dtype=float,
            )
            objectives = objectives[np.isfinite(objectives)]
            runtimes = np.asarray([float(row["runtime_sec"]) for row in group], dtype=float)
            std = float(np.std(objectives, ddof=1)) if objectives.size > 1 else 0.0
            # t_0.975,9 for the documented ten replications.
            tcrit = 2.262 if objectives.size == 10 else 1.96
            writer.writerow(
                {
                    "J": J,
                    "K": K,
                    "n": int(objectives.size),
                    "validated_mean": mean(objectives),
                    "validated_std": std,
                    "validated_ci95_half": (
                        tcrit * std / math.sqrt(objectives.size)
                        if objectives.size
                        else math.nan
                    ),
                    "runtime_mean": mean(runtimes),
                    "runtime_std": (
                        float(np.std(runtimes, ddof=1)) if runtimes.size > 1 else 0.0
                    ),
                    "feasible_runs": sum(
                        row["validation_status"] == "PASS" for row in group
                    ),
                    "evaluations_mean": mean(float(row["evaluations"]) for row in group),
                    "feasible_evaluations_mean": mean(
                        float(row["feasible_evaluations"]) for row in group
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cells",
        default="8:100,12:100,16:100,20:100",
    )
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=12065)
    parser.add_argument("--budget", type=float, default=120.0)
    parser.add_argument("--test-K", type=int, default=1000)
    parser.add_argument("--pop-size", type=int, default=100)
    parser.add_argument("--elite-size", type=int, default=20)
    parser.add_argument("--mutant-size", type=int, default=10)
    parser.add_argument("--inheritance-prob", type=float, default=0.70)
    parser.add_argument("--max-generations", type=int, default=100)
    parser.add_argument("--stall-generations", type=int, default=25)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", default="kbrkga_benchmark_results.csv")
    parser.add_argument("--summary", default="kbrkga_benchmark_summary.csv")
    args = parser.parse_args()

    done = existing_keys(args.output)
    for J, K in parse_cells(args.cells):
        for rep in range(args.reps):
            key = (J, K, rep)
            if key in done:
                continue
            seed_service = int(args.seed_base + 100 * rep)
            seed_lateness = seed_service + 1
            algorithm_seed = seed_service + 30_000_000
            inst = build_scale_stable_instance(
                J=J,
                K=K,
                seed_service=seed_service,
                seed_lateness=seed_lateness,
                buffer_z=0.5,
                terminal_shortfall=0.6,
            )
            solver = KBRKGASolver(
                inst,
                pop_size=args.pop_size,
                elite_size=args.elite_size,
                mutant_size=args.mutant_size,
                inheritance_prob=args.inheritance_prob,
                max_generations=args.max_generations,
                time_limit=args.budget,
                stall_generations=args.stall_generations,
                optimize_timing=False,
                random_seed=algorithm_seed,
                verbose=args.verbose,
            )
            result = solver.solve()
            validated_obj = math.nan
            validation_status = "NO_SCHEDULE"
            validation_max_violation = math.inf
            metrics = {
                "test_mean_direct": math.nan,
                "test_p95_direct": math.nan,
                "test_direct_exceed_rate": math.nan,
                "test_mean_indirect": math.nan,
                "test_p95_indirect": math.nan,
                "test_indirect_exceed_rate": math.nan,
            }
            if result.x is not None and result.y is not None:
                validation = validate_fixed_schedule(inst, result.x, result.y, tol=1e-6)
                validation_status = validation.status
                validation_max_violation = validation.max_violation
                if validation.feasible:
                    validated_obj = validation.objective
                metrics = independent_wait_metrics(
                    inst,
                    result.x,
                    result.y,
                    test_K=args.test_K,
                    seed_service=seed_service + 10_000_000,
                    seed_lateness=seed_lateness + 20_000_000,
                    enforce_cross_stage_precedence=True,
                )
            row = {
                    "protocol_version": "EJOR-KBRKGA-PROTOCOL",
                "method": "K-BRKGA",
                "J": J,
                "K": K,
                "rep": rep,
                "seed_service": seed_service,
                "seed_lateness": seed_lateness,
                "algorithm_seed": algorithm_seed,
                "budget_sec": args.budget,
                "pop_size": args.pop_size,
                "elite_size": args.elite_size,
                "mutant_size": args.mutant_size,
                "inheritance_prob": args.inheritance_prob,
                "max_generations": args.max_generations,
                "stall_generations": args.stall_generations,
                "timing_mode": "fixed_scale_stable_slots",
                "status": result.status,
                "train_obj": result.obj,
                "validated_obj": validated_obj,
                "validation_status": validation_status,
                "validation_max_violation": validation_max_violation,
                "runtime_sec": result.runtime_sec,
                "best_gen": result.best_gen,
                "evaluations": result.evaluations,
                "feasible_evaluations": result.feasible_evaluations,
                "test_K": args.test_K,
                **metrics,
            }
            append_row(args.output, row)
            done.add(key)
            print(
                f"J={J} K={K} rep={rep} K-BRKGA: "
                f"status={validation_status} validated={validated_obj:.6g} "
                f"gen={result.best_gen} evals={result.evaluations} "
                f"time={result.runtime_sec:.2f}s",
                flush=True,
            )
            build_summary(args.output, args.summary)
    build_summary(args.output, args.summary)


if __name__ == "__main__":
    main()
