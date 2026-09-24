"""Patient-level comparison of DE-MILP, LBBD, and CBBD.

The experiment deliberately avoids block aggregation.  It scales the original
permutation model with I=J and compares, under the same wall-clock budget,

* DE-MILP: the complete compact deterministic equivalent solved by Gurobi;
* LBBD: exact full-scenario lazy separation in one branch-and-bound tree;
* CBBD: exact cluster-first lazy separation in the same formulation.

Clustering never removes or reweights a scenario.  Every candidate that survives
the representative screen is checked against all remaining scenarios.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from gurobipy import GRB
except Exception:  # pragma: no cover
    GRB = None

from benders_methods import CBBDSolver, LBBDSolver
from de_milp import DeterministicEquivalentModel
from experiment_utils import build_scale_stable_instance, independent_wait_metrics, parse_ints
from fixed_schedule_validation import validate_fixed_schedule


PROTOCOL_VERSION = "EJOR-COMPUTATIONAL-PROTOCOL"
VALIDATION_REL_TOL = 1e-3
PAPER_CHECKPOINTS = (30.0, 60.0, 120.0)


@dataclass
class ExperimentResult:
    method: str
    status: str
    obj: float = math.inf
    bound: float = -math.inf
    abs_gap: float = math.inf
    rel_gap: float = math.inf
    runtime_sec: float = math.nan
    build_sec: float = math.nan
    solve_sec: float = math.nan
    certified: bool = False
    full_separation: bool = False
    max_final_violation: float = math.inf
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    callback_calls: int = 0
    accepted_incumbents: int = 0
    cuts_added: int = 0
    seed_cuts: int = 0
    aggregate_cuts: int = 0
    representative_screens: int = 0
    # ``full_scenario_screens`` is the legacy nonsentinel-pass counter for CBBD.
    # Always use total_scenario_screens for cross-method workload comparisons.
    full_scenario_screens: int = 0
    total_scenario_screens: int = 0
    full_scans: int = 0
    representative_rejections: int = 0
    batch_initial: int = 0
    batch_final: int = 0
    cluster_count: int = 0
    repair_attempts: int = 0
    repair_feasible: int = 0
    repair_submissions: int = 0
    best_repaired_objective: float = math.nan
    primal_phase_sec: float = 0.0
    primal_phase_candidate: bool = False
    primal_phase_feasible_start: bool = False
    primal_phase_objective: float = math.nan
    path_start_sec: float = 0.0
    path_start_objective: float = math.nan
    path_start_evaluations: int = 0
    path_start_swaps: int = 0
    path_start_timing_moves: int = 0
    accepted_fallback_used: bool = False
    best_accepted_objective: float = math.nan
    num_vars: int = 0
    num_constrs: int = 0
    num_genconstrs: int = 0
    first_feasible_sec: float = math.nan
    progress: Dict[float, Tuple[float, float, float]] = field(default_factory=dict)
    incumbent_trace: List[Tuple[float, float]] = field(default_factory=list)


def gaps(obj: float, bound: float) -> Tuple[float, float]:
    if not np.isfinite(obj) or not np.isfinite(bound):
        return math.inf, math.inf
    absolute = max(0.0, float(obj - bound))
    return absolute, absolute / max(1.0, abs(float(obj)))


def parse_checkpoints(text: str) -> Tuple[float, ...]:
    values = tuple(sorted({float(item.strip()) for item in text.split(",") if item.strip()}))
    unsupported = [value for value in values if value not in PAPER_CHECKPOINTS]
    if unsupported:
        raise ValueError(
            f"CSV schema supports paper checkpoints {PAPER_CHECKPOINTS}; got {unsupported}."
        )
    return values


def solve_de_milp(
    inst: Dict,
    budget: float,
    mip_gap: float,
    threads: int,
    output_flag: int = 0,
    aggregate_strengthening: bool = False,
    no_rel_heur_time: float = 0.0,
    progress_checkpoints: Tuple[float, ...] = PAPER_CHECKPOINTS,
) -> ExperimentResult:
    """Solve the complete compact DE under a total build-plus-solve budget."""
    t0 = time.time()
    solver = DeterministicEquivalentModel(
        s=inst["s"],
        tau=inst["tau"],
        psi=inst["psi"],
        c=inst["c"],
        r_d=inst["r_d"],
        r_s=inst["r_s"],
        Iv=inst["Iv"],
        L=inst["L"],
        p=inst["p"],
        bigM=inst["bigM"],
        time_limit=budget,
        mip_gap=mip_gap,
        output_flag=output_flag,
        threads=threads,
        seed=inst["seed_service"],
    )
    model = solver.build(
        defer_waiting_rows=False,
        aggregate_strengthening=aggregate_strengthening,
        no_rel_heur_time=no_rel_heur_time,
        # The direct model uses a deterministic natural-order feasible start.
        # This is formulation-independent and contains no K-BRKGA solution.
        complete_warm_start=True,
    )
    model.update()
    build_sec = float(time.time() - t0)
    remaining = float(budget) - build_sec
    if remaining <= 0.0:
        return ExperimentResult(
            method="DE-MILP",
            status="BUILD_TIME_LIMIT",
            runtime_sec=float(time.time() - t0),
            build_sec=build_sec,
            solve_sec=0.0,
            num_vars=int(model.NumVars),
            num_constrs=int(model.NumConstrs),
            num_genconstrs=int(model.NumGenConstrs),
            aggregate_cuts=int(getattr(solver, "aggregate_cuts", 0)),
        )
    # Reserve five seconds for model finalization and solver return so that the
    # reported runtime follows the common wall-clock limit.
    direct_tail_reserve = 5.0
    model.Params.TimeLimit = max(0.01, remaining - direct_tail_reserve)
    solve_start = time.time()
    progress: Dict[float, Tuple[float, float, float]] = {}
    first_feasible_sec = math.nan
    incumbent_trace: List[Tuple[float, float]] = []
    best_callback_objective = math.inf

    def clean_callback_value(value: float) -> float:
        value = float(value)
        if not np.isfinite(value) or abs(value) >= 1e90:
            return math.nan
        return value

    def progress_callback(cb_model, where) -> None:
        nonlocal first_feasible_sec, best_callback_objective
        elapsed = float(time.time() - t0)
        if where == GRB.Callback.MIPSOL:
            candidate = clean_callback_value(
                cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
            )
            if not np.isfinite(first_feasible_sec):
                first_feasible_sec = elapsed
            if np.isfinite(candidate) and candidate < best_callback_objective - 1e-10:
                best_callback_objective = candidate
                incumbent_trace.append((elapsed, candidate))
        if where != GRB.Callback.MIP:
            return
        objective = clean_callback_value(cb_model.cbGet(GRB.Callback.MIP_OBJBST))
        bound = clean_callback_value(cb_model.cbGet(GRB.Callback.MIP_OBJBND))
        _, rel_gap = gaps(objective, bound)
        if not np.isfinite(rel_gap):
            rel_gap = math.nan
        for checkpoint in progress_checkpoints:
            if checkpoint not in progress and elapsed >= checkpoint:
                progress[checkpoint] = (objective, bound, rel_gap)

    model.optimize(progress_callback)
    if GRB is not None and model.Status == GRB.INF_OR_UNBD:
        remaining = float(budget) - (time.time() - t0)
        if remaining > 0.05:
            model.Params.DualReductions = 0
            model.Params.TimeLimit = max(0.01, remaining - direct_tail_reserve)
            model.reset()
            model.optimize(progress_callback)
    runtime = float(time.time() - t0)
    result = ExperimentResult(
        method="DE-MILP",
        status=f"STATUS_{model.Status}" if model.SolCount > 0 else f"NO_INCUMBENT_{model.Status}",
        runtime_sec=runtime,
        build_sec=build_sec,
        solve_sec=float(time.time() - solve_start),
        full_separation=True,
        max_final_violation=0.0,
        num_vars=int(model.NumVars),
        num_constrs=int(model.NumConstrs),
        num_genconstrs=int(model.NumGenConstrs),
        aggregate_cuts=int(getattr(solver, "aggregate_cuts", 0)),
        first_feasible_sec=first_feasible_sec,
        progress=dict(progress),
        incumbent_trace=list(incumbent_trace),
    )
    if model.SolCount <= 0:
        return result
    x, y, _, _ = solver.extract()
    result.x, result.y = x, y
    result.obj, result.bound = float(model.ObjVal), float(model.ObjBound)
    result.abs_gap, result.rel_gap = gaps(result.obj, result.bound)
    result.certified = bool(
        model.Status == GRB.OPTIMAL or result.rel_gap <= float(mip_gap) + 1e-12
    )
    if not np.isfinite(result.first_feasible_sec):
        result.first_feasible_sec = runtime
    for checkpoint in progress_checkpoints:
        if checkpoint not in result.progress:
            result.progress[checkpoint] = (
                result.obj,
                result.bound,
                result.rel_gap,
            )
    return result


def solve_benders(
    inst: Dict,
    method: str,
    budget: float,
    mip_gap: float,
    threads: int,
    output_flag: int = 0,
    cluster_count: Optional[int] = None,
    seed_scenarios: Optional[int] = None,
    cuts_per_callback: Optional[int] = None,
    aggregate_strengthening: bool = False,
    incumbent_repair: bool = True,
    no_rel_heur_time: float = 0.0,
    primal_phase_time: float = 0.0,
    path_start_time: float = 5.0,
    path_start_passes: int = 8,
    progress_checkpoints: Tuple[float, ...] = PAPER_CHECKPOINTS,
) -> ExperimentResult:
    common = dict(
        s=inst["s"],
        tau=inst["tau"],
        psi=inst["psi"],
        c=inst["c"],
        r_d=inst["r_d"],
        r_s=inst["r_s"],
        Iv=inst["Iv"],
        L=inst["L"],
        p=inst["p"],
        bigM=inst["bigM"],
        tol=1e-6,
        time_limit=budget,
        mip_gap=mip_gap,
        output_flag=output_flag,
        threads=threads,
        seed=inst["seed_service"],
        cluster_count=cluster_count,
        seed_scenarios=seed_scenarios,
        aggregate_strengthening=aggregate_strengthening,
        incumbent_repair=incumbent_repair,
        no_rel_heur_time=no_rel_heur_time,
        primal_phase_time=primal_phase_time,
        path_start_time=path_start_time,
        path_start_passes=path_start_passes,
        # The solver start remains the deterministic natural schedule; the
        # pathway repair is retained separately as an independently audited
        # feasible incumbent and never comes from K-BRKGA.
        complete_warm_start=True,
        progress_checkpoints=progress_checkpoints,
    )
    if cuts_per_callback is not None:
        common["cuts_per_callback"] = int(cuts_per_callback)
    solver = LBBDSolver(**common) if method == "LBBD" else CBBDSolver(**common)
    raw = solver.solve()
    abs_gap, rel_gap = gaps(raw.obj, raw.bound)
    return ExperimentResult(
        method=method,
        status=raw.status,
        obj=raw.obj,
        bound=raw.bound,
        abs_gap=abs_gap,
        rel_gap=rel_gap,
        runtime_sec=raw.runtime_sec,
        build_sec=raw.build_sec,
        solve_sec=raw.solve_sec,
        certified=raw.certified,
        full_separation=raw.full_separation,
        max_final_violation=raw.max_final_violation,
        x=raw.x,
        y=raw.y,
        callback_calls=raw.callback_calls,
        accepted_incumbents=raw.accepted_incumbents,
        cuts_added=raw.lazy_cuts,
        seed_cuts=raw.seed_cuts,
        aggregate_cuts=raw.aggregate_cuts,
        representative_screens=raw.representative_screens,
        full_scenario_screens=raw.full_scenario_screens,
        total_scenario_screens=(
            raw.representative_screens + raw.full_scenario_screens
        ),
        full_scans=raw.full_scans,
        representative_rejections=raw.representative_rejections,
        batch_initial=raw.batch_initial,
        batch_final=raw.batch_final,
        cluster_count=raw.cluster_count,
        repair_attempts=raw.repair_attempts,
        repair_feasible=raw.repair_feasible,
        repair_submissions=raw.repair_submissions,
        best_repaired_objective=(
            raw.best_repaired_objective
            if np.isfinite(raw.best_repaired_objective)
            else math.nan
        ),
        primal_phase_sec=raw.primal_phase_sec,
        primal_phase_candidate=raw.primal_phase_candidate,
        primal_phase_feasible_start=raw.primal_phase_feasible_start,
        primal_phase_objective=raw.primal_phase_objective,
        path_start_sec=raw.path_start_sec,
        path_start_objective=raw.path_start_objective,
        path_start_evaluations=raw.path_start_evaluations,
        path_start_swaps=raw.path_start_swaps,
        path_start_timing_moves=raw.path_start_timing_moves,
        accepted_fallback_used=raw.accepted_fallback_used,
        best_accepted_objective=(
            raw.best_accepted_objective
            if np.isfinite(raw.best_accepted_objective)
            else math.nan
        ),
        num_vars=raw.num_vars,
        num_constrs=raw.num_constrs,
        num_genconstrs=raw.num_genconstrs,
        first_feasible_sec=raw.first_feasible_sec,
        progress=dict(raw.progress),
        incumbent_trace=list(raw.incumbent_trace),
    )


FIELDS = [
    "protocol_version",
    "checkpoints",
    "experiment",
    "instance_formulation",
    "solver_formulation",
    "window_mode",
    "buffer_z",
    "terminal_shortfall",
    "pattern",
    "I",
    "J",
    "K",
    "physical_h1",
    "physical_h2",
    "physical_h3",
    "visits",
    "L1",
    "L2",
    "L3",
    "rep",
    "seed_service",
    "seed_lateness",
    "threads",
    "budget_sec",
    "target_gap",
    "no_rel_heur_time",
    "primal_phase_time",
    "warm_start_mode",
    "path_start_time",
    "path_start_passes",
    "method",
    "status",
    "full_separation",
    "certified",
    "train_obj",
    "validated_obj",
    "validation_status",
    "validation_abs_diff",
    "validation_rel_diff",
    "validation_max_violation",
    "validation_sec",
    "objective_consistent",
    "protocol_pass",
    "objective_per_visit",
    "train_bound",
    "abs_gap",
    "rel_gap",
    "runtime_sec",
    "build_sec",
    "solve_sec",
    "first_feasible_sec",
    "incumbent_trace_json",
    "obj_at_30",
    "bound_at_30",
    "gap_at_30",
    "obj_at_60",
    "bound_at_60",
    "gap_at_60",
    "obj_at_120",
    "bound_at_120",
    "gap_at_120",
    "max_final_violation",
    "callback_calls",
    "accepted_incumbents",
    "cuts_added",
    "seed_cuts",
    "aggregate_cuts",
    "representative_screens",
    "full_scenario_screens",
    "total_scenario_screens",
    "full_scans",
    "representative_rejections",
    "batch_initial",
    "batch_final",
    "cluster_count",
    "repair_attempts",
    "repair_feasible",
    "repair_submissions",
    "best_repaired_objective",
    "primal_phase_sec",
    "primal_phase_candidate",
    "primal_phase_feasible_start",
    "primal_phase_objective",
    "path_start_sec",
    "path_start_objective",
    "path_start_evaluations",
    "path_start_swaps",
    "path_start_timing_moves",
    "accepted_fallback_used",
    "best_accepted_objective",
    "num_vars",
    "num_constrs",
    "num_genconstrs",
    "test_K",
    "test_mean_direct",
    "test_p95_direct",
    "test_direct_exceed_rate",
    "test_mean_indirect",
    "test_p95_indirect",
    "test_indirect_exceed_rate",
]


def parse_methods(text: str) -> list[str]:
    aliases = {
        "de": "DE-MILP",
        "de-milp": "DE-MILP",
        "lbbd": "LBBD",
        "cbbd": "CBBD",
    }
    methods: list[str] = []
    for item in text.split(","):
        key = item.strip().lower()
        if key not in aliases:
            raise ValueError(f"Unknown method {item!r}.")
        if aliases[key] not in methods:
            methods.append(aliases[key])
    if not methods:
        raise ValueError("At least one method is required.")
    return methods


def existing_keys(path: str) -> set[Tuple[str, ...]]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        return {
            (
                row.get("I", ""),
                row.get("J", ""),
                row.get("K", ""),
                row.get("rep", ""),
                row.get("method", ""),
                row.get("buffer_z", ""),
                row.get("terminal_shortfall", ""),
                row.get("budget_sec", ""),
                row.get("target_gap", ""),
                row.get("no_rel_heur_time", ""),
                row.get("primal_phase_time", ""),
                row.get("warm_start_mode", ""),
                row.get("path_start_time", ""),
                row.get("path_start_passes", ""),
                row.get("protocol_version", "legacy"),
                row.get("checkpoints", ""),
            )
            for row in csv.DictReader(handle)
        }


def token(value: float) -> str:
    return f"{float(value):.8g}"


def append_row(path: str, row: Dict) -> None:
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    if not new_file:
        with open(path, "r", newline="", encoding="utf-8-sig") as existing:
            existing_fields = csv.DictReader(existing).fieldnames
        if existing_fields != FIELDS:
            raise ValueError(
                f"Output schema mismatch for {path!r}; use a new output file for {PROTOCOL_VERSION}."
            )
    with open(path, "a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--J",
        default="8,12,16,20",
        help="I=J patient counts; each must be a multiple of 4",
    )
    parser.add_argument(
        "--primal-phase-time",
        type=float,
        default=0.0,
        help="LBBD/CBBD seeded restricted-master seconds before exact separation",
    )
    parser.add_argument("--path-start-time", type=float, default=5.0)
    parser.add_argument("--path-start-passes", type=int, default=8)
    parser.add_argument("--K", default="100", help="Training scenario count")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument(
        "--rep-list",
        default="",
        help="Optional comma-separated replication indices; otherwise use 0..reps-1.",
    )
    parser.add_argument("--methods", default="DE-MILP,LBBD,CBBD")
    parser.add_argument("--budget", type=float, default=120.0, help="Build-plus-solve seconds per method")
    parser.add_argument("--mip-gap", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--test-K", type=int, default=1000)
    parser.add_argument(
        "--checkpoints",
        default="30,60,120",
        help="Comma-separated total wall-clock checkpoints; supported: 30,60,120",
    )
    parser.add_argument("--buffer-z", type=float, default=0.5)
    parser.add_argument("--terminal-shortfall", type=float, default=0.6)
    parser.add_argument("--seed-base", type=int, default=12065)
    parser.add_argument("--cluster-count", type=int, default=None)
    parser.add_argument("--seed-scenarios", type=int, default=None)
    parser.add_argument("--cuts-per-callback", type=int, default=None)
    parser.add_argument("--aggregate-strengthening", action="store_true")
    repair_group = parser.add_mutually_exclusive_group()
    repair_group.add_argument("--incumbent-repair", dest="incumbent_repair", action="store_true")
    repair_group.add_argument("--no-incumbent-repair", dest="incumbent_repair", action="store_false")
    parser.set_defaults(incumbent_repair=True)
    parser.add_argument(
        "--no-rel-heur-time",
        type=float,
        default=0.0,
        help="Common Gurobi NoRel primal-search seconds before root processing",
    )
    parser.add_argument("--output-flag", type=int, default=0)
    parser.add_argument("--output", default="algorithm_comparison_results.csv")
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    checkpoints = parse_checkpoints(args.checkpoints)
    checkpoint_token = ",".join(f"{value:g}" for value in checkpoints)
    done = existing_keys(args.output)
    rep_indices = (
        [int(item.strip()) for item in args.rep_list.split(",") if item.strip()]
        if args.rep_list.strip()
        else list(range(args.reps))
    )
    if not rep_indices or any(rep < 0 for rep in rep_indices):
        raise ValueError("Replication indices must be nonnegative.")
    for J in parse_ints(args.J):
        if J % 4 != 0:
            raise ValueError("The documented coupled 50/25/25 design requires J to be a multiple of 4.")
        I = J
        for K in parse_ints(args.K):
            for rep in rep_indices:
                seed_service = int(args.seed_base + 100 * rep)
                seed_lateness = seed_service + 1
                inst = build_scale_stable_instance(
                    J=J,
                    K=K,
                    seed_service=seed_service,
                    seed_lateness=seed_lateness,
                    buffer_z=args.buffer_z,
                    terminal_shortfall=args.terminal_shortfall,
                )
                visits = int(sum(inst["physical"]))
                for method in methods:
                    key = (
                        str(I),
                        str(J),
                        str(K),
                        str(rep),
                        method,
                        token(args.buffer_z),
                        token(args.terminal_shortfall),
                        token(args.budget),
                        token(args.mip_gap),
                        token(args.no_rel_heur_time),
                        token(args.primal_phase_time),
                        "complete_natural" if method == "DE-MILP" else "pathwise_repair",
                        token(0.0 if method == "DE-MILP" else args.path_start_time),
                        str(0 if method == "DE-MILP" else args.path_start_passes),
                        PROTOCOL_VERSION,
                        checkpoint_token,
                    )
                    if key in done:
                        continue
                    if method == "DE-MILP":
                        result = solve_de_milp(
                            inst,
                            args.budget,
                            args.mip_gap,
                            args.threads,
                            args.output_flag,
                            aggregate_strengthening=args.aggregate_strengthening,
                            no_rel_heur_time=args.no_rel_heur_time,
                            progress_checkpoints=checkpoints,
                        )
                        solver_formulation = "compact_patient_completion_with_cross_stage_precedence"
                    else:
                        result = solve_benders(
                            inst=inst,
                            method=method,
                            budget=args.budget,
                            mip_gap=args.mip_gap,
                            threads=args.threads,
                            output_flag=args.output_flag,
                            cluster_count=args.cluster_count,
                            seed_scenarios=args.seed_scenarios,
                            cuts_per_callback=args.cuts_per_callback,
                            aggregate_strengthening=args.aggregate_strengthening,
                            incumbent_repair=args.incumbent_repair,
                            no_rel_heur_time=args.no_rel_heur_time,
                            primal_phase_time=args.primal_phase_time,
                            path_start_time=args.path_start_time,
                            path_start_passes=args.path_start_passes,
                            progress_checkpoints=checkpoints,
                        )
                        solver_formulation = (
                            "compact_physical_precedence_lazy_interstage_waiting"
                        )

                    metrics = {
                        "test_mean_direct": math.nan,
                        "test_p95_direct": math.nan,
                        "test_direct_exceed_rate": math.nan,
                        "test_mean_indirect": math.nan,
                        "test_p95_indirect": math.nan,
                        "test_indirect_exceed_rate": math.nan,
                    }
                    if result.x is not None and result.y is not None:
                        metrics = independent_wait_metrics(
                            inst,
                            result.x,
                            result.y,
                            test_K=args.test_K,
                            seed_service=seed_service + 10_000_000,
                            seed_lateness=seed_lateness + 20_000_000,
                            enforce_cross_stage_precedence=True,
                        )
                    validation_status = "NO_SCHEDULE"
                    validated_obj = math.nan
                    validation_abs_diff = math.nan
                    validation_rel_diff = math.nan
                    validation_max_violation = math.inf
                    validation_start = time.time()
                    if result.x is not None and result.y is not None:
                        validation = validate_fixed_schedule(
                            inst,
                            result.x,
                            result.y,
                            tol=1e-6,
                        )
                        validation_status = validation.status
                        validation_max_violation = validation.max_violation
                        if validation.feasible:
                            validated_obj = validation.objective
                            validation_abs_diff = abs(float(result.obj) - validated_obj)
                            validation_rel_diff = validation_abs_diff / max(
                                1.0, abs(validated_obj)
                            )
                    validation_sec = float(time.time() - validation_start)
                    # The validator analytically re-optimizes the continuous
                    # tolerance credits for the returned discrete schedule.
                    # A slightly lower validated objective is therefore a
                    # legitimate recourse recovery, not an inconsistency.  We
                    # retain both values and apply a prespecified 0.1% audit
                    # cap to detect extraction/model mismatches.
                    audit_scale = max(1.0, abs(float(validated_obj)))
                    objective_consistent = bool(
                        np.isfinite(validation_abs_diff)
                        and validation_rel_diff <= VALIDATION_REL_TOL
                        and validated_obj
                        <= float(result.obj) + max(1e-5, 1e-4 * audit_scale)
                    )
                    protocol_pass = bool(
                        validation_status == "PASS"
                        and objective_consistent
                        and result.full_separation
                        and result.runtime_sec
                        <= float(args.budget) + max(1.0, 0.02 * float(args.budget))
                    )
                    progress_values = {}
                    for checkpoint in PAPER_CHECKPOINTS:
                        objective, bound, rel_gap = result.progress.get(
                            checkpoint, (math.nan, math.nan, math.nan)
                        )
                        label = f"{int(checkpoint)}"
                        progress_values[f"obj_at_{label}"] = objective
                        progress_values[f"bound_at_{label}"] = bound
                        progress_values[f"gap_at_{label}"] = rel_gap
                    row = {
                        "protocol_version": PROTOCOL_VERSION,
                        "checkpoints": checkpoint_token,
                        "experiment": "patient_level_algorithm_comparison",
                        "instance_formulation": "I_equals_J_coupled_50_25_25_no_blocks",
                        "solver_formulation": solver_formulation,
                        "window_mode": inst["window_mode"],
                        "buffer_z": token(args.buffer_z),
                        "terminal_shortfall": token(args.terminal_shortfall),
                        "pattern": inst["pattern"],
                        "I": I,
                        "J": J,
                        "K": K,
                        "physical_h1": inst["physical"][0],
                        "physical_h2": inst["physical"][1],
                        "physical_h3": inst["physical"][2],
                        "visits": visits,
                        "L1": inst["L"][0],
                        "L2": inst["L"][1],
                        "L3": inst["L"][2],
                        "rep": rep,
                        "seed_service": seed_service,
                        "seed_lateness": seed_lateness,
                        "threads": args.threads,
                        "budget_sec": token(args.budget),
                        "target_gap": token(args.mip_gap),
                        "no_rel_heur_time": token(args.no_rel_heur_time),
                        "primal_phase_time": token(args.primal_phase_time),
                        "warm_start_mode": (
                            "complete_natural"
                            if method == "DE-MILP"
                            else "pathwise_repair"
                        ),
                        "path_start_time": (
                            0.0 if method == "DE-MILP" else args.path_start_time
                        ),
                        "path_start_passes": (
                            0 if method == "DE-MILP" else args.path_start_passes
                        ),
                        "method": method,
                        "status": result.status,
                        "full_separation": int(result.full_separation),
                        "certified": int(result.certified),
                        "train_obj": result.obj,
                        "validated_obj": validated_obj,
                        "validation_status": validation_status,
                        "validation_abs_diff": validation_abs_diff,
                        "validation_rel_diff": validation_rel_diff,
                        "validation_max_violation": validation_max_violation,
                        "validation_sec": validation_sec,
                        "objective_consistent": int(objective_consistent),
                        "protocol_pass": int(protocol_pass),
                        "objective_per_visit": validated_obj / visits if np.isfinite(validated_obj) else math.nan,
                        "train_bound": result.bound,
                        "abs_gap": result.abs_gap,
                        "rel_gap": result.rel_gap,
                        "runtime_sec": result.runtime_sec,
                        "build_sec": result.build_sec,
                        "solve_sec": result.solve_sec,
                        "first_feasible_sec": result.first_feasible_sec,
                        "incumbent_trace_json": json.dumps(
                            result.incumbent_trace, separators=(",", ":")
                        ),
                        "max_final_violation": result.max_final_violation,
                        "callback_calls": result.callback_calls,
                        "accepted_incumbents": result.accepted_incumbents,
                        "cuts_added": result.cuts_added,
                        "seed_cuts": result.seed_cuts,
                        "aggregate_cuts": result.aggregate_cuts,
                        "representative_screens": result.representative_screens,
                        "full_scenario_screens": result.full_scenario_screens,
                        "total_scenario_screens": result.total_scenario_screens,
                        "full_scans": result.full_scans,
                        "representative_rejections": result.representative_rejections,
                        "batch_initial": result.batch_initial,
                        "batch_final": result.batch_final,
                        "cluster_count": result.cluster_count,
                        "repair_attempts": result.repair_attempts,
                        "repair_feasible": result.repair_feasible,
                        "repair_submissions": result.repair_submissions,
                        "best_repaired_objective": result.best_repaired_objective,
                        "primal_phase_sec": result.primal_phase_sec,
                        "primal_phase_candidate": int(result.primal_phase_candidate),
                        "primal_phase_feasible_start": int(
                            result.primal_phase_feasible_start
                        ),
                        "primal_phase_objective": result.primal_phase_objective,
                        "path_start_sec": result.path_start_sec,
                        "path_start_objective": result.path_start_objective,
                        "path_start_evaluations": result.path_start_evaluations,
                        "path_start_swaps": result.path_start_swaps,
                        "path_start_timing_moves": result.path_start_timing_moves,
                        "accepted_fallback_used": int(
                            result.accepted_fallback_used
                        ),
                        "best_accepted_objective": result.best_accepted_objective,
                        "num_vars": result.num_vars,
                        "num_constrs": result.num_constrs,
                        "num_genconstrs": result.num_genconstrs,
                        "test_K": args.test_K,
                        **progress_values,
                        **metrics,
                    }
                    append_row(args.output, row)
                    done.add(key)
                    print(
                        f"I=J={J} K={K} rep={rep} {method}: "
                        f"status={result.status} obj={result.obj:.6g} "
                        f"validated={validated_obj:.6g} audit={validation_status} "
                        f"protocol={'PASS' if protocol_pass else 'FAIL'} "
                        f"gap={result.rel_gap:.3g} cuts={result.cuts_added} "
                        f"time={result.runtime_sec:.2f}s"
                    )


if __name__ == "__main__":
    main()
