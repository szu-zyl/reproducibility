"""Run the Section 5.2 integrated-versus-stage-wise comparison.

DE-MILP is compared with a stage-wise MILP (SS-MILP) and a Bailey--Welch-style
spacing rule (SS-BW) on the same J=4 instances.  All schedules are evaluated by
the same analytic fixed-schedule validator so that the comparison isolates the
value of integrated planning.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np

from scenario_generation import generate_all_scenarios
from stagewise_baselines import StageByStageHeuristic
from model_utils import compute_data_driven_bigM
from fixed_schedule_validation import FixedScheduleValidation, validate_fixed_schedule
from run_section53_exact import solve_de_milp


RAW_FIELDS = [
    "protocol_version", "J", "K", "rep", "seed_service", "seed_lateness",
    "method", "status", "certified", "validated_obj", "alpha_sum", "beta_sum",
    "avg_direct_burden", "avg_indirect_burden", "avg_total_burden",
    "validation_max_violation", "runtime_sec",
]

PATIENT_FIELDS = [
    "protocol_version", "J", "K", "rep", "seed_service", "seed_lateness",
    "method", "patient", "attended_stages", "direct_burden",
    "indirect_burden", "total_burden",
]


def generate_base_scenarios(count: int, c: np.ndarray, seed_service: int):
    return generate_all_scenarios(
        numSamples=count,
        numStages=3,
        numPositions=4,
        numPatients=4,
        c=c,
        seed=seed_service,
    )


def append_row(path: str, fields: List[str], row: Dict) -> None:
    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})
        handle.flush()


def existing_keys(path: str) -> set[Tuple[int, int, str]]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        return {
            (int(row["rep"]), int(row["K"]), row["method"])
            for row in csv.DictReader(handle)
        }


def finite(values: Iterable[float]) -> np.ndarray:
    data = np.asarray([float(value) for value in values], dtype=float)
    return data[np.isfinite(data)]


def build_summary(raw_path: str, summary_path: str) -> None:
    with open(raw_path, "r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    grouped: Dict[Tuple[int, str], List[Dict]] = {}
    for row in rows:
        grouped.setdefault((int(row["K"]), row["method"]), []).append(row)
    fields = [
        "K", "method", "n", "obj_mean", "obj_std", "obj_ci95_half",
        "alpha_mean", "alpha_std", "beta_mean", "beta_std",
        "direct_burden_mean", "direct_burden_std",
        "indirect_burden_mean", "indirect_burden_std",
        "runtime_mean", "runtime_std", "certified_runs", "validation_pass_runs",
    ]
    t95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
           6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

    def avg(values: np.ndarray) -> float:
        return float(np.mean(values)) if values.size else math.nan

    def std(values: np.ndarray) -> float:
        return float(np.std(values, ddof=1)) if values.size > 1 else 0.0

    with open(summary_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (K, method), group in sorted(grouped.items()):
            obj = finite(row["validated_obj"] for row in group)
            alpha = finite(row["alpha_sum"] for row in group)
            beta = finite(row["beta_sum"] for row in group)
            direct = finite(row["avg_direct_burden"] for row in group)
            indirect = finite(row["avg_indirect_burden"] for row in group)
            runtime = finite(row["runtime_sec"] for row in group)
            critical = t95.get(max(1, obj.size - 1), 1.96)
            ci = critical * std(obj) / math.sqrt(obj.size) if obj.size > 1 else 0.0
            writer.writerow({
                "K": K, "method": method, "n": int(obj.size),
                "obj_mean": avg(obj), "obj_std": std(obj), "obj_ci95_half": ci,
                "alpha_mean": avg(alpha), "alpha_std": std(alpha),
                "beta_mean": avg(beta), "beta_std": std(beta),
                "direct_burden_mean": avg(direct), "direct_burden_std": std(direct),
                "indirect_burden_mean": avg(indirect), "indirect_burden_std": std(indirect),
                "runtime_mean": avg(runtime), "runtime_std": std(runtime),
                "certified_runs": sum(int(row["certified"]) for row in group),
                "validation_pass_runs": sum(row["status"] == "PASS" for row in group),
            })


def burden_metrics(
    inst: Dict, x: np.ndarray, validation: FixedScheduleValidation
) -> Tuple[float, float, List[Dict]]:
    assignment = np.argmax(np.asarray(x), axis=2)
    direct_total = 0.0
    indirect_total = 0.0
    visits = 0
    patient_rows: List[Dict] = []
    for j in range(int(inst["J"])):
        patient_direct = 0.0
        patient_indirect = 0.0
        attended = 0
        for h in range(int(inst["H"])):
            if int(inst["c"][h, j]) != 1:
                continue
            i = int(np.where(assignment[h, :] == j)[0][0])
            d = float(inst["r_d"][j] - validation.alpha[h, i])
            q = float(inst["r_s"][j] - validation.beta[h, i])
            direct_total += d
            indirect_total += q
            patient_direct += d
            patient_indirect += q
            visits += 1
            attended += 1
        patient_rows.append({
            "patient": j, "attended_stages": attended,
            "direct_burden": patient_direct, "indirect_burden": patient_indirect,
            "total_burden": patient_direct + patient_indirect,
        })
    return direct_total / visits, indirect_total / visits, patient_rows


def repair_bw_direct_feasibility(inst: Dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Minimally delay BW appointments whose expected direct wait exceeds tolerance.

    The two-at-session-start BW rule can sit exactly on the tolerance boundary
    when only two patients attend a stage.  The repair preserves the BW order
    and shifts only the offending appointment by the smallest amount required
    by the direct-wait definition.  A
    patient's downstream readiness is the later of its exogenous arrival and
    completion at its previous attended stage.
    """
    repaired = np.asarray(y, dtype=float).copy()
    assignment = np.argmax(np.asarray(x), axis=2)
    first = np.zeros((int(inst["H"]), int(inst["J"])), dtype=int)
    follow = np.zeros_like(first)
    for j in range(int(inst["J"])):
        stages = [h for h in range(int(inst["H"])) if int(inst["c"][h, j]) == 1]
        if stages:
            first[stages[0], j] = 1
            for h in stages[1:]:
                follow[h, j] = 1

    patient_completion: Dict[Tuple[int, int], np.ndarray] = {}
    for h in range(int(inst["H"])):
        completion = None
        for i in range(int(inst["Iv"][h]), int(inst["I"])):
            j = int(assignment[h, i])
            deviation = (
                np.asarray(inst["tau"][:, j], dtype=float) * first[h, j]
                + np.asarray(inst["psi"][:, j], dtype=float) * follow[h, j]
            )

            previous_stages = [
                g for g in range(h) if int(inst["c"][g, j]) == 1
            ]
            predecessor = (
                patient_completion[(previous_stages[-1], j)]
                if previous_stages else np.zeros(int(inst["K"]), dtype=float)
            )

            def direct_at(candidate: float) -> Tuple[float, np.ndarray]:
                arrival = candidate + deviation
                ready = np.maximum(arrival, predecessor)
                start = ready if completion is None else np.maximum(ready, completion)
                direct = float(np.dot(inst["p"], start - ready))
                service = np.asarray(inst["s"][:, h, i, j], dtype=float)
                return direct, start + service

            current_direct, current_completion = direct_at(float(repaired[h, i]))
            tolerance = float(inst["r_d"][j])
            if current_direct > tolerance + 1e-8:
                lower = float(repaired[h, i])
                upper = float(inst["L"][h])
                upper_direct, _ = direct_at(upper)
                if upper_direct > tolerance + 1e-8:
                    raise RuntimeError(
                        f"BW schedule cannot be repaired within L at stage={h}, position={i}."
                    )
                for _ in range(60):
                    middle = 0.5 * (lower + upper)
                    middle_direct, _ = direct_at(middle)
                    if middle_direct <= tolerance:
                        upper = middle
                    else:
                        lower = middle
                repaired[h, i] = upper
                _, current_completion = direct_at(upper)
            completion = current_completion
            patient_completion[(h, j)] = np.asarray(current_completion, dtype=float)
    return repaired


def evaluate_and_store(
    *, inst: Dict, x: np.ndarray, y: np.ndarray, method: str, certified: int,
    runtime_sec: float, rep: int, K: int, raw_path: str, patient_path: str,
) -> None:
    validation = validate_fixed_schedule(inst, x, y)
    direct = indirect = math.nan
    patient_rows: List[Dict] = []
    if validation.feasible:
        direct, indirect, patient_rows = burden_metrics(inst, x, validation)
    common = {
        "protocol_version": "EJOR-SECTION52-PROTOCOL", "J": 4, "K": K, "rep": rep,
        "seed_service": inst["seed_service"], "seed_lateness": inst["seed_lateness"],
        "method": method,
    }
    append_row(raw_path, RAW_FIELDS, {
        **common, "status": validation.status, "certified": int(certified),
        "validated_obj": validation.objective if validation.feasible else math.nan,
        "alpha_sum": float(np.sum(validation.alpha)) if validation.feasible else math.nan,
        "beta_sum": float(np.sum(validation.beta)) if validation.feasible else math.nan,
        "avg_direct_burden": direct, "avg_indirect_burden": indirect,
        "avg_total_burden": direct + indirect,
        "validation_max_violation": validation.max_violation,
        "runtime_sec": runtime_sec,
    })
    for row in patient_rows:
        append_row(patient_path, PATIENT_FIELDS, {**common, **row})
    print(
        f"rep={rep} K={K} {method}: status={validation.status} "
        f"obj={validation.objective:.6g} time={runtime_sec:.2f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=2025)
    parser.add_argument("--seed-stride", type=int, default=10)
    parser.add_argument("--K", default="15,50,200,500")
    parser.add_argument("--budget", type=float, default=120.0)
    parser.add_argument("--mip-gap", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ss-stage-budget", type=float, default=60.0)
    parser.add_argument("--output-flag", type=int, default=0)
    parser.add_argument("--output", default="section52_results_long.csv")
    parser.add_argument("--patients", default="section52_patient_burdens.csv")
    parser.add_argument("--summary", default="section52_summary.csv")
    args = parser.parse_args()

    K_values = [int(item) for item in args.K.split(",") if item.strip()]
    max_K = max(K_values)
    c = np.asarray([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0]], dtype=int)
    r_d = np.asarray([2.0, 2.0, 2.4, 2.4], dtype=float)
    r_s = np.asarray([3.2, 3.2, 4.0, 4.0], dtype=float)
    limits = np.asarray([2.0, 5.0, 2.0], dtype=float)
    iv = [int(np.sum(c[h, :] == 0)) for h in range(3)]
    done = existing_keys(args.output)

    for rep in range(args.reps):
        seed_service = int(args.seed_base + args.seed_stride * rep)
        seed_lateness = seed_service
        s_all, tau_all, psi_all = generate_base_scenarios(
            max_K, c, seed_service
        )
        for K in K_values:
            s = np.asarray(s_all[:K], dtype=float)
            tau = np.asarray(tau_all[:K], dtype=float)
            psi = np.asarray(psi_all[:K], dtype=float)
            p = np.full(K, 1.0 / K, dtype=float)
            big_m = compute_data_driven_bigM(
                s=s, tau=tau, psi=psi, c=c, L=limits, Iv=iv,
            )
            inst = {
                "H": 3, "I": 4, "J": 4, "K": K, "c": c,
                "r_d": r_d, "r_s": r_s, "L": limits, "Iv": iv,
                "s": s, "tau": tau, "psi": psi, "p": p, "bigM": big_m,
                "seed_service": seed_service, "seed_lateness": seed_lateness,
            }
            ss = StageByStageHeuristic(
                s=s, tau=tau, psi=psi, c=c, r_d=r_d, r_s=r_s,
                Iv=iv, p=p, bigM=big_m, threads=args.threads, seed=seed_service,
            )

            if (rep, K, "DE-MILP") not in done:
                result = solve_de_milp(
                    inst, budget=args.budget, mip_gap=args.mip_gap,
                    threads=args.threads, output_flag=args.output_flag,
                    progress_checkpoints=(),
                )
                if result.x is None or result.y is None:
                    raise RuntimeError(f"DE-MILP returned no schedule for rep={rep}, K={K}")
                evaluate_and_store(
                    inst=inst, x=result.x, y=result.y, method="DE-MILP",
                    certified=int(result.certified), runtime_sec=result.runtime_sec,
                    rep=rep, K=K, raw_path=args.output, patient_path=args.patients,
                )
                done.add((rep, K, "DE-MILP"))
                del result

            if (rep, K, "SS-BW") not in done:
                start = time.time()
                schedule = ss.build_ss_bw(kappa=1.0, order_rule="index")
                schedule.y = repair_bw_direct_feasibility(
                    inst, schedule.x, schedule.y
                )
                evaluate_and_store(
                    inst=inst, x=schedule.x, y=schedule.y, method="SS-BW",
                    certified=0, runtime_sec=time.time() - start, rep=rep, K=K,
                    raw_path=args.output, patient_path=args.patients,
                )
                done.add((rep, K, "SS-BW"))

            if (rep, K, "SS-MILP") not in done:
                start = time.time()
                schedule = ss.build_ss_milp(
                    time_limit_per_stage=args.ss_stage_budget,
                    mip_gap=args.mip_gap, output_flag=args.output_flag, L=limits,
                )
                evaluate_and_store(
                    inst=inst, x=schedule.x, y=schedule.y, method="SS-MILP",
                    certified=0, runtime_sec=time.time() - start, rep=rep, K=K,
                    raw_path=args.output, patient_path=args.patients,
                )
                done.add((rep, K, "SS-MILP"))

            build_summary(args.output, args.summary)
            gc.collect()
        del s_all, tau_all, psi_all
        gc.collect()
    build_summary(args.output, args.summary)


if __name__ == "__main__":
    main()
