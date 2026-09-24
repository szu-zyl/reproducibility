"""Create the Section 5.3 tables from the individual run records."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


SCALES = (8, 12, 16, 20)
REPS = tuple(range(10))
METHODS = ("DE-MILP", "LBBD", "CBBD", "K-BRKGA")
T95_DF9 = 2.2621571628540993
TOL = 1e-7


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows were produced for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, str], field: str, default: float = math.nan) -> float:
    try:
        return float(row.get(field, ""))
    except (TypeError, ValueError):
        return default


def integer(row: dict[str, str], field: str, default: int = -1) -> int:
    value = number(row, field, float(default))
    return int(value) if math.isfinite(value) else default


def moments(values: list[float]) -> tuple[float, float, float]:
    average = mean(values)
    deviation = stdev(values) if len(values) > 1 else 0.0
    half_width = T95_DF9 * deviation / math.sqrt(len(values))
    return average, deviation, half_width


def beta_fraction(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) >= 1e-300 else 1e-300)
    result = d
    for iteration in range(1, 201):
        twice = 2 * iteration
        coefficient = iteration * (b - iteration) * x / ((qam + twice) * (a + twice))
        d = 1.0 + coefficient * d
        d = 1.0 / (d if abs(d) >= 1e-300 else 1e-300)
        c = 1.0 + coefficient / c
        c = c if abs(c) >= 1e-300 else 1e-300
        result *= d * c
        coefficient = -(a + iteration) * (qab + iteration) * x / ((a + twice) * (qap + twice))
        d = 1.0 + coefficient * d
        d = 1.0 / (d if abs(d) >= 1e-300 else 1e-300)
        c = 1.0 + coefficient / c
        c = c if abs(c) >= 1e-300 else 1e-300
        change = d * c
        result *= change
        if abs(change - 1.0) < 3e-14:
            break
    return result


def regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * beta_fraction(a, b, x) / a
    return 1.0 - front * beta_fraction(b, a, 1.0 - x) / b


def paired_t_pvalue(differences: list[float]) -> float:
    deviation = stdev(differences)
    if deviation == 0.0:
        return 0.0 if abs(mean(differences)) > TOL else 1.0
    t_value = abs(mean(differences)) / (deviation / math.sqrt(len(differences)))
    degrees = len(differences) - 1
    return regularized_beta(degrees / (degrees + t_value * t_value), degrees / 2.0, 0.5)


def signed_rank_pvalue(differences: list[float]) -> float:
    nonzero = [value for value in differences if abs(value) > TOL]
    if not nonzero:
        return 1.0
    ordered = sorted(range(len(nonzero)), key=lambda index: abs(nonzero[index]))
    ranks = [0.0] * len(nonzero)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and abs(abs(nonzero[ordered[end]]) - abs(nonzero[ordered[position]])) <= TOL:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for at in range(position, end):
            ranks[ordered[at]] = average_rank
        position = end
    observed = abs(sum(math.copysign(rank, value) for rank, value in zip(ranks, nonzero)))
    extreme = 0
    total = 1 << len(ranks)
    for mask in range(total):
        value = sum(rank if mask & (1 << index) else -rank for index, rank in enumerate(ranks))
        extreme += abs(value) >= observed - 1e-12
    return extreme / total


def make_long(exact: list[dict[str, str]], heuristic: list[dict[str, str]]) -> list[dict]:
    rows: list[dict] = []
    for source in exact + heuristic:
        method = source["method"]
        rows.append(
            {
                "J": integer(source, "J"),
                "I": integer(source, "I", integer(source, "J")),
                "K": integer(source, "K"),
                "rep": integer(source, "rep"),
                "seed_service": integer(source, "seed_service"),
                "seed_lateness": integer(source, "seed_lateness"),
                "method": method,
                "algorithm_class": "independent heuristic comparator" if method == "K-BRKGA" else "exact formulation/decomposition",
                "timing_scope": "fixed stable slots" if method == "K-BRKGA" else "joint sequence-and-time optimization",
                "validated_obj": number(source, "validated_obj"),
                "runtime_sec": number(source, "runtime_sec"),
                "budget_sec": number(source, "budget_sec"),
                "validation_status": source["validation_status"],
                "protocol_pass": source.get("protocol_pass", "not_applicable"),
                "test_K": integer(source, "test_K"),
                "test_mean_direct": number(source, "test_mean_direct"),
                "test_p95_direct": number(source, "test_p95_direct"),
                "test_direct_exceed_rate": number(source, "test_direct_exceed_rate"),
                "test_mean_indirect": number(source, "test_mean_indirect"),
                "test_p95_indirect": number(source, "test_p95_indirect"),
                "test_indirect_exceed_rate": number(source, "test_indirect_exceed_rate"),
                "cuts_added": number(source, "cuts_added"),
                "total_scenario_screens": number(source, "total_scenario_screens"),
                "num_vars": number(source, "num_vars"),
                "num_constrs": number(source, "num_constrs"),
                "num_genconstrs": number(source, "num_genconstrs"),
            }
        )
    rows.sort(key=lambda row: (row["J"], row["rep"], METHODS.index(row["method"])))
    expected = {(scale, rep, method) for scale in SCALES for rep in REPS for method in METHODS}
    observed = {(row["J"], row["rep"], row["method"]) for row in rows}
    if observed != expected:
        raise ValueError("The individual run files do not contain the expected 160 method-instance records.")
    return rows


def make_summary(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["J"], row["method"])].append(row)
    output: list[dict] = []
    for scale in SCALES:
        for method in METHODS:
            cell = sorted(groups[(scale, method)], key=lambda row: row["rep"])
            objectives = [row["validated_obj"] for row in cell]
            runtimes = [row["runtime_sec"] for row in cell]
            objective_mean, objective_sd, half_width = moments(objectives)
            runtime_mean, runtime_sd, _ = moments(runtimes)
            output.append(
                {
                    "J": scale, "I": scale, "method": method, "n": len(cell),
                    "validated_obj_mean": objective_mean, "validated_obj_sd": objective_sd,
                    "validated_obj_ci95_low": objective_mean - half_width,
                    "validated_obj_ci95_high": objective_mean + half_width,
                    "runtime_mean_sec": runtime_mean, "runtime_sd_sec": runtime_sd,
                    "validation_passes": sum(row["validation_status"] == "PASS" for row in cell),
                    "test_mean_direct_mean": mean(row["test_mean_direct"] for row in cell),
                    "test_p95_direct_mean": mean(row["test_p95_direct"] for row in cell),
                    "test_direct_exceed_rate_mean": mean(row["test_direct_exceed_rate"] for row in cell),
                    "test_mean_indirect_mean": mean(row["test_mean_indirect"] for row in cell),
                    "test_p95_indirect_mean": mean(row["test_p95_indirect"] for row in cell),
                    "test_indirect_exceed_rate_mean": mean(row["test_indirect_exceed_rate"] for row in cell),
                }
            )
    return output


def make_paired(rows: list[dict]) -> list[dict]:
    lookup = {(row["J"], row["rep"], row["method"]): row["validated_obj"] for row in rows}
    comparisons = (("LBBD", "DE-MILP"), ("CBBD", "DE-MILP"), ("LBBD", "K-BRKGA"), ("CBBD", "K-BRKGA"), ("CBBD", "LBBD"))
    output: list[dict] = []
    for scale in SCALES:
        for first, reference in comparisons:
            first_values = [lookup[(scale, rep, first)] for rep in REPS]
            reference_values = [lookup[(scale, rep, reference)] for rep in REPS]
            differences = [right - left for left, right in zip(first_values, reference_values)]
            percentages = [100.0 * (right - left) / right for left, right in zip(first_values, reference_values)]
            average, deviation, half_width = moments(differences)
            output.append(
                {
                    "J": scale, "I": scale, "first_method": first, "reference_method": reference,
                    "interpretation": "positive reduction means first_method is better (lower)", "n": 10,
                    "mean_objective_reduction": average, "sd_objective_reduction": deviation,
                    "ci95_low": average - half_width, "ci95_high": average + half_width,
                    "mean_percent_reduction": mean(percentages),
                    "wins": sum(left < right - TOL for left, right in zip(first_values, reference_values)),
                    "ties": sum(abs(left - right) <= TOL for left, right in zip(first_values, reference_values)),
                    "losses": sum(left > right + TOL for left, right in zip(first_values, reference_values)),
                    "paired_t_p": paired_t_pvalue(differences),
                    "signed_rank_permutation_p": signed_rank_pvalue(differences),
                }
            )
    return output


def make_work(exact: list[dict[str, str]]) -> list[dict]:
    lookup = {(integer(row, "J"), integer(row, "rep"), row["method"]): row for row in exact}
    output: list[dict] = []
    for scale in SCALES:
        lbdd = [lookup[(scale, rep, "LBBD")] for rep in REPS]
        cbdd = [lookup[(scale, rep, "CBBD")] for rep in REPS]
        lbdd_cuts = mean(number(row, "cuts_added") for row in lbdd)
        cbdd_cuts = mean(number(row, "cuts_added") for row in cbdd)
        lbdd_screens = mean(number(row, "total_scenario_screens") for row in lbdd)
        cbdd_screens = mean(number(row, "total_scenario_screens") for row in cbdd)
        output.append(
            {
                "J": scale, "I": scale,
                "LBBD_mean_cuts": lbdd_cuts, "CBBD_mean_cuts": cbdd_cuts,
                "CBBD_cut_reduction_percent": 100.0 * (lbdd_cuts - cbdd_cuts) / lbdd_cuts,
                "LBBD_mean_scenario_screens": lbdd_screens,
                "CBBD_mean_scenario_screens": cbdd_screens,
                "CBBD_screen_reduction_percent": 100.0 * (lbdd_screens - cbdd_screens) / lbdd_screens,
                "LBBD_mean_vars": mean(number(row, "num_vars") for row in lbdd),
                "CBBD_mean_vars": mean(number(row, "num_vars") for row in cbdd),
                "LBBD_mean_constraints": mean(number(row, "num_constrs") for row in lbdd),
                "CBBD_mean_constraints": mean(number(row, "num_constrs") for row in cbdd),
            }
        )
    return output


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact", type=Path, default=package_root / "data" / "section_5_3" / "exact_runs.csv")
    parser.add_argument("--kbrkga", type=Path, default=package_root / "data" / "section_5_3" / "kbrkga_runs.csv")
    parser.add_argument("--output-dir", type=Path, default=package_root / "data" / "section_5_3")
    args = parser.parse_args()

    exact = read_csv(args.exact)
    heuristic = read_csv(args.kbrkga)
    if len(exact) != 120 or len(heuristic) != 40:
        raise ValueError("Expected 120 exact-method records and 40 K-BRKGA records.")
    long_rows = make_long(exact, heuristic)
    write_csv(args.output_dir / "all_methods.csv", long_rows)
    write_csv(args.output_dir / "summary.csv", make_summary(long_rows))
    write_csv(args.output_dir / "paired_comparisons.csv", make_paired(long_rows))
    write_csv(args.output_dir / "decomposition_work.csv", make_work(exact))
    print(f"Wrote Section 5.3 summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
