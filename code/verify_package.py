"""Verify the released data, instances, summaries, and reported table values."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from experiment_utils import build_scale_stable_instance


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PACKAGE_ROOT / "data"
INSTANCE_ROOT = PACKAGE_ROOT / "instances"
T95_DF9 = 2.262
TOL = 1e-9


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], field: str) -> float:
    return float(row[field])


def as_int(row: dict[str, str], field: str) -> int:
    return int(float(row[field]))


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class Audit:
    def __init__(self) -> None:
        self.rows: list[dict[str, str]] = []

    def record(self, category: str, check: str, observed, expected, passed: bool) -> None:
        self.rows.append(
            {
                "category": category,
                "check": check,
                "observed": str(observed),
                "expected": str(expected),
                "status": "PASS" if passed else "FAIL",
            }
        )

    def equal(self, category: str, check: str, observed, expected) -> None:
        self.record(category, check, observed, expected, observed == expected)

    def close(self, category: str, check: str, observed: float, expected: float, atol: float = 1e-9) -> None:
        passed = math.isclose(float(observed), float(expected), rel_tol=1e-9, abs_tol=atol)
        self.record(category, check, f"{observed:.15g}", f"{expected:.15g}", passed)

    def require(self, category: str, check: str, passed: bool, observed="true", expected="true") -> None:
        self.record(category, check, observed, expected, bool(passed))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)


def grouped(rows: list[dict[str, str]], fields: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    output: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        output[tuple(row[field] for field in fields)].append(row)
    return output


def verify_section52(audit: Audit) -> None:
    runs = read_csv(DATA_ROOT / "section_5_2" / "individual_runs.csv")
    patients = read_csv(DATA_ROOT / "section_5_2" / "patient_burdens.csv")
    summary = read_csv(DATA_ROOT / "section_5_2" / "summary.csv")
    audit.equal("Section 5.2", "individual run count", len(runs), 120)
    audit.equal("Section 5.2", "patient-level row count", len(patients), 480)
    audit.equal("Section 5.2", "summary row count", len(summary), 12)
    audit.require(
        "Section 5.2",
        "all run keys are unique",
        len({(r["K"], r["rep"], r["method"]) for r in runs}) == 120,
    )
    audit.require(
        "Section 5.2",
        "all recorded scenario seeds match the common-seed generator",
        all(r["seed_service"] == r["seed_lateness"] for r in runs),
    )
    lookup = grouped(runs, ("K", "method"))
    for row in summary:
        key = (row["K"], row["method"])
        cell = lookup[key]
        values = {
            "obj": [as_float(x, "validated_obj") for x in cell],
            "alpha": [as_float(x, "alpha_sum") for x in cell],
            "beta": [as_float(x, "beta_sum") for x in cell],
            "direct_burden": [as_float(x, "avg_direct_burden") for x in cell],
            "indirect_burden": [as_float(x, "avg_indirect_burden") for x in cell],
            "runtime": [as_float(x, "runtime_sec") for x in cell],
        }
        audit.equal("Section 5.2", f"{key} replication count", len(cell), 10)
        for prefix, data in values.items():
            audit.close("Section 5.2", f"{key} {prefix} mean", mean(data), as_float(row, f"{prefix}_mean"))
            audit.close("Section 5.2", f"{key} {prefix} standard deviation", stdev(data), as_float(row, f"{prefix}_std"))
        half = T95_DF9 * stdev(values["obj"]) / math.sqrt(10)
        audit.close("Section 5.2", f"{key} objective confidence half-width", half, as_float(row, "obj_ci95_half"))
        audit.equal("Section 5.2", f"{key} validation passes", sum(x["status"] == "PASS" for x in cell), 10)


def beta_continued_fraction(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / max(abs(d), 1e-300) * (1 if d >= 0 else -1)
    h = d
    for iteration in range(1, 201):
        m2 = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) >= 1e-300 else 1e-300)
        c = 1.0 + aa / c
        c = c if abs(c) >= 1e-300 else 1e-300
        h *= d * c
        aa = -(a + iteration) * (qab + iteration) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) >= 1e-300 else 1e-300)
        c = 1.0 + aa / c
        c = c if abs(c) >= 1e-300 else 1e-300
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * beta_continued_fraction(a, b, x) / a
    return 1.0 - front * beta_continued_fraction(b, a, 1.0 - x) / b


def paired_t_pvalue(values: list[float]) -> float:
    sd = stdev(values)
    if sd == 0:
        return 0.0 if abs(mean(values)) > TOL else 1.0
    t_value = abs(mean(values)) / (sd / math.sqrt(len(values)))
    degrees = len(values) - 1
    return regularized_beta(degrees / (degrees + t_value * t_value), degrees / 2.0, 0.5)


def verify_section53(audit: Audit) -> None:
    exact = read_csv(DATA_ROOT / "section_5_3" / "exact_runs.csv")
    heuristic = read_csv(DATA_ROOT / "section_5_3" / "kbrkga_runs.csv")
    all_methods = read_csv(DATA_ROOT / "section_5_3" / "all_methods.csv")
    summary = read_csv(DATA_ROOT / "section_5_3" / "summary.csv")
    paired = read_csv(DATA_ROOT / "section_5_3" / "paired_comparisons.csv")
    work = read_csv(DATA_ROOT / "section_5_3" / "decomposition_work.csv")
    audit.equal("Section 5.3", "exact-method record count", len(exact), 120)
    audit.equal("Section 5.3", "K-BRKGA record count", len(heuristic), 40)
    audit.equal("Section 5.3", "all-method record count", len(all_methods), 160)
    audit.require("Section 5.3", "all fixed-schedule audits pass", all(r["validation_status"] == "PASS" for r in exact + heuristic))
    audit.require("Section 5.3", "all exact objective checks pass", all(as_int(r, "objective_consistent") == 1 for r in exact))
    audit.require("Section 5.3", "all decomposition runs pass full separation", all(as_int(r, "full_separation") == 1 for r in exact if r["method"] in {"LBBD", "CBBD"}))
    exact_keys = {(r["J"], r["rep"], r["method"]): r for r in exact}
    heuristic_keys = {(r["J"], r["rep"], r["method"]): r for r in heuristic}
    long_keys = {(r["J"], r["rep"], r["method"]): r for r in all_methods}
    audit.equal("Section 5.3", "unique exact keys", len(exact_keys), 120)
    audit.equal("Section 5.3", "unique K-BRKGA keys", len(heuristic_keys), 40)
    audit.equal("Section 5.3", "unique combined keys", len(long_keys), 160)
    for key, source in {**exact_keys, **heuristic_keys}.items():
        combined = long_keys[key]
        audit.close("Section 5.3", f"{key} combined objective", as_float(source, "validated_obj"), as_float(combined, "validated_obj"))
        audit.equal("Section 5.3", f"{key} common service seed", source["seed_service"], combined["seed_service"])
        audit.equal("Section 5.3", f"{key} common lateness seed", source["seed_lateness"], combined["seed_lateness"])
    groups = grouped(all_methods, ("J", "method"))
    for row in summary:
        key = (row["J"], row["method"])
        cell = sorted(groups[key], key=lambda x: int(x["rep"]))
        objectives = [as_float(x, "validated_obj") for x in cell]
        runtimes = [as_float(x, "runtime_sec") for x in cell]
        audit.equal("Section 5.3", f"{key} replication count", len(cell), 10)
        audit.close("Section 5.3", f"{key} objective mean", mean(objectives), as_float(row, "validated_obj_mean"))
        audit.close("Section 5.3", f"{key} objective standard deviation", stdev(objectives), as_float(row, "validated_obj_sd"))
        audit.close("Section 5.3", f"{key} runtime mean", mean(runtimes), as_float(row, "runtime_mean_sec"))
        for field in (
            "test_mean_direct", "test_p95_direct", "test_direct_exceed_rate",
            "test_mean_indirect", "test_p95_indirect", "test_indirect_exceed_rate",
        ):
            audit.close(
                "Section 5.3",
                f"{key} {field} mean",
                mean(as_float(x, field) for x in cell),
                as_float(row, f"{field}_mean"),
            )
    for row in paired:
        first = [long_keys[(row["J"], str(rep), row["first_method"])] for rep in range(10)]
        reference = [long_keys[(row["J"], str(rep), row["reference_method"])] for rep in range(10)]
        differences = [as_float(b, "validated_obj") - as_float(a, "validated_obj") for a, b in zip(first, reference)]
        percentages = [100.0 * d / as_float(b, "validated_obj") for d, b in zip(differences, reference)]
        key = (row["J"], row["first_method"], row["reference_method"])
        audit.close("Section 5.3 paired", f"{key} mean reduction", mean(differences), as_float(row, "mean_objective_reduction"))
        audit.close("Section 5.3 paired", f"{key} mean percent reduction", mean(percentages), as_float(row, "mean_percent_reduction"))
        audit.equal("Section 5.3 paired", f"{key} wins", sum(d > TOL for d in differences), as_int(row, "wins"))
        audit.equal("Section 5.3 paired", f"{key} ties", sum(abs(d) <= TOL for d in differences), as_int(row, "ties"))
        audit.equal("Section 5.3 paired", f"{key} losses", sum(d < -TOL for d in differences), as_int(row, "losses"))
        audit.close("Section 5.3 paired", f"{key} paired t p-value", paired_t_pvalue(differences), as_float(row, "paired_t_p"), atol=1e-12)
    for row in work:
        patient_count = row["J"]
        lbbd = [r for r in exact if r["J"] == patient_count and r["method"] == "LBBD"]
        cbbd = [r for r in exact if r["J"] == patient_count and r["method"] == "CBBD"]
        l_cuts = mean(as_float(r, "cuts_added") for r in lbbd)
        c_cuts = mean(as_float(r, "cuts_added") for r in cbbd)
        l_screens = mean(as_float(r, "total_scenario_screens") for r in lbbd)
        c_screens = mean(as_float(r, "total_scenario_screens") for r in cbbd)
        audit.close("Section 5.3 work", f"J={patient_count} LBBD cuts", l_cuts, as_float(row, "LBBD_mean_cuts"))
        audit.close("Section 5.3 work", f"J={patient_count} CBBD cuts", c_cuts, as_float(row, "CBBD_mean_cuts"))
        audit.close("Section 5.3 work", f"J={patient_count} LBBD screens", l_screens, as_float(row, "LBBD_mean_scenario_screens"))
        audit.close("Section 5.3 work", f"J={patient_count} CBBD screens", c_screens, as_float(row, "CBBD_mean_scenario_screens"))


def verify_section54(audit: Audit) -> None:
    runs = read_csv(DATA_ROOT / "section_5_4" / "individual_runs.csv")
    summary = read_csv(DATA_ROOT / "section_5_4" / "summary.csv")
    audit.equal("Section 5.4", "individual run count", len(runs), 180)
    audit.equal("Section 5.4", "summary row count", len(summary), 18)
    audit.require(
        "Section 5.4",
        "all recorded scenario seeds match the common-seed generator",
        all(r["seed_service"] == r["seed_lateness"] for r in runs),
    )
    lookup = grouped(runs, ("K", "kappa"))
    for row in summary:
        key = (row["K"], row["kappa"])
        cell = lookup[key]
        audit.equal("Section 5.4", f"{key} replication count", len(cell), 10)
        fields = {
            "obj": "validated_obj", "alpha": "alpha_sum", "beta": "beta_sum",
            "tau": "tau_mean", "psi": "psi_mean", "runtime": "runtime_sec",
        }
        for prefix, source in fields.items():
            values = [as_float(x, source) for x in cell]
            audit.close("Section 5.4", f"{key} {prefix} mean", mean(values), as_float(row, f"{prefix}_mean"))
            audit.close("Section 5.4", f"{key} {prefix} standard deviation", stdev(values), as_float(row, f"{prefix}_std"))


def verify_instances(audit: Audit) -> None:
    manifest = read_csv(INSTANCE_ROOT / "manifest.csv")
    audit.equal("Instances", "manifest row count", len(manifest), 260)
    audit.equal("Instances", "unique instance identifiers", len({r["instance_id"] for r in manifest}), 260)
    paths = {r["file"] for r in manifest}
    audit.equal("Instances", "static instance file count", len(paths), 60)
    for relative in sorted(paths):
        path = PACKAGE_ROOT / relative
        rows = [r for r in manifest if r["file"] == relative]
        audit.require("Instances", f"{relative} exists", path.is_file())
        audit.equal("Instances", f"{relative} checksum", file_hash(path), rows[0]["sha256"])
    section52_runs = read_csv(DATA_ROOT / "section_5_2" / "individual_runs.csv")
    section54_runs = read_csv(DATA_ROOT / "section_5_4" / "individual_runs.csv")
    section53_exact = read_csv(DATA_ROOT / "section_5_3" / "exact_runs.csv")
    for row in manifest:
        if row["section"] == "5.4":
            source = np.load(PACKAGE_ROOT / row["file"])
            count = int(row["K"])
            kappa = float(row["kappa"])
            tau = np.clip(kappa * source["tau"][:count], 0.0, 0.5)
            psi = np.clip(kappa * source["psi"][:count], 0.0, 0.5)
            result = next(
                x for x in section54_runs
                if x["K"] == row["K"]
                and float(x["kappa"]) == float(row["kappa"])
                and x["rep"] == row["rep"]
            )
            audit.close("Instances", f"{row['instance_id']} tau mean", float(np.mean(tau)), as_float(result, "tau_mean"))
            audit.close("Instances", f"{row['instance_id']} psi mean", float(np.mean(psi)), as_float(result, "psi_mean"))
        elif row["section"] == "5.2":
            matches = [
                x for x in section52_runs
                if x["K"] == row["K"] and x["rep"] == row["rep"]
            ]
            audit.equal("Instances", f"{row['instance_id']} reported methods", len(matches), 3)
        elif row["section"] == "5.3":
            source = np.load(PACKAGE_ROOT / row["file"])
            patient_count = int(row["J"])
            rep = int(row["rep"])
            regenerated = build_scale_stable_instance(
                J=patient_count,
                K=100,
                seed_service=int(row["service_seed"]),
                seed_lateness=int(row["lateness_seed"]),
                buffer_z=0.5,
                terminal_shortfall=0.6,
            )
            audit.require("Instances", f"{row['instance_id']} service scenarios", np.array_equal(source["s"], regenerated["s"]))
            audit.require("Instances", f"{row['instance_id']} initial lateness scenarios", np.array_equal(source["tau"], regenerated["tau"]))
            audit.require("Instances", f"{row['instance_id']} carried lateness scenarios", np.array_equal(source["psi"], regenerated["psi"]))
            matches = [x for x in section53_exact if int(x["J"]) == patient_count and int(x["rep"]) == rep]
            audit.equal("Instances", f"{row['instance_id']} exact-method records", len(matches), 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    audit = Audit()
    verify_section52(audit)
    verify_section53(audit)
    verify_section54(audit)
    verify_instances(audit)
    if args.report is not None:
        audit.write(args.report)
    failures = [row for row in audit.rows if row["status"] == "FAIL"]
    print(f"Verification checks: {len(audit.rows)}")
    print(f"Failures: {len(failures)}")
    if args.report is not None:
        print(f"Report: {args.report}")
    if failures:
        for row in failures[:20]:
            print(f"FAIL [{row['category']}] {row['check']}: {row['observed']} != {row['expected']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
