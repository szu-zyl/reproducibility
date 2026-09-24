"""Small-instance consistency and structural tests for the exact methods."""

from __future__ import annotations

import argparse
import math

import numpy as np

from fixed_schedule_validation import validate_fixed_schedule

from run_section53_exact import (
    build_scale_stable_instance,
    solve_de_milp,
    solve_benders,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=120.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--K", type=int, default=8)
    args = parser.parse_args()

    inst = build_scale_stable_instance(
        J=4,
        K=args.K,
        seed_service=14065,
        seed_lateness=14066,
        buffer_z=0.5,
        terminal_shortfall=0.6,
    )
    results = [
        solve_de_milp(inst, args.budget, 1e-4, args.threads),
        solve_benders(inst, "LBBD", args.budget, 1e-4, args.threads),
        solve_benders(inst, "CBBD", args.budget, 1e-4, args.threads),
    ]
    for result in results:
        if not math.isfinite(result.obj):
            raise AssertionError(f"{result.method} returned no incumbent: {result.status}")
        if not result.certified:
            raise AssertionError(f"{result.method} was not certified: gap={result.rel_gap}")
        if not result.full_separation:
            raise AssertionError(
                f"{result.method} failed full separation: {result.max_final_violation}"
            )
        audit = validate_fixed_schedule(inst, result.x, result.y)
        if not audit.feasible:
            raise AssertionError(
                f"{result.method} independent audit failed: {audit.status}, "
                f"violation={audit.max_violation}"
            )
        if abs(audit.objective - result.obj) > 1e-5:
            raise AssertionError(
                f"{result.method} independent objective mismatch: "
                f"solver={result.obj:.9f}, audit={audit.objective:.9f}"
            )
        position = {
            (h, int(audit.assignment[h, i])): i
            for h in range(inst["H"])
            for i in range(inst["I"])
        }
        for h in range(inst["H"]):
            for j in range(inst["J"]):
                attended = [hh for hh in range(h) if int(inst["c"][hh, j]) == 1]
                if int(inst["c"][h, j]) != 1 or not attended:
                    continue
                previous = attended[-1]
                current_i = position[h, j]
                previous_i = position[previous, j]
                precedence_slack = (
                    audit.start[h, current_i, :]
                    - audit.completion[previous, previous_i, :]
                )
                if float(np.min(precedence_slack)) < -1e-7:
                    raise AssertionError(
                        f"{result.method} violates cross-stage precedence for "
                        f"patient={j}, stage={h}: min slack={np.min(precedence_slack)}"
                    )

    reference = results[0].obj
    for result in results[1:]:
        if abs(result.obj - reference) > 1e-4:
            raise AssertionError(
                f"Objective mismatch: DE={reference:.9f}, {result.method}={result.obj:.9f}"
            )
        if result.num_genconstrs >= results[0].num_genconstrs:
            raise AssertionError(
                f"{result.method} did not reduce the initially generated cross-stage "
                f"family: {result.num_genconstrs} >= {results[0].num_genconstrs}"
            )

    for result in results:
        print(
            f"{result.method}: obj={result.obj:.9f}, gap={result.rel_gap:.3g}, "
            f"time={result.runtime_sec:.3f}s, vars={result.num_vars}, "
            f"rows={result.num_constrs}, gen={result.num_genconstrs}, "
            f"cuts={result.cuts_added}"
        )
    print("Exact-method consistency: PASS")


if __name__ == "__main__":
    main()
