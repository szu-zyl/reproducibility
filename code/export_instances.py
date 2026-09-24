"""Export the numerical instances used in Sections 5.2, 5.3, and 5.4."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

from experiment_utils import build_scale_stable_instance
from scenario_generation import generate_all_scenarios


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
INSTANCE_ROOT = PACKAGE_ROOT / "instances"
SMALL_PATHWAY = np.asarray(
    [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0]], dtype=int
)
SMALL_DIRECT_TOLERANCE = np.asarray([2.0, 2.0, 2.4, 2.4], dtype=float)
SMALL_INTERSTAGE_TOLERANCE = np.asarray([3.2, 3.2, 4.0, 4.0], dtype=float)
SMALL_WINDOWS = np.asarray([2.0, 5.0, 2.0], dtype=float)
SMALL_VIRTUAL_COUNT = np.asarray(
    [np.sum(SMALL_PATHWAY[h, :] == 0) for h in range(3)], dtype=int
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_small_base(section: str, rep: int) -> tuple[Path, int]:
    seed = 2025 + 10 * rep
    service, tau, psi = generate_all_scenarios(
        numSamples=500,
        numStages=3,
        numPositions=4,
        numPatients=4,
        c=SMALL_PATHWAY,
        seed=seed,
    )
    folder = INSTANCE_ROOT / section
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"rep_{rep:02d}_base_K500.npz"
    np.savez_compressed(
        path,
        c=SMALL_PATHWAY,
        r_d=SMALL_DIRECT_TOLERANCE,
        r_s=SMALL_INTERSTAGE_TOLERANCE,
        L=SMALL_WINDOWS,
        Iv=SMALL_VIRTUAL_COUNT,
        s=service,
        tau=tau,
        psi=psi,
        master_seed=np.asarray(seed, dtype=int),
    )
    return path, seed


def export_section52(rows: list[dict[str, str]]) -> None:
    for rep in range(10):
        path, seed = save_small_base("section_5_2", rep)
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        file_hash = digest(path)
        for scenario_count in (15, 50, 200, 500):
            rows.append(
                {
                    "section": "5.2",
                    "instance_id": f"S52_K{scenario_count}_R{rep}",
                    "J": "4",
                    "K": str(scenario_count),
                    "rep": str(rep),
                    "kappa": "",
                    "service_seed": str(seed),
                    "lateness_seed": str(seed),
                    "seed_protocol": "common_replication_seed",
                    "file": relative,
                    "instance_rule": f"use scenarios 0:{scenario_count}",
                    "test_K": "",
                    "test_service_seed": "",
                    "test_lateness_seed": "",
                    "reported_method_count": "3",
                    "sha256": file_hash,
                }
            )


def export_section53(rows: list[dict[str, str]]) -> None:
    folder = INSTANCE_ROOT / "section_5_3"
    folder.mkdir(parents=True, exist_ok=True)
    for patient_count in (8, 12, 16, 20):
        for rep in range(10):
            seed_service = 12065 + 100 * rep
            seed_lateness = seed_service + 1
            inst = build_scale_stable_instance(
                J=patient_count,
                K=100,
                seed_service=seed_service,
                seed_lateness=seed_lateness,
                buffer_z=0.5,
                terminal_shortfall=0.6,
            )
            path = folder / f"J{patient_count}_rep_{rep:02d}_K100.npz"
            np.savez_compressed(
                path,
                c=inst["c"],
                r_d=inst["r_d"],
                r_s=inst["r_s"],
                L=inst["L"],
                spacing=inst["spacing"],
                Iv=np.asarray(inst["Iv"], dtype=int),
                s=inst["s"],
                tau=inst["tau"],
                psi=inst["psi"],
                p=inst["p"],
                bigM=np.asarray(inst["bigM"], dtype=float),
                seed_service=np.asarray(seed_service, dtype=int),
                seed_lateness=np.asarray(seed_lateness, dtype=int),
            )
            rows.append(
                {
                    "section": "5.3",
                    "instance_id": f"S53_J{patient_count}_R{rep}",
                    "J": str(patient_count),
                    "K": "100",
                    "rep": str(rep),
                    "kappa": "",
                    "service_seed": str(seed_service),
                    "lateness_seed": str(seed_lateness),
                    "seed_protocol": "separate_service_and_lateness_seeds",
                    "file": path.relative_to(PACKAGE_ROOT).as_posix(),
                    "instance_rule": "stored instance",
                    "test_K": "1000",
                    "test_service_seed": str(seed_service + 10_000_000),
                    "test_lateness_seed": str(seed_lateness + 20_000_000),
                    "reported_method_count": "4",
                    "sha256": digest(path),
                }
            )


def export_section54(rows: list[dict[str, str]]) -> None:
    for rep in range(10):
        path, seed = save_small_base("section_5_4", rep)
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        file_hash = digest(path)
        for scenario_count in (50, 200, 500):
            for kappa in (0.0, 0.25, 0.75, 1.0, 1.5, 2.0):
                rows.append(
                    {
                        "section": "5.4",
                        "instance_id": f"S54_K{scenario_count}_KAPPA{kappa:g}_R{rep}",
                        "J": "4",
                        "K": str(scenario_count),
                        "rep": str(rep),
                        "kappa": f"{kappa:g}",
                        "service_seed": str(seed),
                        "lateness_seed": str(seed),
                        "seed_protocol": "common_replication_seed",
                        "file": relative,
                        "instance_rule": (
                            f"use scenarios 0:{scenario_count}; "
                            f"tau=clip({kappa:g}*tau,0,0.5); "
                            f"psi=clip({kappa:g}*psi,0,0.5)"
                        ),
                        "test_K": "",
                        "test_service_seed": "",
                        "test_lateness_seed": "",
                        "reported_method_count": "1",
                        "sha256": file_hash,
                    }
                )


def main() -> None:
    rows: list[dict[str, str]] = []
    export_section52(rows)
    export_section53(rows)
    export_section54(rows)
    manifest = INSTANCE_ROOT / "manifest.csv"
    fields = list(rows[0])
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} instance records to {manifest}")


if __name__ == "__main__":
    main()
