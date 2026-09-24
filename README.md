# Computational package for EJOR-D-25-02966R2

This package contains the code, numerical instances, individual run records, summary tables, and validation checks for the computational study in Sections 5.2, 5.3, and 5.4.

The study is motivated by serial maternal and child healthcare pathways. The released numerical instances follow the pathway structure and controlled distributions stated in the paper and Appendix F. They do not contain patient-level hospital records.
 
## Package contents

- `code/` contains the final model, LBBD, CBBD, K-BRKGA, experiment, plotting, instance-export, and verification programs.
- `data/section_5_2/` contains 120 method-instance runs, 480 patient-level burden records, and the 12-row summary used in Table 3 and Figures 4 and 5.
- `data/section_5_3/` contains 120 exact-method runs, 40 K-BRKGA runs, the combined 160-record file, and the summaries used in Tables 5, 6, and 8 and Appendix Tables F.2 to F.5.
- `data/section_5_4/` contains 180 individual runs and the 18-row summary used in Table 7.
- `instances/` contains 60 compressed base-instance files and a 260-row manifest. Prefix and lateness-intensity rules in the manifest identify every logical experimental instance.
- `figures/section_5_2/` contains the two figures reported in Section 5.2.
- `verification/` contains the mapping from the paper tables and figures to their source data.

## Code map

- `scenario_generation.py` generates service-time and late-arrival scenarios from recorded seeds.
- `model_utils.py` provides shared instance and data-driven bound calculations.
- `de_milp_base.py`, `de_milp_compact.py`, and `de_milp.py` implement the deterministic-equivalent model in three layers.
- `benders_core.py` implements the common single-tree separation framework.
- `benders_methods.py` exposes the LBBD and CBBD solver classes.
- `pathwise_primal_start.py` constructs the path-consistent primal start used by LBBD and CBBD.
- `fixed_schedule_validation.py` independently reconstructs and validates a returned schedule.
- `stagewise_baselines.py` implements SS-MILP and SS-BW for Section 5.2.
- `kbrkga.py` implements the independent K-BRKGA benchmark.
- `experiment_utils.py` builds the scale-stable patient-level instances and shared experiment records.
- `run_section52.py` runs the integrated and stage-wise comparison.
- `run_section53_exact.py` runs DE-MILP, LBBD, and CBBD.
- `run_section53_kbrkga.py` runs K-BRKGA separately from the exact methods.
- `run_section54.py` runs the lateness-intensity sensitivity experiment.
- `summarize_section53.py` builds the Section 5.3 combined, summary, paired, and mechanism tables from individual records.
- `plot_section52.py` recreates Figures 4 and 5.
- `export_instances.py` writes the static instance files and their manifest.
- `test_exact_methods.py` compares the three exact methods on a small common instance.
- `verify_package.py` checks data, instances, summaries, and validation fields.

## Software environment

The reported experiments used Python 3.12.3, NumPy 1.26.4, Matplotlib 3.10.8, and Gurobi 11.0.2 on Windows 11. The computer had an Intel Core i5-12400F processor with 6 physical cores, 12 logical cores, and 15.86 GiB of RAM. The MILP experiments used four threads.

## Instance and seed protocol

Section 5.2 uses ten replications with master seeds `2025 + 10r`, where `r=0,...,9`. The 500-scenario base draw for each replication is stored once. Instances with 15, 50, and 200 scenarios use the corresponding prefix.

Section 5.3 uses service-time seeds `12065 + 100r` and late-arrival seeds `12066 + 100r`. The same training and out-of-sample scenarios are used by all four methods for a given scale and replication.

Section 5.4 uses the same ten master seeds as Section 5.2. Each replication stores a 500-scenario base draw. Smaller scenario sets use prefixes, and the lateness multipliers are applied by the clipping rule recorded in `instances/manifest.csv`.

For Sections 5.2 and 5.4, the service and late-arrival arrays are produced by one deterministic scenario-generation call from the recorded master seed. The late-arrival generator applies its fixed internal offset. The `seed_service` and `seed_lateness` fields therefore both record that master seed.

## Verification

From the package root, run:

```text
python code/verify_package.py
python code/test_exact_methods.py --K 8 --budget 120 --threads 1
```

The first command checks record counts, unique instance keys, summary statistics, paired comparisons, validation flags, instance hashes, and links between the stored instances and results. A CSV report can be requested when needed:

```text
python code/verify_package.py --report verification_report.csv
```

The second command solves a small common instance with DE-MILP, LBBD, and CBBD and compares the independently validated objectives.

## Recreating the stored instances

```text
python code/export_instances.py
```

This command deterministically recreates the compressed instance files and `instances/manifest.csv`.

## Repeating the experiments

Create a separate output folder so that the archived records remain unchanged:

```text
mkdir reproduced_results
```

Section 5.2:

```text
python code/run_section52.py --output reproduced_results/section52_runs.csv --patients reproduced_results/section52_patient_burdens.csv --summary reproduced_results/section52_summary.csv
python code/plot_section52.py --raw reproduced_results/section52_runs.csv --patients reproduced_results/section52_patient_burdens.csv --outdir reproduced_results/section52_figures
```

Section 5.3:

```text
python code/run_section53_exact.py --output reproduced_results/section53_exact_runs.csv
python code/run_section53_kbrkga.py --output reproduced_results/section53_kbrkga_runs.csv --summary reproduced_results/section53_kbrkga_summary.csv
python code/summarize_section53.py --exact reproduced_results/section53_exact_runs.csv --kbrkga reproduced_results/section53_kbrkga_runs.csv --output-dir reproduced_results/section_5_3
```

Section 5.4:

```text
python code/run_section54.py --output reproduced_results/section54_runs.csv --summary reproduced_results/section54_summary.csv
```

The defaults in these four experiment runners match the reported scenario counts, scales, replications, time limits, thread count, and random seeds. Wall-clock time can vary with hardware, operating-system load, Gurobi version, and thread scheduling. Objective and validation fields should be used for substantive result checks.

Runtime records retain the observed wall-clock values. Wall-clock time may differ from the nominal solver time limit because of solver callbacks and post-solve bookkeeping.

The correspondence between every paper table or figure and its source file is listed in `verification/table_mapping.csv`.
