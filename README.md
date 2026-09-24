# Computational package for "Cluster-based Benders decomposition for multistage appointment scheduling"

This package contains the numerical instances, individual run records, summary tables, and validation checks for the computational study in Sections 5.2, 5.3, and 5.4.

The study is motivated by serial maternal and child healthcare pathways. The released numerical instances follow the pathway structure and controlled distributions stated in the paper and Appendix F. They do not contain patient-level hospital records.
 
## Package contents

- `data/section_5_2/` contains 120 method-instance runs, 480 patient-level burden records, and the 12-row summary used in Table 2 and Figures 3 and 4.
- `data/section_5_3/` contains 120 exact-method runs, 40 K-BRKGA runs, the combined 160-record file, and the summaries used in Tables 3, 4, and 5 and Appendix Tables F.2 to F.5.
- `data/section_5_4/` contains 180 individual runs and the 18-row summary used in Table 6.
- `instances/` contains 60 compressed base-instance files and a 260-row manifest. Prefix and lateness-intensity rules in the manifest identify every logical experimental instance.
- `figures/section_5_2/` contains the two figures reported in Section 5.2.
- `verification/` contains the mapping from the paper tables and figures to their source data.

## Software environment

The reported experiments used Python 3.12.3, NumPy 1.26.4, Matplotlib 3.10.8, and Gurobi 11.0.2 on Windows 11. The computer had an Intel Core i5-12400F processor with 6 physical cores, 12 logical cores, and 15.86 GiB of RAM. The MILP experiments used four threads.

## Instance and seed protocol

Section 5.2 uses ten replications with master seeds `2025 + 10r`, where `r=0,...,9`. The 500-scenario base draw for each replication is stored once. Instances with 15, 50, and 200 scenarios use the corresponding prefix.

Section 5.3 uses service-time seeds `12065 + 100r` and late-arrival seeds `12066 + 100r`. The same training and out-of-sample scenarios are used by all four methods for a given scale and replication.

Section 5.4 uses the same ten master seeds as Section 5.2. Each replication stores a 500-scenario base draw. Smaller scenario sets use prefixes, and the lateness multipliers are applied by the clipping rule recorded in `instances/manifest.csv`.

For Sections 5.2 and 5.4, the service and late-arrival arrays are produced by one deterministic scenario-generation call from the recorded master seed. The late-arrival generator applies its fixed internal offset. The `seed_service` and `seed_lateness` fields therefore both record that master seed.

## Experiment records

The stored records use the reported scenario counts, scales, replications, time limits, thread count, and random seeds. Wall-clock time can vary with hardware, operating-system load, Gurobi version, and thread scheduling. Objective and validation fields should be used for substantive result checks.

Runtime records retain the observed wall-clock values. Wall-clock time may differ from the nominal solver time limit because of solver callbacks and post-solve bookkeeping.

The correspondence between every paper table or figure and its source file is listed in `verification/table_mapping.csv`.
