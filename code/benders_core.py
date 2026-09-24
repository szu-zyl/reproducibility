"""Exact logic-based decomposition implementation.

The master contains a scenario-specific predecessor-completion variable for
each physical position.  Assignment indicators select the correct upstream
patient completion.  Hence one globally valid position--scenario logic row is
sufficient for inter-stage waiting and avoids a Cartesian product over current
patients and upstream positions.  These compact rows are
separated at integer incumbents with ``cbLazy``.

This is exact: no scenario is discarded, and an incumbent is accepted only
after every training scenario has been checked.  The CBBD variant uses clusters
only to decide which scenarios are checked first; it performs a full scan
whenever representative scenarios do not already reject the incumbent.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None

from de_milp_base import DeterministicEquivalentBase
from de_milp import (
    add_expected_pair_strengthening,
    add_expected_position_strengthening,
    apply_complete_natural_mip_start,
    apply_complete_schedule_mip_start,
    configure_model,
    natural_assignment,
)
from pathwise_primal_start import build_pathwise_primal_start
from fixed_schedule_validation import validate_fixed_schedule


CutKey = Tuple[int, int, int]  # stage h, current physical position i, scenario k
PhysicalKey = Tuple[int, int, int, int, int, int]
# current stage/position, previous stage/position, patient, scenario


@dataclass
class BendersResult:
    obj: float = math.inf
    bound: float = -math.inf
    status: str = "NOT_RUN"
    runtime_sec: float = 0.0
    build_sec: float = 0.0
    solve_sec: float = 0.0
    certified: bool = False
    full_separation: bool = False
    max_final_violation: float = math.inf
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    alpha: Optional[np.ndarray] = None
    beta: Optional[np.ndarray] = None
    callback_calls: int = 0
    accepted_incumbents: int = 0
    lazy_cuts: int = 0
    seed_cuts: int = 0
    aggregate_cuts: int = 0
    representative_screens: int = 0
    # Legacy field name retained for CSV compatibility.  For CBBD this counts
    # only nonsentinel checks; total work is representative_screens plus this.
    full_scenario_screens: int = 0
    full_scans: int = 0
    representative_rejections: int = 0
    repair_attempts: int = 0
    repair_feasible: int = 0
    repair_submissions: int = 0
    best_repaired_objective: float = math.inf
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
    best_accepted_objective: float = math.inf
    batch_initial: int = 0
    batch_final: int = 0
    cluster_count: int = 0
    num_vars: int = 0
    num_constrs: int = 0
    num_genconstrs: int = 0
    first_feasible_sec: float = math.nan
    progress: Dict[float, Tuple[float, float, float]] = field(default_factory=dict)
    incumbent_trace: List[Tuple[float, float]] = field(default_factory=list)


class BendersSolver(DeterministicEquivalentBase):
    """Restricted master with exact incumbent separation."""

    def __init__(
        self,
        *args,
        strategy: str = "lbbd",
        tol: float = 1e-6,
        cuts_per_callback: int = 48,
        max_cuts_per_callback: int = 256,
        batch_growth: float = 1.5,
        cluster_count: Optional[int] = None,
        seed_scenarios: Optional[int] = None,
        aggregate_strengthening: bool = False,
        incumbent_repair: bool = False,
        symmetry_breaking: bool = True,
        warm_start: bool = True,
        complete_warm_start: bool = False,
        no_rel_heur_time: float = 0.0,
        primal_phase_time: float = 0.0,
        path_start_time: float = 5.0,
        path_start_passes: int = 8,
        progress_checkpoints: Sequence[float] = (30.0, 60.0, 120.0),
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        strategy = str(strategy).lower()
        if strategy not in {"lbbd", "cbbd"}:
            raise ValueError("strategy must be 'lbbd' or 'cbbd'.")
        if self.I != self.J:
            raise ValueError("The implementation requires the paper's I=J permutation formulation.")
        self.strategy = strategy
        self.tol = float(tol)
        self.cuts_per_callback = max(1, int(cuts_per_callback))
        self.max_cuts_per_callback = max(self.cuts_per_callback, int(max_cuts_per_callback))
        self.batch_growth = max(1.0, float(batch_growth))
        self.current_batch = self.cuts_per_callback
        self.cluster_count_requested = cluster_count
        self.seed_scenarios_requested = seed_scenarios
        self.aggregate_strengthening = bool(aggregate_strengthening)
        self.incumbent_repair = bool(incumbent_repair)
        self.symmetry_breaking = bool(symmetry_breaking)
        self.warm_start = bool(warm_start)
        self.complete_warm_start_requested = bool(complete_warm_start)
        self.no_rel_heur_time = max(0.0, float(no_rel_heur_time))
        self.primal_phase_time = max(0.0, float(primal_phase_time))
        self.path_start_time = max(0.0, float(path_start_time))
        self.path_start_passes = max(0, int(path_start_passes))
        self.progress_checkpoints = tuple(
            sorted({float(value) for value in progress_checkpoints if float(value) > 0.0})
        )

        self.previous_stage = self._make_previous_stage()
        self.active_cuts: set[CutKey] = set()
        self.active_physical_cuts: set[PhysicalKey] = set()
        self.seed_cuts = 0
        self.aggregate_cuts = 0
        self.lazy_cuts = 0
        self.callback_calls = 0
        self.accepted_incumbents = 0
        self.representative_screens = 0
        self.full_scenario_screens = 0
        self.full_scans = 0
        self.representative_rejections = 0
        self.repair_attempts = 0
        self.repair_feasible = 0
        self.repair_submissions = 0
        self.best_repaired_objective = math.inf
        self.best_accepted_objective = math.inf
        self.best_accepted_x: Optional[np.ndarray] = None
        self.best_accepted_y: Optional[np.ndarray] = None
        self.best_accepted_alpha: Optional[np.ndarray] = None
        self.best_accepted_beta: Optional[np.ndarray] = None
        self._callback_error: Optional[BaseException] = None
        self._progress_start_time: Optional[float] = None
        self._wall_deadline: Optional[float] = None
        self.progress: Dict[float, Tuple[float, float, float]] = {}
        self.first_feasible_sec = math.nan
        self.incumbent_trace: List[Tuple[float, float]] = []
        self.path_start_sec = 0.0
        self.path_start_objective = math.nan
        self.path_start_evaluations = 0
        self.path_start_swaps = 0
        self.path_start_timing_moves = 0
        self.path_start_x: Optional[np.ndarray] = None
        self.path_start_y: Optional[np.ndarray] = None
        self.path_start_alpha: Optional[np.ndarray] = None
        self.path_start_beta: Optional[np.ndarray] = None

        self.scenario_features = self._make_scenario_features()
        self.scenario_score = np.sum(self.scenario_features, axis=1)
        self.clusters, self.representatives = self._make_clusters()
        self.cluster_count = len(self.clusters)
        self._active_x_vars: List = []
        self._active_x_meta: List[Tuple[int, int, int]] = []
        self._all_model_vars: List = []
        self._var_slot: Dict = {}
        self.first_entry_expr: Dict[Tuple[int, int], object] = {}

    def _make_previous_stage(self) -> np.ndarray:
        previous = np.full((self.H, self.J), -1, dtype=int)
        for j in range(self.J):
            stages = [h for h in range(self.H) if int(self.c[h, j]) == 1]
            for q in range(1, len(stages)):
                previous[stages[q], j] = stages[q - 1]
        return previous

    def _make_scenario_features(self) -> np.ndarray:
        features: List[List[float]] = []
        for k in range(self.K):
            row: List[float] = []
            for h in range(self.H):
                active_j = np.flatnonzero(self.c[h, :] == 1)
                values = self.s[k, h, self.Iv[h] :, :][:, active_j].ravel()
                row.extend(
                    [
                        float(np.mean(values)),
                        float(np.std(values)),
                        float(np.max(values)),
                    ]
                )
            row.extend(
                [
                    float(np.mean(self.tau[k, :])),
                    float(np.max(self.tau[k, :])),
                    float(np.mean(self.psi[k, :])),
                    float(np.max(self.psi[k, :])),
                ]
            )
            features.append(row)
        raw = np.asarray(features, dtype=float)
        scale = np.std(raw, axis=0)
        scale[scale < 1e-9] = 1.0
        return (raw - np.mean(raw, axis=0)) / scale

    def _make_clusters(self) -> Tuple[List[List[int]], List[int]]:
        if self.K <= 0:
            return [], []
        requested = self.cluster_count_requested
        q = int(math.ceil(math.sqrt(self.K))) if requested is None else int(requested)
        q = min(self.K, max(1, q))
        if q == self.K:
            clusters = [[k] for k in range(self.K)]
            return clusters, list(range(self.K))

        # Deterministic farthest-first partition.  Starting from a high-burden
        # scenario makes the resulting sentinels useful for early rejection.
        centers = [int(np.argmax(np.sum(self.scenario_features, axis=1)))]
        min_dist = np.sum((self.scenario_features - self.scenario_features[centers[0]]) ** 2, axis=1)
        while len(centers) < q:
            candidate = int(np.argmax(min_dist))
            if candidate in centers:
                break
            centers.append(candidate)
            distance = np.sum((self.scenario_features - self.scenario_features[candidate]) ** 2, axis=1)
            min_dist = np.minimum(min_dist, distance)

        center_matrix = self.scenario_features[centers, :]
        distances = np.sum(
            (self.scenario_features[:, None, :] - center_matrix[None, :, :]) ** 2,
            axis=2,
        )
        labels = np.argmin(distances, axis=1)
        clusters = [[int(k) for k in np.flatnonzero(labels == q_idx)] for q_idx in range(len(centers))]
        clusters = [cluster for cluster in clusters if cluster]
        # A sentinel is the most demanding member of its cluster, not a reduced
        # replacement for that cluster.  All other members remain in full scans.
        reps = [max(cluster, key=lambda k: float(self.scenario_score[k])) for cluster in clusters]
        return clusters, reps

    def tard_expr(self, h: int, i: int, k: int):
        return gp.quicksum(
            (
                float(self.tau[k, j]) * float(self.first_stage[h, j])
                + float(self.psi[k, j]) * float(self.follow_stage[h, j])
            )
            * self.x[h, i, j]
            for j in range(self.J)
        )

    def build(self):
        if gp is None:
            raise RuntimeError("gurobipy is required to run the decomposition models.")
        m = gp.Model("LBBD" if self.strategy == "lbbd" else "CBBD")
        self.model = m
        m.Params.OutputFlag = self.output_flag
        if self.time_limit is not None:
            m.Params.TimeLimit = float(self.time_limit)
        if self.mip_gap is not None:
            m.Params.MIPGap = float(self.mip_gap)
        if self.threads is not None:
            m.Params.Threads = int(self.threads)
        m.Params.Seed = self.seed
        m.Params.LazyConstraints = 1
        m.Params.PreCrush = 1

        H, I, J, K = self.H, self.I, self.J, self.K
        self.x = m.addVars(H, I, J, vtype=GRB.BINARY, name="x")
        self.y = m.addVars(H, I, lb=0.0, vtype=GRB.CONTINUOUS, name="y")
        self.alpha = m.addVars(H, I, lb=0.0, vtype=GRB.CONTINUOUS, name="alpha")
        self.beta = m.addVars(H, I, lb=0.0, vtype=GRB.CONTINUOUS, name="beta")
        self.gamma = m.addVars(H, I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="gamma")
        self.theta = m.addVars(H, I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="theta")
        self.arrival = m.addVars(H, I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="arrival")
        self.start = m.addVars(H, I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="start")
        self.completion = m.addVars(H, I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="completion")
        self.patient_completion = None
        self.predecessor_completion = m.addVars(
            H, I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="predecessor_completion"
        )
        self.ready = m.addVars(H, I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="ready")
        self.zeta = None

        for h in range(H):
            Iv_h = self.Iv[h]
            for i in range(I):
                m.addConstr(gp.quicksum(self.x[h, i, j] for j in range(J)) == 1, name=f"pos_{h}_{i}")
                if i < Iv_h:
                    m.addConstr(self.y[h, i] == 0.0, name=f"virt_y_{h}_{i}")
                    m.addConstr(self.alpha[h, i] == 0.0, name=f"virt_alpha_{h}_{i}")
                    m.addConstr(self.beta[h, i] == 0.0, name=f"virt_beta_{h}_{i}")
                    for k in range(K):
                        m.addConstr(self.gamma[h, i, k] == 0.0, name=f"virt_gamma_{h}_{i}_{k}")
                        m.addConstr(self.theta[h, i, k] == 0.0, name=f"virt_theta_{h}_{i}_{k}")
                        m.addConstr(self.arrival[h, i, k] == 0.0, name=f"virt_arrival_{h}_{i}_{k}")
                        m.addConstr(self.start[h, i, k] == 0.0, name=f"virt_start_{h}_{i}_{k}")
                        m.addConstr(self.completion[h, i, k] == 0.0, name=f"virt_completion_{h}_{i}_{k}")
                        m.addConstr(
                            self.predecessor_completion[h, i, k] == 0.0,
                            name=f"virt_predecessor_completion_{h}_{i}_{k}",
                        )
                        m.addConstr(
                            self.ready[h, i, k] == 0.0,
                            name=f"virt_ready_{h}_{i}_{k}",
                        )
                if self.L is not None:
                    m.addConstr(self.y[h, i] <= float(self.L[h]), name=f"appointment_ub_{h}_{i}")

            for j in range(J):
                m.addConstr(gp.quicksum(self.x[h, i, j] for i in range(I)) == 1, name=f"patient_{h}_{j}")
            for i in range(Iv_h):
                for j in range(J):
                    m.addConstr(self.x[h, i, j] <= 1 - int(self.c[h, j]), name=f"virt_only_{h}_{i}_{j}")
            for i in range(Iv_h, I):
                self._active_x_vars.extend(self.x[h, i, j] for j in range(J) if int(self.c[h, j]) == 1)
                self._active_x_meta.extend((h, i, j) for j in range(J) if int(self.c[h, j]) == 1)
                for j in range(J):
                    m.addConstr(self.x[h, i, j] <= int(self.c[h, j]), name=f"real_only_{h}_{i}_{j}")
                rd_i = gp.quicksum(float(self.r_d[j]) * self.x[h, i, j] for j in range(J))
                rs_i = gp.quicksum(float(self.r_s[j]) * self.x[h, i, j] for j in range(J))
                m.addConstr(self.alpha[h, i] <= rd_i, name=f"alpha_ub_{h}_{i}")
                m.addConstr(self.beta[h, i] <= rs_i, name=f"beta_ub_{h}_{i}")
                m.addConstr(
                    gp.quicksum(float(self.p[k]) * self.gamma[h, i, k] for k in range(K)) <= rd_i,
                    name=f"gamma_expectation_{h}_{i}",
                )
                first_entry = gp.quicksum(
                    self.x[h, i, j] for j in range(J) if int(self.previous_stage[h, j]) < 0
                )
                self.first_entry_expr[h, i] = first_entry
                rs_first = gp.quicksum(
                    float(self.r_s[j]) * self.x[h, i, j]
                    for j in range(J)
                    if int(self.previous_stage[h, j]) < 0
                )
                if h == 0:
                    m.addConstr(self.beta[h, i] == rs_i, name=f"beta_first_stage_{h}_{i}")
                    for k in range(K):
                        m.addConstr(self.theta[h, i, k] == 0.0, name=f"theta_first_stage_{h}_{i}_{k}")
                else:
                    m.addConstr(
                        gp.quicksum(float(self.p[k]) * self.theta[h, i, k] for k in range(K)) <= rs_i,
                        name=f"theta_expectation_{h}_{i}",
                    )
                    m.addConstr(self.beta[h, i] >= rs_first, name=f"beta_first_entry_{h}_{i}")
                    for k in range(K):
                        m.addConstr(
                            self.theta[h, i, k] <= self.bigM * (1 - first_entry),
                            name=f"theta_first_entry_{h}_{i}_{k}",
                        )
                for k in range(K):
                    m.addConstr(
                        self.predecessor_completion[h, i, k]
                        <= self.bigM * (1 - first_entry),
                        name=f"predecessor_first_entry_{h}_{i}_{k}",
                    )
            for i in range(Iv_h, I - 1):
                m.addConstr(self.y[h, i + 1] >= self.y[h, i], name=f"appointment_order_{h}_{i}")

        for h in range(H):
            for i in range(self.Iv[h], I):
                for k in range(K):
                    m.addConstr(
                        self.arrival[h, i, k] == self.y[h, i] + self.tard_expr(h, i, k),
                        name=f"arrival_def_{h}_{i}_{k}",
                    )
                    service = gp.quicksum(
                        float(self.s[k, h, i, j]) * self.x[h, i, j] for j in range(J)
                    )
                    m.addGenConstrMax(
                        self.ready[h, i, k],
                        [
                            self.arrival[h, i, k],
                            self.predecessor_completion[h, i, k],
                        ],
                        name=f"ready_max_{h}_{i}_{k}",
                    )
                    max_inputs = [self.ready[h, i, k]]
                    if i > self.Iv[h]:
                        max_inputs.append(self.completion[h, i - 1, k])
                    m.addGenConstrMax(
                        self.start[h, i, k],
                        max_inputs,
                        name=f"start_max_with_predecessor_{h}_{i}_{k}",
                    )
                    m.addConstr(
                        self.completion[h, i, k] == self.start[h, i, k] + service,
                        name=f"completion_def_{h}_{i}_{k}",
                    )
                    # Direct within-department waits are only O(KHI), are useful
                    # in the root relaxation, and are therefore kept explicitly.
                    self.add_direct_waiting_row(h, i, k, name=f"direct_wait_{h}_{i}_{k}")

        # The O(KJ^2) patient/position indicator layer is omitted from the
        # restricted master.  Its selected physical-precedence implication is
        # separated lazily for every integer incumbent and original scenario.

        base = gp.quicksum(
            (float(self.r_d[j]) + float(self.r_s[j])) * self.x[h, i, j]
            for h in range(H)
            for i in range(self.Iv[h], I)
            for j in range(J)
        )
        satisfaction = gp.quicksum(
            self.alpha[h, i] + self.beta[h, i]
            for h in range(H)
            for i in range(self.Iv[h], I)
        )
        m.setObjective(base - satisfaction, GRB.MINIMIZE)
        self.aggregate_cuts = (
            add_expected_pair_strengthening(
                self,
                prefix="lazy_expected_pair",
            )
            if self.aggregate_strengthening
            else 0
        )
        configure_model(
            self,
            symmetry_breaking=self.symmetry_breaking,
            warm_start=self.warm_start,
            complete_warm_start=self.complete_warm_start_requested,
            no_rel_heur_time=self.no_rel_heur_time,
        )
        self.aggregate_cuts += add_expected_position_strengthening(
            self, prefix="lazy_expected_position"
        )
        if self.path_start_time > 0.0:
            start_result = build_pathwise_primal_start(
                {
                    "H": self.H,
                    "I": self.I,
                    "J": self.J,
                    "K": self.K,
                    "c": self.c,
                    "s": self.s,
                    "tau": self.tau,
                    "psi": self.psi,
                    "p": self.p,
                    "r_d": self.r_d,
                    "r_s": self.r_s,
                    "Iv": self.Iv,
                    "L": self.L,
                },
                max_passes=self.path_start_passes,
                time_limit=self.path_start_time,
            )
            if start_result is not None:
                self.path_start_sec = start_result.runtime_sec
                self.path_start_objective = start_result.objective
                self.path_start_evaluations = start_result.evaluations
                self.path_start_swaps = start_result.accepted_swaps
                self.path_start_timing_moves = start_result.accepted_timing_moves
                self.path_start_x = np.zeros(
                    (self.H, self.I, self.J), dtype=float
                )
                for h in range(self.H):
                    for i in range(self.I):
                        self.path_start_x[
                            h, i, int(start_result.assignment[h, i])
                        ] = 1.0
                self.path_start_y = np.asarray(start_result.y, dtype=float)
                # Retain this independently feasible schedule as an external
                # incumbent.  It is compared with the fully separated solver
                # incumbent at reporting time, but is not used as a Cutoff:
                # cutoff filtering could suppress candidates before lazy
                # separation in the intentionally relaxed master.
                audit = validate_fixed_schedule(
                    {
                        "H": self.H,
                        "I": self.I,
                        "J": self.J,
                        "K": self.K,
                        "c": self.c,
                        "s": self.s,
                        "tau": self.tau,
                        "psi": self.psi,
                        "p": self.p,
                        "r_d": self.r_d,
                        "r_s": self.r_s,
                        "Iv": self.Iv,
                        "L": self.L,
                    },
                    self.path_start_x,
                    self.path_start_y,
                    tol=max(1e-8, self.tol),
                )
                self.path_start_alpha = audit.alpha
                self.path_start_beta = audit.beta
                self._add_start_physical_rows(start_result.assignment)
        self._add_all_compact_waiting_rows()
        # The restricted master has a deliberately weak lower bound before its
        # logic rows are separated.  Long rounds of generic root cutting do not
        # exploit that missing logic and delay integer incumbents.  Limit them
        # so branch-and-check can start promptly.
        self.model.Params.Cuts = 0
        return m

    def _add_all_compact_waiting_rows(self) -> None:
        """Keep the O(KJ) waiting epigraph explicit in the restricted master.

        The decomposed family is the much larger O(KJ^2) selected physical
        predecessor linkage.  Retaining these compact rows strengthens the
        root and lets a fully audited primal schedule be recognized directly.
        """
        for k in range(self.K):
            for h in range(1, self.H):
                for i in range(self.Iv[h], self.I):
                    key = (h, i, int(k))
                    self._add_compact_logic_row(
                        h, i, int(k), f"compact_wait_{h}_{i}_{k}"
                    )
                    self.active_cuts.add(key)
                    self.seed_cuts += 1

    def _add_start_physical_rows(self, assignment: np.ndarray) -> None:
        """Seed valid physical-link rows for the audited primal schedule.

        The rows are activated only if both appearances of the same patient
        occupy the seeded current/upstream positions.  They neither fix the
        schedule nor restrict any other position pair.
        """
        position = {
            (h, int(assignment[h, i])): int(i)
            for h in range(self.H)
            for i in range(self.I)
        }
        for h in range(1, self.H):
            for i in range(self.Iv[h], self.I):
                j = int(assignment[h, i])
                previous = int(self.previous_stage[h, j])
                if previous < 0:
                    continue
                previous_i = int(position[previous, j])
                inactive = self.bigM * (
                    2 - self.x[h, i, j] - self.x[previous, previous_i, j]
                )
                for k in range(self.K):
                    key = (h, i, previous, previous_i, j, k)
                    self.model.addConstr(
                        self.predecessor_completion[h, i, k]
                        - self.completion[previous, previous_i, k]
                        <= inactive,
                        name=f"physical_seed_up_{h}_{i}_{previous}_{previous_i}_{j}_{k}",
                    )
                    self.model.addConstr(
                        self.completion[previous, previous_i, k]
                        - self.predecessor_completion[h, i, k]
                        <= inactive,
                        name=f"physical_seed_down_{h}_{i}_{previous}_{previous_i}_{j}_{k}",
                    )
                    self.active_physical_cuts.add(key)
                    self.seed_cuts += 2

    def add_direct_waiting_row(self, h: int, i: int, k: int, name=None):
        """Link direct waiting to service start minus patient readiness."""
        if self.model is None:
            raise RuntimeError("Build the model before adding direct-waiting rows.")
        return self.model.addConstr(
            self.start[h, i, k] - self.ready[h, i, k] + self.alpha[h, i]
            <= self.gamma[h, i, k],
            name=name or f"direct_wait_{h}_{i}_{k}",
        )

    def _seed_scenario_indices(self) -> List[int]:
        """Return the common high-burden seed set used by LBBD and CBBD.

        Keeping this root set identical isolates the intended algorithmic
        difference: CBBD changes only the subsequent scenario inspection
        order.  Cluster representatives therefore remain sentinels in the
        callback, but do not define a different initial relaxation.
        """
        requested = self.seed_scenarios_requested
        if self.strategy == "cbbd" and requested is None:
            # CBBD starts with one actual high-burden member from each cluster.
            # These are ordinary exact logic rows, not averaged or reduced
            # scenarios.  Broad representative coverage also means subsequent
            # sentinel passes tend to trigger a full exact nonsentinel scan,
            # instead of repeatedly rejecting on a narrow local subset.
            return [int(k) for k in self.representatives]
        count = (
            min(self.K, max(1, int(math.ceil(math.sqrt(self.K)) / 2)))
            if requested is None
            else int(requested)
        )
        count = min(self.K, max(0, count))
        if count == 0:
            return []
        return [int(k) for k in np.argsort(-self.scenario_score)[:count]]

    def _add_compact_logic_row(self, h: int, i: int, k: int, name: str):
        """Add the position-level logic cut implied by selected precedence.

        ``predecessor_completion[h,i,k]`` already selects the completion of
        the patient assigned to position ``(h,i)`` at its previous attended
        stage.  Therefore no current-patient/upstream-position Cartesian
        product is needed in the waiting cut.  ``first_entry_expr`` deactivates
        the row only when that patient starts its pathway at stage ``h``.
        """
        return self.model.addConstr(
            self.arrival[h, i, k]
            - self.predecessor_completion[h, i, k]
            + self.beta[h, i]
            <= self.theta[h, i, k]
            + self.bigM * self.first_entry_expr[h, i],
            name=name,
        )

    def _add_seed_rows(self) -> None:
        """Seed a small common set of position-level logic rows.

        LBBD and CBBD receive exactly the same valid initial rows.  This keeps
        their root relaxation comparable.  Every other position--scenario row
        remains subject to exact lazy separation; the seed is unrelated to the
        independent K-BRKGA comparison.
        """
        for k in self._seed_scenario_indices():
            for h in range(self.H):
                for i in range(self.Iv[h], self.I):
                    if h == 0:
                        continue
                    key = (h, i, int(k))
                    if key in self.active_cuts:
                        continue
                    self._add_compact_logic_row(
                        h,
                        i,
                        int(k),
                        f"compact_seed_{h}_{i}_{k}",
                    )
                    self.active_cuts.add(key)
                    self.seed_cuts += 1

    def _incumbent_assignment(self, model) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]:
        values = model.cbGetSolution(self._active_x_vars)
        best: Dict[Tuple[int, int], Tuple[float, int]] = {}
        for (h, i, j), value in zip(self._active_x_meta, values):
            key = (h, i)
            if key not in best or float(value) > best[key][0]:
                best[key] = (float(value), int(j))
        patient_position: Dict[Tuple[int, int], int] = {}
        position_patient: Dict[Tuple[int, int], int] = {}
        for (h, i), (_, j) in best.items():
            position_patient[h, i] = j
            patient_position[h, j] = i
        return position_patient, patient_position

    def _scan(
        self,
        model,
        scenarios: Iterable[int],
        position_patient: Dict[Tuple[int, int], int],
        patient_position: Dict[Tuple[int, int], int],
    ) -> List[Tuple[float, CutKey]]:
        if self._deadline_reached():
            return []
        scenario_list = [int(k) for k in scenarios]
        if not scenario_list:
            return []
        metadata: List[CutKey] = []
        arrival_vars: List = []
        predecessor_vars: List = []
        beta_vars: List = []
        theta_vars: List = []
        for h in range(self.H):
            for i in range(self.Iv[h], self.I):
                j = int(position_patient[h, i])
                previous = int(self.previous_stage[h, j])
                if previous < 0:
                    continue
                for k in scenario_list:
                    key = (h, i, k)
                    # Gurobi may present a MIPSOL candidate that was generated
                    # before a previously submitted lazy row was fully applied.
                    # Therefore every incumbent must be checked again, even
                    # when this key has already been submitted.  ``active_cuts``
                    # is bookkeeping only; using it to skip separation can
                    # admit an invalid recourse allocation at a time limit.
                    metadata.append(key)
                    arrival_vars.append(self.arrival[h, i, k])
                    predecessor_vars.append(self.predecessor_completion[h, i, k])
                    beta_vars.append(self.beta[h, i])
                    theta_vars.append(self.theta[h, i, k])
        if not metadata:
            return []
        arrivals = model.cbGetSolution(arrival_vars)
        predecessors = model.cbGetSolution(predecessor_vars)
        betas = model.cbGetSolution(beta_vars)
        thetas = model.cbGetSolution(theta_vars)
        if self._deadline_reached():
            return []
        violations: List[Tuple[float, CutKey]] = []
        for key, arrival, predecessor, beta, theta in zip(
            metadata,
            arrivals,
            predecessors,
            betas,
            thetas,
        ):
            waiting_violation = (
                float(arrival) - float(predecessor) + float(beta) - float(theta)
            )
            if waiting_violation > self.tol:
                violations.append((waiting_violation, key))
        return violations

    def _scan_physical_links(
        self,
        model,
        position_patient: Dict[Tuple[int, int], int],
        patient_position: Dict[Tuple[int, int], int],
        scenarios: Iterable[int],
    ) -> List[Tuple[float, PhysicalKey]]:
        """Check selected predecessor completion against the physical path."""
        scenario_list = [int(k) for k in scenarios]
        metadata: List[PhysicalKey] = []
        predecessor_vars: List = []
        completion_vars: List = []
        for h in range(1, self.H):
            for i in range(self.Iv[h], self.I):
                j = int(position_patient[h, i])
                previous = int(self.previous_stage[h, j])
                if previous < 0:
                    continue
                previous_i = int(patient_position[previous, j])
                for k in scenario_list:
                    metadata.append((h, i, previous, previous_i, j, k))
                    predecessor_vars.append(self.predecessor_completion[h, i, k])
                    completion_vars.append(self.completion[previous, previous_i, k])
        if not metadata:
            return []
        predecessor_values = model.cbGetSolution(predecessor_vars)
        completion_values = model.cbGetSolution(completion_vars)
        return [
            (abs(float(predecessor) - float(completion)), key)
            for key, predecessor, completion in zip(
                metadata, predecessor_values, completion_values
            )
            if abs(float(predecessor) - float(completion)) > self.tol
        ]

    def _add_physical_lazy(self, model, key: PhysicalKey) -> None:
        h, i, previous, previous_i, j, k = key
        inactive = self.bigM * (
            2 - self.x[h, i, j] - self.x[previous, previous_i, j]
        )
        model.cbLazy(
            self.predecessor_completion[h, i, k]
            - self.completion[previous, previous_i, k]
            <= inactive
        )
        model.cbLazy(
            self.completion[previous, previous_i, k]
            - self.predecessor_completion[h, i, k]
            <= inactive
        )
        self.active_physical_cuts.add(key)
        self.lazy_cuts += 2

    def _select_batch(self, candidates: List[Tuple[float, CutKey]]) -> List[Tuple[float, CutKey]]:
        ordered = sorted(candidates, key=lambda item: item[0], reverse=True)
        limit = min(len(ordered), self.current_batch)
        if len(ordered) <= limit:
            return ordered

        selected: List[Tuple[float, CutKey]] = []
        selected_keys: set[CutKey] = set()
        seen_scenarios: set[int] = set()
        seen_positions: set[Tuple[int, int]] = set()
        for item in ordered:
            key = item[1]
            if key[2] not in seen_scenarios:
                selected.append(item)
                selected_keys.add(key)
                seen_scenarios.add(key[2])
                if len(selected) == limit:
                    return selected
        for item in ordered:
            key = item[1]
            position = key[:2]
            if key not in selected_keys and position not in seen_positions:
                selected.append(item)
                selected_keys.add(key)
                seen_positions.add(position)
                if len(selected) == limit:
                    return selected
        for item in ordered:
            if item[1] not in selected_keys:
                selected.append(item)
                if len(selected) == limit:
                    return selected
        return selected

    def _add_lazy(self, model, key: CutKey) -> None:
        h, i, k = key
        model.cbLazy(
            self.arrival[h, i, k]
            - self.predecessor_completion[h, i, k]
            + self.beta[h, i]
            <= self.theta[h, i, k]
            + self.bigM * self.first_entry_expr[h, i]
        )
        self.active_cuts.add(key)
        self.lazy_cuts += 1

    def _repair_incumbent(
        self,
        model,
        position_patient: Dict[Tuple[int, int], int],
        patient_position: Dict[Tuple[int, int], int],
    ) -> bool:
        """Complete the incumbent with exact best alpha/beta recourse for x/y.

        For fixed x/y, direct wait is known and
        ``alpha = r_d - E[direct_wait]``.  For a follow-up visit, the largest
        feasible beta solves ``E[max(0, raw_indirect_wait + beta)] <= r_s``.
        This one-dimensional monotone equation is solved by bisection.  The
        resulting full solution is submitted back to Gurobi as a feasible
        incumbent without changing the exact branch-and-bound search.
        """
        self.repair_attempts += 1
        if self._deadline_reached():
            return False
        values = list(model.cbGetSolution(self._all_model_vars))

        def value(var) -> float:
            return float(values[self._var_slot[var]])

        def set_value(var, new_value: float) -> None:
            values[self._var_slot[var]] = float(new_value)

        satisfaction = 0.0
        for h in range(self.H):
            for i in range(self.Iv[h], self.I):
                if self._deadline_reached():
                    return False
                j = int(position_patient[h, i])
                direct = np.asarray(
                    [
                        max(
                            0.0,
                            value(self.start[h, i, k]) - value(self.ready[h, i, k]),
                        )
                        for k in range(self.K)
                    ],
                    dtype=float,
                )
                alpha_value = min(
                    float(self.r_d[j]),
                    max(0.0, float(self.r_d[j]) - float(np.dot(self.p, direct))),
                )
                set_value(self.alpha[h, i], alpha_value)
                satisfaction += alpha_value
                for k in range(self.K):
                    set_value(self.gamma[h, i, k], direct[k] + alpha_value)

                previous = int(self.previous_stage[h, j])
                if previous < 0:
                    beta_value = float(self.r_s[j])
                    set_value(self.beta[h, i], beta_value)
                    satisfaction += beta_value
                    for k in range(self.K):
                        set_value(self.theta[h, i, k], 0.0)
                    continue

                m_idx = int(patient_position[previous, j])
                raw = np.asarray(
                    [
                        value(self.arrival[h, i, k])
                        - value(self.completion[previous, m_idx, k])
                        for k in range(self.K)
                    ],
                    dtype=float,
                )
                tolerance = float(self.r_s[j])

                def expected_required(beta_value: float) -> float:
                    return float(np.dot(self.p, np.maximum(0.0, raw + beta_value)))

                if expected_required(0.0) > tolerance + max(1e-7, 10.0 * self.tol):
                    return False
                if expected_required(tolerance) <= tolerance:
                    beta_value = tolerance
                else:
                    lower, upper = 0.0, tolerance
                    for _ in range(45):
                        middle = 0.5 * (lower + upper)
                        if expected_required(middle) <= tolerance:
                            lower = middle
                        else:
                            upper = middle
                    beta_value = lower
                set_value(self.beta[h, i], beta_value)
                satisfaction += beta_value
                required_theta = np.maximum(0.0, raw + beta_value)
                for k in range(self.K):
                    set_value(self.theta[h, i, k], required_theta[k])

        self.repair_feasible += 1
        base = float(
            sum(
                float(self.r_d[j]) + float(self.r_s[j])
                for h in range(self.H)
                for j in range(self.J)
                if int(self.c[h, j]) == 1
            )
        )
        repaired_objective = base - satisfaction
        self.best_repaired_objective = min(self.best_repaired_objective, repaired_objective)
        model.cbSetSolution(self._all_model_vars, values)
        self.repair_submissions += 1
        if not self._deadline_reached():
            try:
                model.cbUseSolution()
            except gp.GurobiError:
                # At MIPSOL, submitted values are processed when the callback
                # returns on Gurobi versions that do not allow cbUseSolution here.
                pass
        return True

    def _deadline_reached(self, reserve_sec: float = 10.0) -> bool:
        """Reserve callback-return time inside the build-plus-solve wall budget."""
        return bool(
            self._wall_deadline is not None
            and time.time() >= self._wall_deadline - max(0.0, float(reserve_sec))
        )

    @staticmethod
    def _progress_gap(obj: float, bound: float) -> float:
        if not np.isfinite(obj) or not np.isfinite(bound):
            return math.nan
        return max(0.0, float(obj - bound)) / max(1.0, abs(float(obj)))

    def _record_mip_progress(self, model) -> None:
        if self._progress_start_time is None:
            return
        elapsed = float(time.time() - self._progress_start_time)
        try:
            bound = float(model.cbGet(GRB.Callback.MIP_OBJBND))
        except (gp.GurobiError, AttributeError):
            bound = math.nan
        known_objectives = [
            value
            for value in (
                self.best_accepted_objective,
                self.path_start_objective,
            )
            if np.isfinite(value)
        ]
        objective = min(known_objectives) if known_objectives else math.nan
        for checkpoint in self.progress_checkpoints:
            if checkpoint not in self.progress and elapsed >= checkpoint:
                self.progress[checkpoint] = (
                    objective,
                    bound,
                    self._progress_gap(objective, bound),
                )

    def _callback(self, model, where) -> None:
        if where == GRB.Callback.MIP:
            self._record_mip_progress(model)
            if self._deadline_reached():
                model.terminate()
            return
        if where != GRB.Callback.MIPSOL:
            return
        try:
            self.callback_calls += 1
            if self._deadline_reached():
                model.terminate()
                return
            position_patient, patient_position = self._incumbent_assignment(model)
            if self.strategy == "cbbd":
                physical_candidates = self._scan_physical_links(
                    model,
                    position_patient,
                    patient_position,
                    self.representatives,
                )
                self.representative_screens += len(self.representatives)
                if physical_candidates:
                    self.representative_rejections += 1
                else:
                    remaining = [
                        k for k in range(self.K)
                        if k not in set(self.representatives)
                    ]
                    physical_candidates = self._scan_physical_links(
                        model,
                        position_patient,
                        patient_position,
                        remaining,
                    )
                    self.full_scenario_screens += len(remaining)
                    self.full_scans += 1
            else:
                physical_candidates = self._scan_physical_links(
                    model,
                    position_patient,
                    patient_position,
                    range(self.K),
                )
                self.full_scenario_screens += self.K
                self.full_scans += 1
            if physical_candidates:
                for _, key in sorted(
                    physical_candidates, key=lambda item: item[0], reverse=True
                )[: self.current_batch]:
                    if self._deadline_reached():
                        model.terminate()
                        return
                    self._add_physical_lazy(model, key)
                if (
                    len(physical_candidates) > self.current_batch
                    and self.current_batch < self.max_cuts_per_callback
                ):
                    self.current_batch = min(
                        self.max_cuts_per_callback,
                        max(
                            self.current_batch + 1,
                            int(math.ceil(self.current_batch * self.batch_growth)),
                        ),
                    )
                return
            if self.incumbent_repair:
                self._repair_incumbent(
                    model,
                    position_patient,
                    patient_position,
                )
            if self._deadline_reached():
                model.terminate()
                return
            # All compact waiting rows are explicit in the master.  Once
            # selected physical links pass, no second scenario scan is needed.
            candidates: List[Tuple[float, CutKey]] = []

            if self._deadline_reached():
                model.terminate()
                return

            if not candidates:
                self.accepted_incumbents += 1
                if not np.isfinite(self.first_feasible_sec):
                    self.first_feasible_sec = float(
                        time.time() - self._progress_start_time
                    ) if self._progress_start_time is not None else math.nan
                candidate_objective = float(
                    model.cbGet(GRB.Callback.MIPSOL_OBJ)
                )
                if candidate_objective < self.best_accepted_objective - 1e-10:
                    elapsed = float(
                        time.time() - self._progress_start_time
                    ) if self._progress_start_time is not None else math.nan
                    self.incumbent_trace.append((elapsed, candidate_objective))
                    values = list(model.cbGetSolution(self._all_model_vars))

                    def value(var) -> float:
                        return float(values[self._var_slot[var]])

                    accepted_x = np.zeros(
                        (self.H, self.I, self.J), dtype=float
                    )
                    accepted_y = np.zeros((self.H, self.I), dtype=float)
                    accepted_alpha = np.zeros((self.H, self.I), dtype=float)
                    accepted_beta = np.zeros((self.H, self.I), dtype=float)
                    for h in range(self.H):
                        for i in range(self.I):
                            accepted_y[h, i] = value(self.y[h, i])
                            accepted_alpha[h, i] = value(self.alpha[h, i])
                            accepted_beta[h, i] = value(self.beta[h, i])
                            for j in range(self.J):
                                accepted_x[h, i, j] = value(self.x[h, i, j])
                    self.best_accepted_objective = candidate_objective
                    self.best_accepted_x = accepted_x
                    self.best_accepted_y = accepted_y
                    self.best_accepted_alpha = accepted_alpha
                    self.best_accepted_beta = accepted_beta
                return
            selected = self._select_batch(candidates)
            for _, key in selected:
                if self._deadline_reached():
                    model.terminate()
                    return
                self._add_lazy(model, key)
            if len(candidates) > self.current_batch and self.current_batch < self.max_cuts_per_callback:
                grown = max(self.current_batch + 1, int(math.ceil(self.current_batch * self.batch_growth)))
                self.current_batch = min(self.max_cuts_per_callback, grown)
        except BaseException as exc:  # Gurobi otherwise only prints callback errors.
            self._callback_error = exc
            model.terminate()

    def audit_incumbent(self) -> float:
        """Audit existence of feasible gamma/theta recourse for final x/y.

        ``gamma`` and ``theta`` are cost-free allocation variables: several
        scenario-wise allocations can represent the same feasible expected
        wait.  Gurobi may reconstruct a different degenerate allocation after
        presolve, so auditing an arbitrary stored theta value row by row is too
        strict.  The exact feasibility test is whether the probability-weighted
        *minimal required* slacks fit their assigned tolerance budgets.
        """
        if self.model is None or self.model.SolCount <= 0:
            return math.inf
        position_patient: Dict[Tuple[int, int], int] = {}
        patient_position: Dict[Tuple[int, int], int] = {}
        for h in range(self.H):
            for i in range(self.Iv[h], self.I):
                j = max(
                    (jj for jj in range(self.J) if int(self.c[h, jj]) == 1),
                    key=lambda jj: float(self.x[h, i, jj].X),
                )
                position_patient[h, i] = int(j)
                patient_position[h, int(j)] = i
        worst = 0.0
        # Direct-wait recourse feasibility.
        for h in range(self.H):
            for i in range(self.Iv[h], self.I):
                j = position_patient[h, i]
                required = [
                    max(
                        0.0,
                        float(self.start[h, i, k].X)
                        - float(self.ready[h, i, k].X)
                        + float(self.alpha[h, i].X),
                    )
                    for k in range(self.K)
                ]
                excess = float(np.dot(self.p, required) - float(self.r_d[j]))
                worst = max(worst, excess)

        # Cross-stage-wait recourse feasibility.
        for h in range(self.H):
            for i in range(self.Iv[h], self.I):
                j = position_patient[h, i]
                previous = int(self.previous_stage[h, j])
                if previous < 0:
                    continue
                m_idx = patient_position[previous, j]
                for k in range(self.K):
                    selected_completion = float(
                        self.completion[previous, m_idx, k].X
                    )
                    predecessor_value = float(
                        self.predecessor_completion[h, i, k].X
                    )
                    same_stage_completion = (
                        float(self.completion[h, i - 1, k].X)
                        if i > self.Iv[h]
                        else 0.0
                    )
                    expected_start = max(
                        float(self.arrival[h, i, k].X),
                        selected_completion,
                        same_stage_completion,
                    )
                    worst = max(
                        worst,
                        abs(predecessor_value - selected_completion),
                        abs(float(self.start[h, i, k].X) - expected_start),
                    )
                required = [
                    max(
                        0.0,
                        float(self.arrival[h, i, k].X)
                        - float(self.completion[previous, m_idx, k].X)
                        + float(self.beta[h, i].X),
                    )
                    for k in range(self.K)
                ]
                excess = float(np.dot(self.p, required) - float(self.r_s[j]))
                worst = max(worst, excess)
        return float(max(0.0, worst))

    def solve(self) -> BendersResult:
        t0 = time.time()
        self._progress_start_time = t0
        self._wall_deadline = (
            t0 + float(self.time_limit) if self.time_limit is not None else None
        )
        if self.model is None:
            self.build()
        self.model.update()
        self._all_model_vars = list(self.model.getVars())
        self._var_slot = {var: index for index, var in enumerate(self._all_model_vars)}
        build_sec = float(time.time() - t0)
        if self.time_limit is not None:
            remaining = float(self.time_limit) - build_sec
            if remaining <= 0.0:
                return BendersResult(
                    status="BUILD_TIME_LIMIT",
                    runtime_sec=time.time() - t0,
                    build_sec=build_sec,
                )
            self.model.Params.TimeLimit = max(0.01, remaining)
        solve_start = time.time()
        primal_phase_sec = 0.0
        primal_phase_candidate = False
        primal_phase_feasible_start = False
        primal_phase_objective = math.nan
        if self.primal_phase_time > 0.0:
            available = (
                float(self.time_limit) - (time.time() - t0)
                if self.time_limit is not None
                else self.primal_phase_time
            )
            phase_budget = min(self.primal_phase_time, max(0.0, available - 1.0))
            if phase_budget > 0.01:
                phase_start = time.time()
                self.model.Params.TimeLimit = phase_budget
                # Solve the seeded restricted master first.  Its solution is
                # never reported as an exact result; it is used only as an x/y
                # proposal for the subsequent fully separated solve.
                self.model.optimize()
                primal_phase_sec = float(time.time() - phase_start)
                if self.model.SolCount > 0:
                    primal_phase_candidate = True
                    primal_phase_objective = float(self.model.ObjVal)
                    candidate_assignment = np.empty(
                        (self.H, self.I), dtype=int
                    )
                    candidate_y = np.zeros((self.H, self.I), dtype=float)
                    for h in range(self.H):
                        for i in range(self.I):
                            candidate_assignment[h, i] = max(
                                range(self.J),
                                key=lambda j: float(self.x[h, i, j].X),
                            )
                            candidate_y[h, i] = float(self.y[h, i].X)
                self.model.reset()
                # Restore the common exact start first; overwrite it only when
                # the restricted-master candidate has feasible full recourse.
                apply_complete_natural_mip_start(self)
                if primal_phase_candidate:
                    primal_phase_feasible_start = (
                        apply_complete_schedule_mip_start(
                            self,
                            candidate_assignment,
                            candidate_y,
                            tol=max(1e-8, self.tol),
                        )
                    )
                if self.time_limit is not None:
                    remaining = float(self.time_limit) - (time.time() - t0)
                    if remaining <= 0.0:
                        return BendersResult(
                            status="PRIMAL_PHASE_TIME_LIMIT",
                            runtime_sec=float(time.time() - t0),
                            build_sec=build_sec,
                            solve_sec=float(time.time() - solve_start),
                            primal_phase_sec=primal_phase_sec,
                            primal_phase_candidate=primal_phase_candidate,
                            primal_phase_feasible_start=primal_phase_feasible_start,
                            primal_phase_objective=primal_phase_objective,
                        )
                    self.model.Params.TimeLimit = max(0.01, remaining)
        self.model.optimize(self._callback)
        runtime = float(time.time() - t0)
        if self._callback_error is not None:
            raise RuntimeError("The decomposition callback failed.") from self._callback_error
        result = BendersResult(
            status=f"STATUS_{self.model.Status}",
            runtime_sec=runtime,
            build_sec=build_sec,
            solve_sec=float(time.time() - solve_start),
            callback_calls=self.callback_calls,
            accepted_incumbents=self.accepted_incumbents,
            lazy_cuts=self.lazy_cuts,
            seed_cuts=self.seed_cuts,
            aggregate_cuts=self.aggregate_cuts,
            representative_screens=self.representative_screens,
            full_scenario_screens=self.full_scenario_screens,
            full_scans=self.full_scans,
            representative_rejections=self.representative_rejections,
            repair_attempts=self.repair_attempts,
            repair_feasible=self.repair_feasible,
            repair_submissions=self.repair_submissions,
            best_repaired_objective=self.best_repaired_objective,
            primal_phase_sec=primal_phase_sec,
            primal_phase_candidate=primal_phase_candidate,
            primal_phase_feasible_start=primal_phase_feasible_start,
            primal_phase_objective=primal_phase_objective,
            path_start_sec=self.path_start_sec,
            path_start_objective=self.path_start_objective,
            path_start_evaluations=self.path_start_evaluations,
            path_start_swaps=self.path_start_swaps,
            path_start_timing_moves=self.path_start_timing_moves,
            best_accepted_objective=self.best_accepted_objective,
            batch_initial=self.cuts_per_callback,
            batch_final=self.current_batch,
            cluster_count=self.cluster_count if self.strategy == "cbbd" else 0,
            num_vars=int(self.model.NumVars),
            num_constrs=int(self.model.NumConstrs),
            num_genconstrs=int(self.model.NumGenConstrs),
            first_feasible_sec=self.first_feasible_sec,
            progress=dict(self.progress),
            incumbent_trace=list(self.incumbent_trace),
        )
        if self.model.SolCount <= 0:
            if (
                self.path_start_x is not None
                and self.path_start_y is not None
                and np.isfinite(self.path_start_objective)
            ):
                result.obj = float(self.path_start_objective)
                result.bound = float(self.model.ObjBound)
                result.x = self.path_start_x.copy()
                result.y = self.path_start_y.copy()
                result.alpha = (
                    None if self.path_start_alpha is None else self.path_start_alpha.copy()
                )
                result.beta = (
                    None if self.path_start_beta is None else self.path_start_beta.copy()
                )
                result.max_final_violation = 0.0
                result.full_separation = True
                result.accepted_fallback_used = True
                result.first_feasible_sec = self.path_start_sec
                for checkpoint in self.progress_checkpoints:
                    if checkpoint not in result.progress:
                        result.progress[checkpoint] = (
                            result.obj,
                            result.bound,
                            self._progress_gap(result.obj, result.bound),
                        )
            return result
        x, y, alpha, beta = self.extract()
        result.obj = float(self.model.ObjVal)
        result.bound = float(self.model.ObjBound)
        result.x, result.y, result.alpha, result.beta = x, y, alpha, beta
        result.max_final_violation = self.audit_incumbent()
        result.full_separation = result.max_final_violation <= max(1e-5, 10.0 * self.tol)
        if (
            not result.full_separation
            and self.best_accepted_x is not None
            and np.isfinite(self.best_accepted_objective)
        ):
            # At a time limit Gurobi can expose the last candidate values even
            # though the callback rejected that candidate with lazy cuts.
            # Report only the best incumbent explicitly accepted after a full
            # scenario scan.
            result.obj = float(self.best_accepted_objective)
            result.x = self.best_accepted_x.copy()
            result.y = self.best_accepted_y.copy()
            result.alpha = self.best_accepted_alpha.copy()
            result.beta = self.best_accepted_beta.copy()
            result.max_final_violation = 0.0
            result.full_separation = True
            result.accepted_fallback_used = True
        if (
            self.path_start_x is not None
            and self.path_start_y is not None
            and np.isfinite(self.path_start_objective)
            and (
                not result.full_separation
                or self.path_start_objective < result.obj - 1e-10
            )
        ):
            # Method-level incumbent: the pathwise schedule was evaluated by
            # the exact full-scenario recourse oracle before branch-and-bound.
            # It is therefore legitimate to report it if it dominates the best
            # solver incumbent found within the common wall-clock budget.
            result.obj = float(self.path_start_objective)
            result.x = self.path_start_x.copy()
            result.y = self.path_start_y.copy()
            result.alpha = (
                None if self.path_start_alpha is None else self.path_start_alpha.copy()
            )
            result.beta = (
                None if self.path_start_beta is None else self.path_start_beta.copy()
            )
            result.max_final_violation = 0.0
            result.full_separation = True
            result.accepted_fallback_used = True
        gap = max(0.0, result.obj - result.bound) / max(1.0, abs(result.obj))
        target_gap = 1e-4 if self.mip_gap is None else float(self.mip_gap)
        result.certified = bool(
            result.full_separation
            and (self.model.Status == GRB.OPTIMAL or gap <= target_gap + 1e-12)
        )
        if not np.isfinite(result.first_feasible_sec) and result.full_separation:
            result.first_feasible_sec = runtime
        if np.isfinite(self.path_start_objective):
            result.first_feasible_sec = min(
                float(result.first_feasible_sec),
                float(self.path_start_sec),
            )
        for checkpoint in self.progress_checkpoints:
            if checkpoint not in result.progress:
                result.progress[checkpoint] = (
                    result.obj if np.isfinite(result.obj) else math.nan,
                    result.bound if np.isfinite(result.bound) else math.nan,
                    self._progress_gap(result.obj, result.bound),
                )
        return result


__all__ = ["BendersSolver", "BendersResult"]
