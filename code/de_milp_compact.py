"""Compact deterministic-equivalent formulation.

The formulation removes the position-pair variable
``zeta[h,i,m,j]``.  A scenario-specific patient completion variable is linked
to the patient's assigned position by indicators.  The next-stage waiting row
then needs only the current assignment ``x[h,i,j]``.  This changes the
cross-stage formulation from O(K*H*I^2*J) to O(K*H*I*J).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None

from de_milp_base import DeterministicEquivalentBase, DeterministicEquivalentResult


class CompactDeterministicEquivalent(DeterministicEquivalentBase):
    """Deterministic equivalent with compact predecessor-completion indicators."""

    def _make_previous_stage(self) -> np.ndarray:
        previous = np.full((self.H, self.J), -1, dtype=int)
        for j in range(self.J):
            stages = [h for h in range(self.H) if int(self.c[h, j]) == 1]
            for index in range(1, len(stages)):
                previous[stages[index], j] = stages[index - 1]
        return previous

    def add_indirect_waiting_row(
        self,
        h: int,
        i: int,
        j: int,
        k: int,
        name: Optional[str] = None,
    ):
        if self.model is None:
            raise RuntimeError("Build the model before adding waiting rows.")
        previous = int(self.previous_stage[h, j])
        if previous < 0:
            raise ValueError(f"Patient {j} has no predecessor at stage {h}.")
        return self.model.addGenConstrIndicator(
            self.x[h, i, j],
            True,
            self.arrival[h, i, k]
            - self.patient_completion[previous, j, k]
            + self.beta[h, i]
            <= self.theta[h, i, k],
            name=name or f"compact_indirect_{h}_{i}_{j}_{k}",
        )

    def indirect_waiting_violation_value(self, h: int, i: int, j: int, k: int) -> float:
        if float(self.x[h, i, j].X) < 0.5:
            return 0.0
        previous = int(self.previous_stage[h, j])
        return float(
            self.arrival[h, i, k].X
            - self.patient_completion[previous, j, k].X
            + self.beta[h, i].X
            - self.theta[h, i, k].X
        )

    def build(self, defer_waiting_rows: bool = False):
        if gp is None:
            raise RuntimeError("gurobipy is required to run CompactDeterministicEquivalent.")
        self.defer_waiting_rows = bool(defer_waiting_rows)
        m = gp.Model("DE_compact" if not defer_waiting_rows else "restricted_master_compact")
        self.model = m
        m.Params.OutputFlag = self.output_flag
        if self.time_limit is not None:
            m.Params.TimeLimit = float(self.time_limit)
        if self.mip_gap is not None:
            m.Params.MIPGap = float(self.mip_gap)
        if self.threads is not None:
            m.Params.Threads = int(self.threads)
        m.Params.Seed = self.seed

        H, I, J, K = self.H, self.I, self.J, self.K
        self.previous_stage = self._make_previous_stage()
        self.x = m.addVars(H, I, J, vtype=GRB.BINARY, name="x")
        self.y = m.addVars(H, I, lb=0.0, vtype=GRB.CONTINUOUS, name="y")
        self.alpha = m.addVars(H, I, lb=0.0, vtype=GRB.CONTINUOUS, name="alpha")
        self.beta = m.addVars(H, I, lb=0.0, vtype=GRB.CONTINUOUS, name="beta")
        self.gamma = m.addVars(H, I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="gamma")
        self.theta = m.addVars(H, I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="theta")
        self.arrival = m.addVars(H, I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="arrival")
        self.start = m.addVars(H, I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="start")
        self.completion = m.addVars(H, I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="completion")
        enforce_precedence = bool(getattr(self, "enforce_cross_stage_precedence", False))
        self.predecessor_completion = (
            m.addVars(H, I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="predecessor_completion")
            if enforce_precedence
            else None
        )
        self.ready = (
            m.addVars(H, I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="ready")
            if enforce_precedence
            else None
        )
        self.patient_completion = m.addVars(
            H, J, K, lb=0.0, vtype=GRB.CONTINUOUS, name="patient_completion"
        )
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
                        if enforce_precedence:
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
                # A first-entry patient has no inter-stage wait.  The equality is
                # imposed by assignment-dependent upper/lower bounds.
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
                        gp.quicksum(float(self.p[k]) * self.theta[h, i, k] for k in range(K))
                        <= rs_i,
                        name=f"theta_expectation_{h}_{i}",
                    )
                    m.addConstr(self.beta[h, i] >= rs_first, name=f"beta_first_entry_{h}_{i}")
                    for k in range(K):
                        m.addConstr(
                            self.theta[h, i, k] <= self.bigM * (1 - first_entry),
                            name=f"theta_first_entry_{h}_{i}_{k}",
                        )
            for i in range(Iv_h, I - 1):
                m.addConstr(self.y[h, i + 1] >= self.y[h, i], name=f"appointment_order_{h}_{i}")

        for h in range(H):
            Iv_h = self.Iv[h]
            for i in range(Iv_h, I):
                for k in range(K):
                    m.addConstr(
                        self.arrival[h, i, k] == self.y[h, i] + self.tard_expr(h, i, k),
                        name=f"arrival_def_{h}_{i}_{k}",
                    )
                    service = gp.quicksum(
                        float(self.s[k, h, i, j]) * self.x[h, i, j] for j in range(J)
                    )
                    if enforce_precedence:
                        m.addGenConstrMax(
                            self.ready[h, i, k],
                            [
                                self.arrival[h, i, k],
                                self.predecessor_completion[h, i, k],
                            ],
                            name=f"ready_max_{h}_{i}_{k}",
                        )
                        start_inputs = [
                            self.ready[h, i, k],
                        ]
                        if i > Iv_h:
                            start_inputs.append(self.completion[h, i - 1, k])
                        m.addGenConstrMax(
                            self.start[h, i, k],
                            start_inputs,
                            name=f"start_max_with_predecessor_{h}_{i}_{k}",
                        )
                    elif i == Iv_h:
                        m.addConstr(self.start[h, i, k] == self.arrival[h, i, k], name=f"start_first_{h}_{i}_{k}")
                    else:
                        m.addGenConstrMax(
                            self.start[h, i, k],
                            [self.arrival[h, i, k], self.completion[h, i - 1, k]],
                            name=f"start_max_{h}_{i}_{k}",
                        )
                    m.addConstr(
                        self.completion[h, i, k] == self.start[h, i, k] + service,
                        name=f"completion_def_{h}_{i}_{k}",
                    )
                    if not defer_waiting_rows:
                        self.add_direct_waiting_row(h, i, k)

        # Link each attended patient's scenario completion to exactly one
        # physical position.  This replaces all previous/current position pairs.
        for h in range(H):
            for j in range(J):
                if int(self.c[h, j]) == 0:
                    for k in range(K):
                        m.addConstr(self.patient_completion[h, j, k] == 0.0, name=f"pc_zero_{h}_{j}_{k}")
                    continue
                for i in range(self.Iv[h], I):
                    for k in range(K):
                        m.addGenConstrIndicator(
                            self.x[h, i, j],
                            True,
                            self.patient_completion[h, j, k] == self.completion[h, i, k],
                            name=f"pc_link_{h}_{i}_{j}_{k}",
                        )

        # The completion of the same patient's previous attended stage is a
        # third service-readiness term.  The selected value is linked through
        # the current assignment.
        if enforce_precedence:
            for h in range(H):
                for i in range(self.Iv[h], I):
                    for j in range(J):
                        if int(self.c[h, j]) != 1:
                            continue
                        previous = int(self.previous_stage[h, j])
                        for k in range(K):
                            rhs = (
                                self.patient_completion[previous, j, k]
                                if previous >= 0
                                else 0.0
                            )
                            m.addGenConstrIndicator(
                                self.x[h, i, j],
                                True,
                                self.predecessor_completion[h, i, k] == rhs,
                                name=f"pred_completion_link_{h}_{i}_{j}_{k}",
                            )

        if not defer_waiting_rows:
            for h in range(H):
                for i in range(self.Iv[h], I):
                    for j in range(J):
                        if int(self.c[h, j]) != 1 or int(self.previous_stage[h, j]) < 0:
                            continue
                        for k in range(K):
                            self.add_indirect_waiting_row(h, i, j, k)

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
        return m


CompactDeterministicEquivalent = CompactDeterministicEquivalent

__all__ = ["CompactDeterministicEquivalent", "CompactDeterministicEquivalent", "DeterministicEquivalentResult"]
