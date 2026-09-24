"""Base deterministic-equivalent MILP for multistage appointment scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:  # pragma: no cover
    gp = None
    GRB = None


def _first_follow(c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    H, J = c.shape
    first = np.zeros((H, J), dtype=int)
    follow = np.zeros((H, J), dtype=int)
    for j in range(J):
        stages = [h for h in range(H) if int(c[h, j]) == 1]
        if stages:
            first[stages[0], j] = 1
            for h in stages[1:]:
                follow[h, j] = 1
    return first, follow


@dataclass
class DeterministicEquivalentResult:
    obj: float
    bound: float
    x: np.ndarray
    y: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray


class DeterministicEquivalentBase:
    def __init__(
        self,
        s: np.ndarray,
        tau: np.ndarray,
        psi: np.ndarray,
        c: np.ndarray,
        r_d: np.ndarray,
        r_s: np.ndarray,
        Iv: Sequence[int],
        L: Optional[Sequence[float]] = None,
        p: Optional[np.ndarray] = None,
        bigM: float = 1e5,
        time_limit: Optional[float] = None,
        mip_gap: Optional[float] = 1e-4,
        output_flag: int = 0,
        threads: Optional[int] = 1,
        seed: int = 2025,
    ) -> None:
        if gp is None:
            raise RuntimeError("gurobipy is required to run DeterministicEquivalentBase.")
        self.s = np.asarray(s, dtype=float)
        self.tau = np.asarray(tau, dtype=float)
        self.psi = np.asarray(psi, dtype=float)
        self.c = np.asarray(c, dtype=int)
        self.r_d = np.asarray(r_d, dtype=float)
        self.r_s = np.asarray(r_s, dtype=float)
        self.Iv = [int(v) for v in Iv]
        self.L = None if L is None else np.asarray(L, dtype=float)
        self.K, self.H, self.I, self.J = self.s.shape
        if self.tau.shape != (self.K, self.J) or self.psi.shape != (self.K, self.J):
            raise ValueError("tau and psi must have shape (K,J).")
        if self.c.shape != (self.H, self.J):
            raise ValueError("c must have shape (H,J).")
        if len(self.Iv) != self.H:
            raise ValueError("Iv must have length H.")
        if self.L is not None and self.L.shape != (self.H,):
            raise ValueError("L must have length H.")
        if p is None:
            self.p = np.full(self.K, 1.0 / self.K, dtype=float)
        else:
            self.p = np.asarray(p, dtype=float)
            self.p = self.p / np.sum(self.p)

        self.bigM = float(bigM)
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.output_flag = int(output_flag)
        self.threads = threads
        self.seed = int(seed)
        self.first_stage, self.follow_stage = _first_follow(self.c)

        self.model: Optional[gp.Model] = None
        self.x = None
        self.y = None
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.theta = None
        self.zeta = None
        self.arrival = None
        self.start = None
        self.completion = None
        self.defer_waiting_rows = False

    def tard_expr(self, h: int, i: int, k: int) -> gp.LinExpr:
        return gp.quicksum(
            (
                float(self.tau[k, j]) * float(self.first_stage[h, j])
                + float(self.psi[k, j]) * float(self.follow_stage[h, j])
            )
            * self.x[h, i, j]
            for j in range(self.J)
        )

    def add_direct_waiting_row(self, h: int, i: int, k: int, name: Optional[str] = None):
        if self.model is None:
            raise RuntimeError("Build the model before adding waiting rows.")
        return self.model.addConstr(
            self.start[h, i, k] - self.arrival[h, i, k] + self.alpha[h, i]
            <= self.gamma[h, i, k],
            name=name or f"base_direct_{h}_{i}_{k}",
        )

    def add_indirect_waiting_row(
        self,
        h: int,
        i: int,
        m_idx: int,
        j: int,
        k: int,
        name: Optional[str] = None,
    ):
        if self.model is None:
            raise RuntimeError("Build the model before adding waiting rows.")
        tard_i = self.tard_expr(h, i, k)
        return self.model.addConstr(
            self.y[h, i]
            - self.completion[h - 1, m_idx, k]
            - tard_i
            + self.beta[h, i]
            <= self.theta[h, i, k] + self.bigM * (1 - self.zeta[h, i, m_idx, j]),
            name=name or f"base_indirect_{h}_{i}_{m_idx}_{j}_{k}",
        )

    def indirect_waiting_violation_value(self, h: int, i: int, m_idx: int, j: int, k: int) -> float:
        tard_i = float(self.arrival[h, i, k].X - self.y[h, i].X)
        return float(
            self.y[h, i].X
            - self.completion[h - 1, m_idx, k].X
            - tard_i
            + self.beta[h, i].X
            - self.theta[h, i, k].X
            - self.bigM * (1.0 - self.zeta[h, i, m_idx, j].X)
        )

    def build(self, defer_waiting_rows: bool = False) -> gp.Model:
        self.defer_waiting_rows = bool(defer_waiting_rows)
        m = gp.Model("DE_base" if not defer_waiting_rows else "restricted_master_base")
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
        self.x = m.addVars(H, I, J, vtype=GRB.BINARY, name="x")
        self.y = m.addVars(H, I, lb=0.0, vtype=GRB.CONTINUOUS, name="y")
        self.alpha = m.addVars(H, I, lb=0.0, vtype=GRB.CONTINUOUS, name="alpha")
        self.beta = m.addVars(H, I, lb=0.0, vtype=GRB.CONTINUOUS, name="beta")
        self.gamma = m.addVars(H, I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="gamma")
        self.theta = m.addVars(H, I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="theta")
        self.zeta = m.addVars(H, I, I, J, vtype=GRB.BINARY, name="zeta")
        self.arrival = m.addVars(H, I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="arrival")
        self.start = m.addVars(H, I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="start")
        self.completion = m.addVars(H, I, K, lb=0.0, vtype=GRB.CONTINUOUS, name="completion")

        # Assignment, virtual nodes, bounds and within-stage appointment order.
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
                if h == 0:
                    # No inter-stage waiting exists at the first stage.
                    m.addConstr(self.beta[h, i] == rs_i, name=f"beta_stage1_{h}_{i}")
                    for k in range(K):
                        m.addConstr(self.theta[h, i, k] == 0.0, name=f"theta_stage1_{h}_{i}_{k}")
                else:
                    m.addConstr(
                        gp.quicksum(float(self.p[k]) * self.theta[h, i, k] for k in range(K)) <= rs_i,
                        name=f"theta_expectation_{h}_{i}",
                    )
            for i in range(Iv_h, I - 1):
                m.addConstr(self.y[h, i + 1] >= self.y[h, i], name=f"appointment_order_{h}_{i}")

        # Scenario arrival/start/completion recursion.  The max constraint makes
        # start equal to, rather than merely greater than, the two lower bounds.
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
                    if i == Iv_h:
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

        # Consecutive-stage patient mapping.  The manuscript experiments use
        # consecutive pathways or first entry at a later stage.  A route that
        # skips an intermediate attended stage requires a generalized predecessor
        # map and should not be silently treated as consecutive.
        for h in range(H):
            for i in range(I):
                for m_idx in range(I):
                    for j in range(J):
                        active = (
                            h >= 1
                            and i >= self.Iv[h]
                            and m_idx >= self.Iv[h - 1]
                            and int(self.c[h, j]) == 1
                            and int(self.c[h - 1, j]) == 1
                        )
                        if not active:
                            m.addConstr(self.zeta[h, i, m_idx, j] == 0, name=f"zeta_zero_{h}_{i}_{m_idx}_{j}")
                            continue
                        m.addConstr(self.zeta[h, i, m_idx, j] <= self.x[h, i, j], name=f"zeta1_{h}_{i}_{m_idx}_{j}")
                        m.addConstr(self.zeta[h, i, m_idx, j] <= self.x[h - 1, m_idx, j], name=f"zeta2_{h}_{i}_{m_idx}_{j}")
                        m.addConstr(
                            self.zeta[h, i, m_idx, j]
                            >= self.x[h, i, j] + self.x[h - 1, m_idx, j] - 1,
                            name=f"zeta3_{h}_{i}_{m_idx}_{j}",
                        )
                        if not defer_waiting_rows:
                            for k in range(K):
                                self.add_indirect_waiting_row(h, i, m_idx, j, k)

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

    def extract(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.model is None or self.model.SolCount <= 0:
            raise RuntimeError("No solution is available for extraction.")
        x = np.zeros((self.H, self.I, self.J), dtype=float)
        y = np.zeros((self.H, self.I), dtype=float)
        alpha = np.zeros((self.H, self.I), dtype=float)
        beta = np.zeros((self.H, self.I), dtype=float)
        for h in range(self.H):
            for i in range(self.I):
                y[h, i] = float(self.y[h, i].X)
                alpha[h, i] = float(self.alpha[h, i].X)
                beta[h, i] = float(self.beta[h, i].X)
                for j in range(self.J):
                    x[h, i, j] = float(self.x[h, i, j].X)
        return x, y, alpha, beta

    def solve(self) -> DeterministicEquivalentResult:
        if self.model is None:
            self.build(defer_waiting_rows=False)
        self.model.optimize()
        if self.model.SolCount <= 0:
            raise RuntimeError(f"DeterministicEquivalentBase returned no incumbent; status={self.model.Status}")
        x, y, alpha, beta = self.extract()
        return DeterministicEquivalentResult(
            obj=float(self.model.ObjVal),
            bound=float(self.model.ObjBound),
            x=x,
            y=y,
            alpha=alpha,
            beta=beta,
        )


DeterministicEquivalentBase = DeterministicEquivalentBase

__all__ = ["DeterministicEquivalentBase", "DeterministicEquivalentBase", "DeterministicEquivalentResult"]
