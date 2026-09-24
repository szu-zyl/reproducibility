"""Stage-by-stage heuristic baselines for multistage appointment scheduling.

SS-MILP solves each stage independently with a single-stage MILP, while SS-BW
uses a Bailey--Welch-style spacing rule.  Returned schedules are evaluated with
the common fixed-schedule validator used by the experiment driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None


def _compute_first_follow(c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    first_stage[h,j] = 1 if stage h is first visited stage for patient j
    follow_stage[h,j] = 1 if stage h is visited and not first
    """
    H, J = c.shape
    first = np.zeros((H, J), dtype=int)
    follow = np.zeros((H, J), dtype=int)
    for j in range(J):
        visited = [h for h in range(H) if int(c[h, j]) == 1]
        if not visited:
            continue
        first_h = visited[0]
        for h in visited:
            if h == first_h:
                first[h, j] = 1
            else:
                follow[h, j] = 1
    return first, follow


@dataclass
class ScheduleResult:
    method: str
    x: np.ndarray          # (H,I,J)
    y: np.ndarray          # (H,I)


class StageByStageHeuristic:
    """
    Build stage-by-stage schedules with SS-MILP and SS-BW.

    Inputs follow the common experiment-data convention:
      s   : (K,H,I,J)  service time scenarios
      tau : (K,J)      lateness scenarios for first visited stage
      psi : (K,J)      lateness scenarios for follow-up stages
      c   : (H,J)      pathway (1 if patient j visits stage h)
      r_d : (J,)       direct waiting tolerance
      r_s : (J,)       indirect waiting tolerance
      Iv  : list(H)    number of virtual nodes per stage (physical i >= Iv[h])

    Both baselines ignore multistage coupling during schedule construction.  The
    experiment driver evaluates their schedules with the common validator.
    """

    def __init__(
        self,
        s: np.ndarray,
        tau: np.ndarray,
        psi: np.ndarray,
        c: np.ndarray,
        r_d: np.ndarray,
        r_s: np.ndarray,
        Iv: List[int],
        p: Optional[np.ndarray] = None,
        bigM: float = 1e5,
        threads: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        self.s = np.asarray(s, dtype=float)
        self.tau = np.asarray(tau, dtype=float)
        self.psi = np.asarray(psi, dtype=float)
        self.c = np.asarray(c, dtype=int)
        self.r_d = np.asarray(r_d, dtype=float)
        self.r_s = np.asarray(r_s, dtype=float)
        self.Iv = list(Iv)

        self.K, self.H, self.I, self.J = self.s.shape
        assert self.tau.shape == (self.K, self.J)
        assert self.psi.shape == (self.K, self.J)
        assert self.c.shape == (self.H, self.J)
        assert self.r_d.shape == (self.J,)
        assert self.r_s.shape == (self.J,)
        assert len(self.Iv) == self.H

        if p is None:
            self.p = np.ones(self.K, dtype=float) / float(self.K)
        else:
            self.p = np.asarray(p, dtype=float)
            self.p = self.p / np.sum(self.p)

        self.bigM = float(bigM)
        self.threads = int(threads)
        self.seed = seed

        self.first_stage, self.follow_stage = _compute_first_follow(self.c)

        # Precompute means for SS-BW spacing
        self.mean_s = np.mean(self.s, axis=0)  # (H,I,J)

    # ---------- tardiness helper for SS-MILP stage model ----------
    def _tard_expr_stage(self, h: int, i: int, k: int, x_stage) -> gp.LinExpr:
        # x_stage is m.addVars(I,J)
        return gp.quicksum(
            (float(self.tau[k, j]) * float(self.first_stage[h, j]) +
             float(self.psi[k, j]) * float(self.follow_stage[h, j])) * x_stage[i, j]
            for j in range(self.J)
        )

    # ---------- SS-BW (rule-based) ----------
    def build_ss_bw(self, kappa: float = 1.0, order_rule: str = "index") -> ScheduleResult:
        """
        Stage-by-stage rule baseline with permutation + virtual-first structure (consistent with full model).

        order_rule:
          - "index": order participating patients by index
          - "mean_spt": order participating patients by mean service time (ascending)

        Appointment times (real slots only):
          y[first_real] = 0, y[second_real] = 0 (two-at-start)
          y[next] = y[prev] + kappa * mean_service(prev_slot, assigned_patient)
        """
        H, I, J = self.H, self.I, self.J
        x = np.zeros((H, I, J), dtype=float)
        y = np.zeros((H, I), dtype=float)

        for h in range(H):
            Iv_h = int(self.Iv[h])
            virt_pos = list(range(0, Iv_h))
            real_pos = list(range(Iv_h, I))

            virt_pat = [j for j in range(J) if int(self.c[h, j]) == 0]
            real_pat = [j for j in range(J) if int(self.c[h, j]) == 1]

            # Sanity: must match permutation structure
            if len(virt_pat) != Iv_h or len(real_pat) != (I - Iv_h):
                raise ValueError(
                    f"Stage {h}: Iv={Iv_h}, #virt_pat={len(virt_pat)}, #real_pat={len(real_pat)}, I={I}"
                )

            # --- choose ordering for real patients ---
            if order_rule == "mean_spt":
                # proxy: use mean service at the first real position
                i0 = real_pos[0] if real_pos else 0
                real_pat = sorted(real_pat, key=lambda jj: float(self.mean_s[h, i0, jj]))
            else:
                real_pat = sorted(real_pat)

            # virtual patients: stable order (by index)
            virt_pat = sorted(virt_pat)

            # --- build permutation assignment x ---
            for idx, j in enumerate(virt_pat):
                x[h, virt_pos[idx], j] = 1.0
            for idx, j in enumerate(real_pat):
                x[h, real_pos[idx], j] = 1.0

            # --- build appointment times y ---
            for i in virt_pos:
                y[h, i] = 0.0

            if not real_pos:
                continue

            # two-at-start on first two real slots
            y[h, real_pos[0]] = 0.0
            if len(real_pos) >= 2:
                y[h, real_pos[1]] = 0.0

            # cumulative spacing afterwards
            for t in range(2, len(real_pos)):
                i_prev = real_pos[t - 1]
                i_cur = real_pos[t]
                j_prev = int(np.argmax(x[h, i_prev, :]))
                inc = float(self.mean_s[h, i_prev, j_prev])
                y[h, i_cur] = y[h, i_prev] + float(kappa) * inc

        return ScheduleResult(method=f"SS_BW_perm(kappa={kappa},order={order_rule})", x=x, y=y)

    # ---------- SS-MILP (per-stage MILP) ----------
    def build_ss_milp(
            self,
            time_limit_per_stage: float = 60.0,
            mip_gap: Optional[float] = None,
            output_flag: int = 0,
            L: Optional[np.ndarray] = None,  # NEW: clinic time boundary per stage
    ) -> ScheduleResult:
        """
        Solve each stage independently using a single-stage MILP that mirrors the direct-wait recourse structure.

        This stage MILP:
          - decides assignment x^h_{ij} and appointment times y^h_i
          - models direct-wait satisfaction alpha^h_i with scenario auxiliaries gamma^h_{i,k}
          - DOES NOT include cross-stage (rho/sigma/zeta) and DOES NOT include indirect satisfaction beta

        Objective per stage: minimize sum_j r_d[j] * x - sum_i alpha[i]
        (equivalently maximize direct satisfaction).
        """
        if gp is None:
            raise RuntimeError("gurobipy is required for SS-MILP.")

        H, I, J, K = self.H, self.I, self.J, self.K
        x_all = np.zeros((H, I, J), dtype=float)
        y_all = np.zeros((H, I), dtype=float)
        L_arr = None
        if L is not None:
            L_arr = np.asarray(L, dtype=float).reshape(-1)
            if L_arr.size != H:
                raise ValueError(f"L must have length H={H}, got {L_arr.size}")

        for h in range(H):
            Iv_h = self.Iv[h]
            phys = list(range(Iv_h, I))

            m = gp.Model(f"SS_MILP_stage_{h+1}")
            m.Params.OutputFlag = int(output_flag)
            m.Params.Threads = int(self.threads)
            if self.seed is not None:
                m.Params.Seed = int(self.seed)
            if time_limit_per_stage is not None:
                m.Params.TimeLimit = float(time_limit_per_stage)
            if mip_gap is not None:
                m.Params.MIPGap = float(mip_gap)

            xh = m.addVars(I, J, vtype=GRB.BINARY, name=f"x_{h}")
            yh = m.addVars(I, vtype=GRB.CONTINUOUS, lb=0.0, name=f"y_{h}")
            alpha = m.addVars(I, vtype=GRB.CONTINUOUS, lb=0.0, name=f"alpha_{h}")
            gamma = m.addVars(I, K, vtype=GRB.CONTINUOUS, lb=0.0, name=f"gamma_{h}")

            # permutation assignment (consistent with full model)
            for i in range(I):
                m.addConstr(gp.quicksum(xh[i, j] for j in range(J)) == 1, name=f"pos_one_{h}_{i}")
            for j in range(J):
                m.addConstr(gp.quicksum(xh[i, j] for i in range(I)) == 1, name=f"pat_one_{h}_{j}")

            # ---------- virtual positions: only non-participating patients; pin time to 0 ----------
            for i in range(Iv_h):
                # virtual nodes have no appointment activity in this stage
                m.addConstr(yh[i] == 0.0, name=f"y_virt_{h}_{i}")
                m.addConstr(alpha[i] == 0.0, name=f"alpha_virt_{h}_{i}")
                for k in range(K):
                    m.addConstr(gamma[i, k] == 0.0, name=f"virt_gamma_{h}_{i}_{k}")
                for j in range(J):
                    m.addConstr(xh[i, j] <= 1 - int(self.c[h, j]), name=f"virt_only_{h}_{i}_{j}")

            # ---------- real positions: only participating patients ----------
            for i in range(Iv_h, I):
                for j in range(J):
                    m.addConstr(xh[i, j] <= int(self.c[h, j]), name=f"real_only_{h}_{i}_{j}")

            # ---------- clinic time boundary ----------
            if L_arr is not None:
                L_h = float(L_arr[h])
                for i in range(I):
                    m.addConstr(yh[i] <= L_h, name=f"YUB_ss_{h}_{i}")

            # (optional but recommended) If your Iv design ensures no empty physical slots, enforce equality.
            n_visit = int(np.sum(self.c[h, :]))
            if n_visit == len(phys):
                for i in phys:
                    m.addConstr(gp.quicksum(xh[i, j] for j in range(J)) == 1, name=f"fill_{h}_{i}")

            # time ordering among physical slots
            for idx in range(len(phys) - 1):
                i = phys[idx]
                ip = phys[idx + 1]
                m.addConstr(yh[ip] >= yh[i], name=f"order_{h}_{i}")

            # alpha upper bounds
            for i in phys:
                m.addConstr(alpha[i] <= gp.quicksum(self.r_d[j] * xh[i, j] for j in range(J)),
                            name=f"alpha_ub_{h}_{i}")

            # first physical node: alpha equals rd*x (same as your full model structure)
            if phys:
                first_i = phys[0]
                m.addConstr(alpha[first_i] == gp.quicksum(self.r_d[j] * xh[first_i, j] for j in range(J)),
                            name=f"alpha_first_{h}")

            # direct waiting linearization for i > first physical
            for idx in range(1, len(phys)):
                i = phys[idx]

                # E[gamma] <= rd*x
                m.addConstr(
                    gp.quicksum(self.p[k] * gamma[i, k] for k in range(K))
                    <= gp.quicksum(self.r_d[j] * xh[i, j] for j in range(J)),
                    name=f"gamma_exp_{h}_{i}",
                )

                for k in range(K):
                    tard_i = self._tard_expr_stage(h, i, k, xh)
                    for t in phys[:idx]:  # all previous physical nodes
                        tard_t = self._tard_expr_stage(h, t, k, xh)
                        serv_sum = gp.quicksum(
                            self.s[k, h, a, j] * xh[a, j]
                            for a in range(t, i)
                            for j in range(J)
                        )
                        lhs = serv_sum + yh[t] + tard_t - yh[i] - tard_i + alpha[i]
                        m.addConstr(lhs <= gamma[i, k], name=f"c26_{h}_{i}_{t}_{k}")

            # objective for this stage: minimize rd*x - alpha
            base_h = gp.quicksum(self.r_d[j] * xh[i, j] for i in phys for j in range(J))
            sat_h = gp.quicksum(alpha[i] for i in phys)
            m.setObjective(base_h - sat_h, GRB.MINIMIZE)

            m.optimize()

            if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
                raise RuntimeError(f"SS-MILP stage {h+1} ended with status {m.Status}")

            # extract
            for i in range(I):
                y_all[h, i] = float(yh[i].X)
                for j in range(J):
                    x_all[h, i, j] = float(xh[i, j].X)

        return ScheduleResult(method="SS_MILP", x=x_all, y=y_all)

