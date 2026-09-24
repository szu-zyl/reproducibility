"""The direct benchmark uses the compact deterministic equivalent.  The
formulation-independent strengthening is a harmless
symmetry break for dummy (non-attending) positions.  The same symmetry break
and the same complete feasible MIP start are also used by the decomposition
models.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

try:
    import gurobipy as gp
except Exception:  # pragma: no cover
    gp = None

from de_milp_compact import CompactDeterministicEquivalent


def natural_assignment(c: np.ndarray, Iv: list[int]) -> Tuple[np.ndarray, Dict[Tuple[int, int], int]]:
    """Return a deterministic feasible permutation at every stage."""
    c = np.asarray(c, dtype=int)
    H, J = c.shape
    I = J
    assignment = np.full((H, I), -1, dtype=int)
    position: Dict[Tuple[int, int], int] = {}
    for h in range(H):
        inactive = [j for j in range(J) if int(c[h, j]) == 0]
        active = [j for j in range(J) if int(c[h, j]) == 1]
        if len(inactive) != int(Iv[h]) or len(inactive) + len(active) != I:
            raise ValueError("The permutation formulation requires I=J and Iv[h] equal to the number of inactive patients.")
        order = inactive + active
        assignment[h, :] = order
        for i, j in enumerate(order):
            position[h, j] = i
    return assignment, position


def apply_virtual_symmetry_breaking(solver) -> int:
    """Fix the arbitrary permutation of patients over dummy positions.

    Dummy assignments do not affect the schedule or objective.  Fixing them is
    therefore exact and removes a large source of branch-and-bound symmetry.
    """
    assignment, _ = natural_assignment(solver.c, solver.Iv)
    fixed = 0
    for h in range(solver.H):
        for i in range(solver.Iv[h]):
            j = int(assignment[h, i])
            solver.x[h, i, j].LB = 1.0
            solver.x[h, i, j].UB = 1.0
            fixed += 1
    return fixed


def apply_natural_mip_start(solver) -> None:
    """Apply the common assignment and appointment-time MIP start."""
    assignment, _ = natural_assignment(solver.c, solver.Iv)
    for h in range(solver.H):
        active_count = solver.I - solver.Iv[h]
        for i in range(solver.I):
            selected = int(assignment[h, i])
            for j in range(solver.J):
                solver.x[h, i, j].Start = 1.0 if j == selected else 0.0
            if i < solver.Iv[h]:
                solver.y[h, i].Start = 0.0
            elif active_count <= 1 or solver.L is None:
                solver.y[h, i].Start = 0.0
            else:
                rank = i - solver.Iv[h]
                solver.y[h, i].Start = float(solver.L[h]) * rank / (active_count - 1)


def apply_complete_schedule_mip_start(
    solver,
    assignment: np.ndarray,
    y_value: np.ndarray,
    tol: float = 1e-8,
) -> bool:
    """Complete a fixed assignment/schedule with exact scenario recourse.

    Values are written only after the full-scenario feasibility calculation
    succeeds, so an infeasible candidate cannot overwrite an earlier start.
    """
    assignment = np.asarray(assignment, dtype=int)
    y_value = np.asarray(y_value, dtype=float)
    if assignment.shape != (solver.H, solver.I):
        raise ValueError("assignment must have shape (H,I)")
    if y_value.shape != (solver.H, solver.I):
        raise ValueError("y_value must have shape (H,I)")
    position: Dict[Tuple[int, int], int] = {}
    for h in range(solver.H):
        if sorted(int(j) for j in assignment[h, :]) != list(range(solver.J)):
            raise ValueError("Each assignment row must be a patient permutation.")
        for i in range(solver.I):
            j = int(assignment[h, i])
            if int(solver.c[h, j]) != int(i >= solver.Iv[h]):
                return False
            position[h, j] = i
    previous_stage = getattr(solver, "previous_stage", None)
    if previous_stage is None:
        previous_stage = solver._make_previous_stage()

    arrival = np.zeros((solver.H, solver.I, solver.K), dtype=float)
    start = np.zeros_like(arrival)
    completion = np.zeros_like(arrival)
    predecessor_completion = np.zeros_like(arrival)
    ready = np.zeros_like(arrival)
    alpha = np.zeros((solver.H, solver.I), dtype=float)
    beta = np.zeros((solver.H, solver.I), dtype=float)
    gamma = np.zeros_like(arrival)
    theta = np.zeros_like(arrival)

    for h in range(solver.H):
        for i in range(solver.Iv[h], solver.I):
            j = int(assignment[h, i])
            previous = int(previous_stage[h, j])
            for k in range(solver.K):
                tardiness = (
                    float(solver.tau[k, j]) * float(solver.first_stage[h, j])
                    + float(solver.psi[k, j]) * float(solver.follow_stage[h, j])
                )
                arrival[h, i, k] = y_value[h, i] + tardiness
                if previous >= 0:
                    previous_i = int(position[previous, j])
                    predecessor_completion[h, i, k] = completion[previous, previous_i, k]
                ready[h, i, k] = max(
                    arrival[h, i, k], predecessor_completion[h, i, k]
                )
                start_inputs = [
                    ready[h, i, k],
                ]
                if i > solver.Iv[h]:
                    start_inputs.append(completion[h, i - 1, k])
                start[h, i, k] = max(start_inputs)
                completion[h, i, k] = (
                    start[h, i, k] + float(solver.s[k, h, i, j])
                )

    for h in range(solver.H):
        for i in range(solver.Iv[h], solver.I):
            j = int(assignment[h, i])
            direct = start[h, i, :] - ready[h, i, :]
            alpha[h, i] = min(
                float(solver.r_d[j]),
                max(
                    0.0,
                    float(solver.r_d[j]) - float(np.dot(solver.p, direct)),
                ),
            )
            gamma[h, i, :] = direct + alpha[h, i]

            previous = int(previous_stage[h, j])
            if previous < 0:
                beta[h, i] = float(solver.r_s[j])
                continue
            previous_i = int(position[previous, j])
            raw = arrival[h, i, :] - completion[previous, previous_i, :]
            tolerance = float(solver.r_s[j])

            def expected_required(beta_value: float) -> float:
                return float(
                    np.dot(solver.p, np.maximum(0.0, raw + beta_value))
                )

            if expected_required(0.0) > tolerance + tol:
                return False
            if expected_required(tolerance) <= tolerance + tol:
                beta[h, i] = tolerance
            else:
                lower, upper = 0.0, tolerance
                for _ in range(50):
                    middle = 0.5 * (lower + upper)
                    if expected_required(middle) <= tolerance:
                        lower = middle
                    else:
                        upper = middle
                beta[h, i] = lower
            theta[h, i, :] = np.maximum(0.0, raw + beta[h, i])

    for h in range(solver.H):
        for i in range(solver.I):
            selected = int(assignment[h, i])
            for j in range(solver.J):
                solver.x[h, i, j].Start = 1.0 if j == selected else 0.0
            solver.y[h, i].Start = float(y_value[h, i])
            solver.alpha[h, i].Start = float(alpha[h, i])
            solver.beta[h, i].Start = float(beta[h, i])
            for k in range(solver.K):
                solver.gamma[h, i, k].Start = float(gamma[h, i, k])
                solver.theta[h, i, k].Start = float(theta[h, i, k])
                solver.arrival[h, i, k].Start = float(arrival[h, i, k])
                solver.start[h, i, k].Start = float(start[h, i, k])
                solver.completion[h, i, k].Start = float(completion[h, i, k])
                selected_predecessor = getattr(solver, "predecessor_completion", None)
                if selected_predecessor is not None:
                    selected_predecessor[h, i, k].Start = float(
                        predecessor_completion[h, i, k]
                    )
                selected_ready = getattr(solver, "ready", None)
                if selected_ready is not None:
                    selected_ready[h, i, k].Start = float(ready[h, i, k])

    patient_completion = getattr(solver, "patient_completion", None)
    if patient_completion is not None:
        for h in range(solver.H):
            for j in range(solver.J):
                for k in range(solver.K):
                    patient_completion[h, j, k].Start = (
                        float(completion[h, int(position[h, j]), k])
                        if int(solver.c[h, j]) == 1
                        else 0.0
                    )
    return True


def apply_complete_natural_mip_start(solver, tol: float = 1e-8) -> bool:
    """Complete the common natural start with exact scenario recourse.

    A partial ``x/y`` start becomes increasingly expensive for Gurobi to
    complete when ``I=J`` grows.  Here all timing and recourse values are
    evaluated analytically for that same deterministic schedule.  If the
    schedule cannot meet an indirect-wait tolerance even with ``beta=0``, the
    original partial start remains in place.  The calculation is formulation
    independent and is therefore applied to DE, LBBD, and CBBD alike.
    """
    apply_natural_mip_start(solver)
    assignment, _ = natural_assignment(solver.c, solver.Iv)
    # Use an evenly spaced appointment grid over each stage window.  This is a
    # deterministic feasibility rule: it performs no permutation search and
    # K-BRKGA remains a completely independent comparison method.
    y_value = np.zeros((solver.H, solver.I), dtype=float)
    for h in range(solver.H):
        active_count = solver.I - solver.Iv[h]
        for i in range(solver.Iv[h], solver.I):
            if active_count > 1 and solver.L is not None:
                rank = i - solver.Iv[h]
                y_value[h, i] = (
                    float(solver.L[h]) * rank / (active_count - 1)
                )
    return apply_complete_schedule_mip_start(
        solver, assignment, y_value, tol=tol
    )


def configure_model(
    solver,
    *,
    symmetry_breaking: bool = True,
    warm_start: bool = True,
    complete_warm_start: bool = False,
    mip_focus: int = 1,
    no_rel_heur_time: float = 0.0,
) -> None:
    """Apply settings shared by the direct model and both decompositions."""
    if solver.model is None:
        raise RuntimeError("Build the model before applying the solver configuration.")
    if symmetry_breaking:
        apply_virtual_symmetry_breaking(solver)
    if warm_start:
        if complete_warm_start:
            solver.complete_warm_start = apply_complete_natural_mip_start(solver)
        else:
            apply_natural_mip_start(solver)
            solver.complete_warm_start = False
    else:
        solver.complete_warm_start = False
    solver.model.Params.Presolve = 2
    solver.model.Params.Symmetry = 2
    solver.model.Params.MIPFocus = int(mip_focus)
    solver.model.Params.NoRelHeurTime = max(0.0, float(no_rel_heur_time))
    # Pairwise lazy rows contain assignment-dependent big-M terms.  Tight
    # integrality/feasibility tolerances prevent an almost-integral incumbent
    # from receiving a material artificial recourse credit.
    solver.model.Params.IntFeasTol = 1e-9
    solver.model.Params.FeasibilityTol = 1e-8


def add_expected_pair_strengthening(solver, prefix: str = "expected_pair") -> int:
    """Add scenario-aggregated valid inequalities for every feasible stage pair.

    These rows are implied by the scenario-wise cross-stage waiting constraints
    and therefore do not change the feasible set.  Unlike the full rows, their
    count does not grow with K.  They give both the complete DE and the lazy
    model a comparable, stronger root relaxation.
    """
    if solver.model is None:
        raise RuntimeError("Build the variables before adding strengthening rows.")
    previous_stage = getattr(solver, "previous_stage", None)
    if previous_stage is None:
        previous_stage = solver._make_previous_stage()
    added = 0
    for h in range(solver.H):
        for j in range(solver.J):
            previous = int(previous_stage[h, j])
            if int(solver.c[h, j]) != 1 or previous < 0:
                continue
            for i in range(solver.Iv[h], solver.I):
                for m_idx in range(solver.Iv[previous], solver.I):
                    lhs = gp.quicksum(
                        float(solver.p[k])
                        * (
                            solver.arrival[h, i, k]
                            - solver.completion[previous, m_idx, k]
                            - solver.theta[h, i, k]
                        )
                        for k in range(solver.K)
                    ) + solver.beta[h, i]
                    solver.model.addConstr(
                        lhs
                        <= solver.bigM
                        * (
                            2
                            - solver.x[h, i, j]
                            - solver.x[previous, m_idx, j]
                        ),
                        name=f"{prefix}_{h}_{i}_{m_idx}_{j}",
                    )
                    added += 1
    return added


def add_expected_position_strengthening(
    solver, prefix: str = "expected_position"
) -> int:
    """Add O(HI) expected-value consequences of inter-stage logic rows.

    The row is obtained by multiplying every exact scenario row by its original
    probability and summing.  It is globally valid for the unreduced SAA and
    uses the selected predecessor-completion variable, so no assignment-pair
    Cartesian product is introduced.
    """
    if solver.model is None:
        raise RuntimeError("Build the variables before adding strengthening rows.")
    first_entry_expr = getattr(solver, "first_entry_expr", None)
    added = 0
    for h in range(1, solver.H):
        for i in range(solver.Iv[h], solver.I):
            if first_entry_expr is not None:
                first_entry = first_entry_expr[h, i]
            else:
                first_entry = gp.quicksum(
                    solver.x[h, i, j]
                    for j in range(solver.J)
                    if int(solver.previous_stage[h, j]) < 0
                )
            solver.model.addConstr(
                gp.quicksum(
                    float(solver.p[k])
                    * (
                        solver.arrival[h, i, k]
                        - solver.predecessor_completion[h, i, k]
                        - solver.theta[h, i, k]
                    )
                    for k in range(solver.K)
                )
                + solver.beta[h, i]
                <= solver.bigM * first_entry,
                name=f"{prefix}_{h}_{i}",
            )
            added += 1
    return added


class DeterministicEquivalentModel(CompactDeterministicEquivalent):
    """Direct deterministic-equivalent benchmark used in the experiments."""

    def add_direct_waiting_row(self, h: int, i: int, k: int, name=None):
        """Link direct waiting to service start minus cross-stage readiness."""
        if self.model is None:
            raise RuntimeError("Build the model before adding direct-waiting rows.")
        return self.model.addConstr(
            self.start[h, i, k] - self.ready[h, i, k] + self.alpha[h, i]
            <= self.gamma[h, i, k],
            name=name or f"direct_wait_{h}_{i}_{k}",
        )

    def build(
        self,
        defer_waiting_rows: bool = False,
        *,
        symmetry_breaking: bool = True,
        warm_start: bool = True,
        complete_warm_start: bool = False,
        aggregate_strengthening: bool = False,
        no_rel_heur_time: float = 0.0,
    ):
        self.enforce_cross_stage_precedence = True
        model = super().build(defer_waiting_rows=defer_waiting_rows)
        model.ModelName = "DE_MILP" if not defer_waiting_rows else "restricted_DE_master"
        self.aggregate_cuts = (
            add_expected_pair_strengthening(self, prefix="DE_expected_pair")
            if aggregate_strengthening
            else 0
        )
        configure_model(
            self,
            symmetry_breaking=symmetry_breaking,
            warm_start=warm_start,
            complete_warm_start=complete_warm_start,
            no_rel_heur_time=no_rel_heur_time,
        )
        return model


DeterministicEquivalentModel = DeterministicEquivalentModel

__all__ = [
    "DeterministicEquivalentModel",
    "DeterministicEquivalentModel",
    "natural_assignment",
    "apply_virtual_symmetry_breaking",
    "apply_natural_mip_start",
    "apply_complete_schedule_mip_start",
    "apply_complete_natural_mip_start",
    "configure_model",
    "add_expected_pair_strengthening",
    "add_expected_position_strengthening",
]
