"""Public LBBD and CBBD solver configurations used in Section 5.3."""

from __future__ import annotations

from benders_core import BendersResult, BendersSolver


class LBBDSolver(BendersSolver):
    """Logic-based Benders decomposition with full-scenario separation."""

    def __init__(self, *args, **kwargs):
        kwargs["strategy"] = "lbbd"
        kwargs.setdefault("cuts_per_callback", 2048)
        kwargs.setdefault("max_cuts_per_callback", 8192)
        kwargs.setdefault("batch_growth", 2.0)
        super().__init__(*args, **kwargs)


class CBBDSolver(BendersSolver):
    """Cluster-guided Benders decomposition with complete scenario checks."""

    def __init__(self, *args, **kwargs):
        kwargs["strategy"] = "cbbd"
        kwargs.setdefault("cuts_per_callback", 2048)
        kwargs.setdefault("max_cuts_per_callback", 8192)
        kwargs.setdefault("batch_growth", 2.0)
        super().__init__(*args, **kwargs)


__all__ = ["LBBDSolver", "CBBDSolver", "BendersResult"]
