"""Shared utilities for the scheduling models and experiments."""

from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np


def compute_first_follow(c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return first-stage and follow-up-stage indicators implied by pathway matrix c."""
    c = np.asarray(c, dtype=int)
    H, J = c.shape
    first = np.zeros((H, J), dtype=int)
    follow = np.zeros((H, J), dtype=int)
    for j in range(J):
        visited = [h for h in range(H) if int(c[h, j]) == 1]
        if not visited:
            continue
        first_h = visited[0]
        first[first_h, j] = 1
        for h in visited[1:]:
            follow[h, j] = 1
    return first, follow


def compute_data_driven_bigM(
    s: np.ndarray,
    tau: np.ndarray,
    psi: np.ndarray,
    c: np.ndarray,
    L: Sequence[float] | None,
    Iv: Sequence[int],
    fallback: float = 1e5,
) -> float:
    """
    Data-driven valid Big-M used in Appendix C:
        M_h = L_h + max arrival deviation at stage h + |physical nodes at h| * max service time at h.

    If L is not provided, return fallback to preserve backward compatibility.
    """
    if L is None:
        return float(fallback)

    s = np.asarray(s, dtype=float)
    tau = np.asarray(tau, dtype=float)
    psi = np.asarray(psi, dtype=float)
    c = np.asarray(c, dtype=int)
    L = np.asarray(L, dtype=float)
    Iv = list(map(int, Iv))

    if s.ndim != 4:
        raise ValueError(f"s must have shape (K,H,I,J); got {s.shape}")
    K, H, I, J = s.shape
    if c.shape != (H, J):
        raise ValueError(f"c must have shape (H,J)=({H},{J}); got {c.shape}")
    if tau.shape != (K, J):
        raise ValueError(f"tau must have shape (K,J)=({K},{J}); got {tau.shape}")
    if psi.shape != (K, J):
        raise ValueError(f"psi must have shape (K,J)=({K},{J}); got {psi.shape}")
    if len(Iv) != H:
        raise ValueError(f"Iv length must be H={H}; got {len(Iv)}")
    if L.shape[0] != H:
        raise ValueError(f"L must have length H={H}; got {L.shape}")

    first, follow = compute_first_follow(c)
    M_values = []
    for h in range(H):
        phys = list(range(Iv[h], I))
        sbar = float(np.max(s[:, h, phys, :])) if phys else 0.0
        xi = tau * first[h, :][None, :] + psi * follow[h, :][None, :]
        xibar = float(np.max(xi)) if xi.size else 0.0
        M_values.append(float(L[h]) + xibar + len(phys) * sbar)

    M = max(M_values) if M_values else float(fallback)
    # keep a tiny positive lower bound for numerical safety
    return float(max(M, 1e-6))
