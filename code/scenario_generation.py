"""Scenario generators used by the three computational experiments."""

from __future__ import annotations

import numpy as np


def _truncated_normal(rng, mean, sd, low=0.0, high=None, size=None, max_iter=2000):
    """Truncated normal sampler using rejection sampling (no SciPy dependency)."""
    if size is None:
        size = 1

    shape = (size,) if isinstance(size, int) else tuple(size)
    n = int(np.prod(shape))

    out = rng.normal(mean, sd, size=n)

    if high is None:
        mask = out < low
        it = 0
        while np.any(mask) and it < max_iter:
            out[mask] = rng.normal(mean, sd, size=int(np.sum(mask)))
            mask = out < low
            it += 1
    else:
        mask = (out < low) | (out > high)
        it = 0
        while np.any(mask) and it < max_iter:
            out[mask] = rng.normal(mean, sd, size=int(np.sum(mask)))
            mask = (out < low) | (out > high)
            it += 1

    if np.any(mask):
        # Fallback safeguard (should be rare).
        out = np.clip(out, low, high if high is not None else np.inf)

    return out.reshape(shape)


def generate_service_times(numSamples, numStages, numPositions, numPatients, seed=39):
    """Generate service-time scenarios S[k][h][i][j]."""
    rng = np.random.default_rng(seed)
    S = np.zeros((numSamples, numStages, numPositions, numPatients), dtype=float)

    # Stage-dependent standard deviation (consistent with the original code).
    for h in range(numStages):
        sigma_h = h / 3.0
        if sigma_h <= 0:
            S[:, h, :, :] = 2.0
        else:
            S[:, h, :, :] = _truncated_normal(
                rng,
                mean=2.0,
                sd=sigma_h,
                low=0.05,
                high=None,
                size=(numSamples, numPositions, numPatients),
            )

    return S


def generate_lateness_times(numSamples, numPatients, c, seed=39):
    """Generate lateness scenarios (tau: first-stage, psi: follow-up)."""
    rng = np.random.default_rng(seed + 101)
    tau = np.zeros((numSamples, numPatients), dtype=float)
    psi = np.zeros((numSamples, numPatients), dtype=float)

    c = np.asarray(c)
    for j in range(numPatients):
        # If patient j never participates, keep zeros.
        if c.ndim == 2 and np.sum(c[:, j]) == 0:
            continue

        # first-stage lateness
        tau[:, j] = _truncated_normal(
            rng, mean=0.2, sd=0.1, low=0.0, high=0.5, size=numSamples
        )
        # follow-up lateness
        psi[:, j] = _truncated_normal(
            rng, mean=0.1, sd=0.05, low=0.0, high=0.5, size=numSamples
        )

    return tau, psi


def generate_all_scenarios(numSamples, numStages, numPositions, numPatients, c, seed=39):
    """Generate all scenario samples used in experiments."""
    S = generate_service_times(numSamples, numStages, numPositions, numPatients, seed=seed)
    tau, psi = generate_lateness_times(numSamples, numPatients, c, seed=seed)
    return S, tau, psi


def generate_all_scenarios_independent(
    numSamples,
    numStages,
    numPositions,
    numPatients,
    c,
    seed_service=2025,
    seed_lateness=2026,
):
    """Generate scenarios with separately controlled service and lateness seeds."""
    service = generate_service_times(
        numSamples,
        numStages,
        numPositions,
        numPatients,
        seed=int(seed_service),
    )
    tau, psi = generate_lateness_times(
        numSamples,
        numPatients,
        c,
        seed=int(seed_lateness),
    )
    return service, tau, psi


__all__ = [
    "generate_service_times",
    "generate_lateness_times",
    "generate_all_scenarios",
    "generate_all_scenarios_independent",
]
