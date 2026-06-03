"""
nanophotonics.py
================
Coupled-mode-theory (CMT) schematic linking the abstract SSH couplings (t1, t2)
to a dimerised chain of evanescently-coupled single-mode resonators.

SCOPE / HONESTY STATEMENT
-------------------------
This module is a PARAMETER MAPPING, not an electromagnetic validation. It does
NOT solve Maxwell's equations. Its only purpose is to show that the tight-binding
operating point (t1 = 0.8, t2 = 1.2, ratio 1.5) is realisable by a physically
sensible geometry, and to do so SELF-CONSISTENTLY (the extracted couplings
reproduce the model's couplings exactly). Quantitative nanophotonic predictions
(absolute Purcell factors, radiative losses, band frequencies) require a
full-wave solver and are explicitly out of scope.

Physics used: the coupling between two evanescently-coupled modes decays
EXPONENTIALLY with the edge-to-edge gap d,

    t(d) = t0 * exp(-d / L),

where L is the evanescent field-decay length set by the index contrast and
frequency (standard CMT). (The v1 Gaussian-overlap proxy gave exp(-d^2/...),
which is the wrong functional form for evanescent coupling and is removed.)
"""

import numpy as np


def evanescent_coupling(d, t0, L):
    """Evanescent CMT coupling t(d) = t0 * exp(-d / L) for edge-to-edge gap d."""
    return t0 * np.exp(-np.asarray(d, float) / L)


def calibrate_cmt(t1, t2, d_intra, d_inter):
    """
    Solve for the single decay length L and prefactor t0 such that

        t1 = t0 exp(-d_intra / L),   t2 = t0 exp(-d_inter / L).

    A single (L, t0) reproduces BOTH target couplings exactly, making the
    nanophotonic mapping self-consistent with the tight-binding model.

    Requires t2 > t1 and d_inter < d_intra (topological geometry).

    Returns
    -------
    dict: L, t0, ratio_geom (= exp((d_intra-d_inter)/L)), and the reproduced t1,t2.
    """
    if not (d_intra > d_inter):
        raise ValueError("topological geometry requires d_inter < d_intra")
    if not (t2 > 0 and t1 > 0):
        raise ValueError("couplings must be positive")
    L = (d_intra - d_inter) / np.log(t2 / t1)
    t0 = t1 / np.exp(-d_intra / L)
    return {
        "L": float(L), "t0": float(t0),
        "ratio_geom": float(np.exp((d_intra - d_inter) / L)),
        "t1_reproduced": float(evanescent_coupling(d_intra, t0, L)),
        "t2_reproduced": float(evanescent_coupling(d_inter, t0, L)),
    }


def mode_profile(x, center, L):
    """
    Illustrative bound-mode amplitude with the physically-correct EXPONENTIAL
    evanescent tail, ~ exp(-|x - center| / L). (Real guided/bound modes decay
    exponentially outside the core; Gaussians do not.)
    """
    return np.exp(-np.abs(np.asarray(x, float) - center) / L)


def build_resonator_chain(N, a, d_intra, d_inter, L, x_resolution: int = 4000):
    """
    Dimerised resonator chain with SSH spacing. `d_intra`, `d_inter`, `a`, `L` are
    lengths in the same units. Returns (centers (2N,), modes list, x grid).
    """
    centers, pos = [], 0.0
    for _ in range(N):
        centers.append(pos)
        centers.append(pos + d_intra)
        pos += d_intra + d_inter
    centers = np.array(centers)

    margin = 6.0 * L
    x = np.linspace(centers[0] - margin, centers[-1] + margin, x_resolution)
    modes = [mode_profile(x, c, L) for c in centers]
    return centers, modes, x


def extract_ssh_couplings(N, t0, L, d_intra, d_inter) -> dict:
    """
    SSH couplings from the calibrated evanescent law (exact, self-consistent):
    every t1 bond is t0 exp(-d_intra/L), every t2 bond is t0 exp(-d_inter/L).

    Returns dict: t1_vals, t2_vals, t1_mean, t2_mean, ratio.
    """
    t1_vals = np.full(N, evanescent_coupling(d_intra, t0, L))
    t2_vals = np.full(max(N - 1, 0), evanescent_coupling(d_inter, t0, L))
    return {
        "t1_vals": t1_vals, "t2_vals": t2_vals,
        "t1_mean": float(t1_vals.mean()), "t2_mean": float(t2_vals.mean()),
        "ratio": float(t2_vals.mean() / t1_vals.mean()),
    }
