"""
sweeps.py
=========
Parameter sweep and finite-size convergence calculations.
"""

import numpy as np
from .hamiltonian import build_ssh
from .ldos import enhancement_factor, integrated_ldos


def enhancement_map(N: int, r_vals: np.ndarray, gamma_vals: np.ndarray,
                    gap_omegas: np.ndarray, eta: float,
                    edge_site: int = 0) -> np.ndarray:
    """
    Compute the LDOS enhancement factor on a 2D parameter grid (r, gamma).

    r = t2/t1,   gamma = loss rate on B sublattice.
    t1 is set to 1.0 (dimensionless units).

    Parameters
    ----------
    N          : number of unit cells
    r_vals     : array of t2/t1 values  (x-axis)
    gamma_vals : array of gamma/t1 values (y-axis)
    gap_omegas : frequency grid over the gap window
    eta        : broadening
    edge_site  : edge site index (default 0)

    Returns
    -------
    enh : ndarray (len(gamma_vals), len(r_vals))
    """
    bulk_site = N  # centre A-sublattice site
    enh = np.zeros((len(gamma_vals), len(r_vals)))
    for i, g in enumerate(gamma_vals):
        for j, r in enumerate(r_vals):
            H = build_ssh(N, 1.0, r, g)
            enh[i, j] = enhancement_factor(H, edge_site, bulk_site,
                                           gap_omegas, eta)
    return enh


def finite_size_convergence(t1: float, t2: float, gamma: float,
                            N_vals: list, gap_omegas: np.ndarray,
                            eta: float) -> dict:
    """
    Compute LDOS enhancement vs. system size N.

    Returns dict with keys: N_vals, enhancement, gap_obc
    """
    from scipy import linalg as la
    enh_vals = []
    gap_obc  = []

    for N in N_vals:
        H = build_ssh(N, t1, t2, gamma)
        bulk_site = N
        E = enhancement_factor(H, 0, bulk_site, gap_omegas, eta)
        enh_vals.append(E)

        # OBC gap: smallest |Re(E)| outside the gap window
        evals_r = np.sort(la.eigvals(H).real)
        outside = evals_r[np.abs(evals_r) > np.max(np.abs(gap_omegas))]
        gap_obc.append(float(np.min(np.abs(outside))) if len(outside) else np.nan)

    return {
        "N_vals":      N_vals,
        "enhancement": enh_vals,
        "gap_obc":     gap_obc,
    }
