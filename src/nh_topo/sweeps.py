"""
sweeps.py
=========
Parameter sweeps and finite-size convergence.

The bulk reference site is ALWAYS the central A-sublattice (lossless) site
`central_bulk_site(N) = 2*(N//2)`. Taking the bulk index = N (as in v1) puts the
reference on a lossy B site for odd N, producing a spurious sublattice-parity
artifact in the finite-size scan (e.g. a non-physical spike at N = 15); that bug
is fixed here.
"""

import numpy as np

from .hamiltonian import build_ssh, build_ssh_general
from .ldos import (enhancement_factor, central_bulk_site,
                   edge_mode_spectral_weight)


def hoppings_for_ratio(r, mean_hopping=None):
    """
    Hoppings (t1, t2) realising the ratio r = t2/t1.

    With `mean_hopping` given, the pair lies on the line (t1 + t2)/2 = mean_hopping,
    so the energy unit, the loss rate, the broadening and the gap window keep the
    same meaning at every r and a sweep passes through the operating point
    (t1, t2) = (0.8, 1.2) at r = 1.5 when mean_hopping = 1. Without it, t1 = 1.
    """
    if mean_hopping is None:
        return 1.0, float(r)
    t1 = 2.0 * mean_hopping / (1.0 + r)
    return t1, r * t1


def enhancement_map(N, r_vals, gamma_vals, gap_omegas, eta,
                    edge_site: int = 0, mean_hopping=None) -> np.ndarray:
    """
    Integrated edge/bulk LDOS enhancement on a 2D (r = t2/t1, gamma) grid.
    Bulk reference = central A site (parity-safe).

    `mean_hopping` selects the parametrisation of each ratio r; see
    `hoppings_for_ratio`. The default (None) keeps t1 = 1.

    Returns enh : ndarray (len(gamma_vals), len(r_vals)).
    """
    bulk_site = central_bulk_site(N)
    enh = np.zeros((len(gamma_vals), len(r_vals)))
    for i, g in enumerate(gamma_vals):
        for j, r in enumerate(r_vals):
            t1, t2 = hoppings_for_ratio(r, mean_hopping)
            H = build_ssh(N, t1, t2, g)
            enh[i, j] = enhancement_factor(H, edge_site, bulk_site, gap_omegas, eta)
    return enh


def edge_weight_map(N, r_vals, gamma_vals, edge_site: int = 0) -> np.ndarray:
    """
    eta-INDEPENDENT protected-edge spectral weight W0 on the (r, gamma) grid.
    Complements `enhancement_map` with the broadening-free topological observable.

    Returns w0 : ndarray (len(gamma_vals), len(r_vals)).
    """
    w0 = np.zeros((len(gamma_vals), len(r_vals)))
    for i, g in enumerate(gamma_vals):
        for j, r in enumerate(r_vals):
            H = build_ssh(N, 1.0, r, g)
            w0[i, j] = edge_mode_spectral_weight(H, edge_site)
    return w0


def finite_size_convergence(t1, t2, gamma, N_vals, gap_omegas, eta) -> dict:
    """
    LDOS enhancement, protected-edge weight, and OBC band-edge gap vs system size.

    Bulk reference = central A site for every N (parity-safe).

    Returns dict: N_vals, enhancement, edge_weight, gap_obc.
    """
    from scipy import linalg as la

    enh_vals, w0_vals, gap_obc = [], [], []
    for N in N_vals:
        H = build_ssh(N, t1, t2, gamma)
        bulk_site = central_bulk_site(N)
        enh_vals.append(enhancement_factor(H, 0, bulk_site, gap_omegas, eta))
        w0_vals.append(edge_mode_spectral_weight(H, 0))

        evals_r = np.sort(la.eigvals(H).real)
        outside = evals_r[np.abs(evals_r) > np.max(np.abs(gap_omegas))]
        gap_obc.append(float(np.min(np.abs(outside))) if len(outside) else np.nan)

    return {
        "N_vals":      list(N_vals),
        "enhancement": enh_vals,
        "edge_weight": w0_vals,
        "gap_obc":     gap_obc,
    }
