"""
disorder.py
===========
Disorder models and statistical analysis for LDOS robustness studies.

Disorder type: uniform on-site frequency fluctuations U[-W/2, W/2].
This models the dominant fabrication imperfection in lithographically
defined photonic structures (resonance frequency variation).

Gap context (for W interpretation):
    Topological gap half-width = |t2 - t1| / 2
    W < gap/2  → sub-gap (weak): topology robust
    W ~ gap    → near-gap: edge mode begins to degrade
    W > gap    → above-gap (strong): protection lost
"""

import numpy as np
from typing import Callable


def add_onsite_disorder(H: np.ndarray, W: float,
                        rng: np.random.Generator) -> np.ndarray:
    """
    Add uniform on-site disorder U[-W/2, W/2] to all diagonal elements.

    Parameters
    ----------
    H   : base Hamiltonian
    W   : disorder amplitude (full width)
    rng : numpy random Generator (use np.random.default_rng(seed))

    Returns
    -------
    H_disordered : copy of H with random diagonal shifts
    """
    dim = H.shape[0]
    shifts = W * (rng.random(dim) - 0.5)          # U[-W/2, W/2]
    return H + np.diag(shifts.astype(np.complex128))


def disorder_average_ldos(H_base: np.ndarray, W: float,
                          edge_site: int, bulk_site: int,
                          omega_range: np.ndarray, eta: float,
                          gap_omegas: np.ndarray,
                          num_realizations: int,
                          rng: np.random.Generator) -> dict:
    """
    Compute disorder-averaged LDOS and enhancement statistics.

    Parameters
    ----------
    H_base           : base (clean) Hamiltonian
    W                : disorder strength
    edge_site        : site index for edge LDOS
    bulk_site        : site index for bulk LDOS
    omega_range      : full frequency grid for averaged spectra
    eta              : broadening
    gap_omegas       : narrow frequency grid over gap (for enhancement metric)
    num_realizations : ensemble size
    rng              : random generator

    Returns
    -------
    dict with keys:
        edge_avg     : disorder-averaged edge LDOS spectrum
        bulk_avg     : disorder-averaged bulk LDOS spectrum
        enh_all      : enhancement values for each realization
        enh_mean     : mean enhancement
        enh_std      : std of enhancement
        enh_median   : median enhancement
    """
    from .ldos import ldos_spectrum, integrated_ldos

    n_omega = len(omega_range)
    edge_accum = np.zeros(n_omega)
    bulk_accum = np.zeros(n_omega)
    enh_list = []

    for _ in range(num_realizations):
        Hd = add_onsite_disorder(H_base, W, rng)

        re_spec = ldos_spectrum(Hd, edge_site, omega_range, eta)
        rb_spec = ldos_spectrum(Hd, bulk_site, omega_range, eta)
        edge_accum += re_spec
        bulk_accum += rb_spec

        re_gap = integrated_ldos(Hd, edge_site, gap_omegas, eta)
        rb_gap = integrated_ldos(Hd, bulk_site, gap_omegas, eta)
        enh_list.append(re_gap / max(rb_gap, 1e-12))

    enh_arr = np.array(enh_list)
    return {
        "edge_avg":   edge_accum / num_realizations,
        "bulk_avg":   bulk_accum / num_realizations,
        "enh_all":    enh_arr,
        "enh_mean":   float(enh_arr.mean()),
        "enh_std":    float(enh_arr.std()),
        "enh_median": float(np.median(enh_arr)),
    }


def run_disorder_sweep(H_base: np.ndarray, W_vals: list,
                       edge_site: int, bulk_site: int,
                       omega_range: np.ndarray, eta: float,
                       gap_omegas: np.ndarray,
                       num_realizations: int, seed: int = 42) -> dict:
    """
    Run disorder averaging across multiple disorder strengths.

    Returns dict keyed by W with disorder_average_ldos results.
    """
    rng = np.random.default_rng(seed)
    results = {}
    for W in W_vals:
        results[W] = disorder_average_ldos(
            H_base, W, edge_site, bulk_site,
            omega_range, eta, gap_omegas,
            num_realizations, rng
        )
    return results
