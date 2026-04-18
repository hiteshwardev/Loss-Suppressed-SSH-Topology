"""
nanophotonics.py
================
Nanophotonic realization of the SSH lattice via an effective-mode approach.

Each resonator supports a localized Gaussian mode. Coupling between adjacent
resonators arises from evanescent overlap. The normalized overlap integral:

    t_ij = integral phi_i(x) phi_j(x) dx
           / sqrt[integral phi_i^2 dx * integral phi_j^2 dx]

is the leading-order approximation to evanescent coupling in coupled-mode
theory. Only the RATIO t2/t1 is physically interpreted; absolute values
require full electromagnetic simulation.

Topological condition: t2 > t1 requires the inter-cell gap (d2) to be
SMALLER than the intra-cell gap (d1), since smaller separation → stronger
evanescent overlap. So d_intra > d_inter for topological phase.
"""

import numpy as np


def gaussian_mode(x: np.ndarray, center: float, width: float) -> np.ndarray:
    """Normalized Gaussian mode profile centred at `center` with half-width `width`."""
    return np.exp(-((x - center) ** 2) / (2 * width ** 2))


def normalized_overlap(mode_a: np.ndarray, mode_b: np.ndarray,
                       x: np.ndarray) -> float:
    """
    Normalized overlap integral: <phi_a | phi_b> / sqrt(<phi_a|phi_a><phi_b|phi_b>).

    This is the coupled-mode-theory coupling proxy. It correctly captures the
    exponential dependence on separation: for Gaussians separated by d and
    half-width w, the overlap scales as exp(-d^2 / (4w^2)).
    """
    overlap = np.trapezoid(mode_a * mode_b, x)
    norm_a  = np.sqrt(np.trapezoid(mode_a ** 2, x))
    norm_b  = np.sqrt(np.trapezoid(mode_b ** 2, x))
    return float(overlap / (norm_a * norm_b))


def build_resonator_chain(N: int, a: float, d_intra: float,
                          d_inter: float, width: float,
                          x_resolution: int = 5000) -> tuple:
    """
    Build a 1D resonator chain with SSH dimerization.

    Parameters
    ----------
    N           : int   Number of unit cells
    a           : float Lattice constant (sets length scale)
    d_intra     : float Intra-cell gap (A to B within a cell)
    d_inter     : float Inter-cell gap (B to A of next cell)
    width       : float Mode half-width
    x_resolution: int   Number of spatial grid points

    Returns
    -------
    centers : ndarray (2N,)   resonator positions
    modes   : list of ndarrays, each of length x_resolution
    x       : ndarray (x_resolution,)  spatial grid
    """
    # Build resonator positions
    centers = []
    pos = 0.0
    for _ in range(N):
        centers.append(pos)           # A site
        centers.append(pos + d_intra) # B site
        pos += d_intra + d_inter
    centers = np.array(centers)

    # Spatial grid covering the full chain with margin
    margin = 3.0 * width
    x = np.linspace(centers[0] - margin, centers[-1] + margin, x_resolution)
    modes = [gaussian_mode(x, c, width) for c in centers]
    return centers, modes, x


def extract_ssh_couplings(N: int, modes: list, x: np.ndarray) -> dict:
    """
    Extract t1 (intra-cell) and t2 (inter-cell) coupling strengths
    from mode overlap integrals.

    Returns
    -------
    dict with keys: t1_vals, t2_vals, t1_mean, t2_mean, ratio
    """
    t1_vals, t2_vals = [], []
    for n in range(N):
        t1_vals.append(normalized_overlap(modes[2 * n], modes[2 * n + 1], x))
        if n < N - 1:
            t2_vals.append(normalized_overlap(modes[2 * n + 1], modes[2 * n + 2], x))

    t1_mean = float(np.mean(t1_vals))
    t2_mean = float(np.mean(t2_vals))
    return {
        "t1_vals": np.array(t1_vals),
        "t2_vals": np.array(t2_vals),
        "t1_mean": t1_mean,
        "t2_mean": t2_mean,
        "ratio":   t2_mean / t1_mean,
    }
