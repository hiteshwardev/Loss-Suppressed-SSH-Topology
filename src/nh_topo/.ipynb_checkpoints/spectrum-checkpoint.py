"""
spectrum.py
===========
Eigenvalue analysis, edge-mode identification, and localization diagnostics
for the non-Hermitian SSH Hamiltonian.

For H = H^T (complex-symmetric), the biorthogonal inner product is:
    <psi_L^m | psi_R^n> = sum_i vR[i,m] * vR[i,n]   (NOT conjugate)

This follows from H^T = H  =>  left eigenvectors = (right eigenvectors)^T.
The biorthogonal density at site i for mode n is:
    rho_i^(n) = Re(vR[i,n]^2) / Re(norm_n)
"""

import numpy as np
import scipy.linalg as la


def diagonalize(H: np.ndarray) -> tuple:
    """
    Diagonalize H, sort by Re(E).

    Returns
    -------
    evals  : ndarray (dim,) complex, sorted by real part
    evecs  : ndarray (dim, dim) complex, right eigenvectors as columns
    """
    evals_raw, evecs_raw = la.eig(H)
    idx = np.argsort(evals_raw.real)
    return evals_raw[idx], evecs_raw[:, idx]


def biorthogonal_density(evecs: np.ndarray) -> np.ndarray:
    """
    Compute the biorthogonal probability density for each eigenstate.

    For H = H^T: rho[i, n] = Re(vR[i,n]^2) / Re(sum_j vR[j,n]^2)

    Returns
    -------
    density : ndarray (dim, dim), real  — density[site, eigenstate]
    """
    bio_norms = np.einsum("ij,ij->j", evecs, evecs)   # complex biorthogonal norms
    density = np.real(evecs ** 2) / np.real(bio_norms)[np.newaxis, :]
    return density


def edge_localization_weights(evecs: np.ndarray, edge_cells: int = 1) -> tuple:
    """
    Compute left and right edge localization weights using right eigenvectors
    (valid for visualization; for strict observables use biorthogonal_density).

    Parameters
    ----------
    evecs      : (dim, n_states) eigenvector matrix
    edge_cells : number of unit cells counted as 'edge' (default 1 = 2 sites)

    Returns
    -------
    left_w, right_w : ndarray (n_states,) — weight near left/right boundary
    """
    dim = evecs.shape[0]
    n_edge_sites = 2 * edge_cells
    left_w  = np.sum(np.abs(evecs[:n_edge_sites,  :]) ** 2, axis=0)
    right_w = np.sum(np.abs(evecs[-n_edge_sites:, :]) ** 2, axis=0)
    return left_w, right_w


def find_edge_modes(evals: np.ndarray, evecs: np.ndarray,
                    gap_threshold: float = 0.15,
                    loss_threshold: float = 0.01) -> dict:
    """
    Identify topological edge modes in the spectrum.

    Criteria:
    - 'protected': |Re(E)| < gap_threshold AND |Im(E)| < loss_threshold
                   (A-sublattice mode, near-zero linewidth)
    - 'lossy':     |Re(E)| < gap_threshold AND |Im(E)| >= loss_threshold
                   (B-sublattice mode, linewidth ~ gamma)

    Returns
    -------
    dict with keys 'protected_idx', 'lossy_idx', 'protected_evals', 'lossy_evals'
    """
    protected = np.where(
        (np.abs(evals.real) < gap_threshold) &
        (np.abs(evals.imag) < loss_threshold)
    )[0]
    lossy = np.where(
        (np.abs(evals.real) < gap_threshold) &
        (np.abs(evals.imag) >= loss_threshold)
    )[0]
    return {
        "protected_idx":   protected,
        "lossy_idx":       lossy,
        "protected_evals": evals[protected],
        "lossy_evals":     evals[lossy],
    }


def fit_localization_length(evals: np.ndarray, evecs: np.ndarray,
                            N: int) -> float:
    """
    Fit the exponential decay of the protected edge mode on A sublattice.

    Returns the fitted localization length in unit cells.
    """
    edge_info = find_edge_modes(evals, evecs)
    if len(edge_info["protected_idx"]) == 0:
        return np.nan

    idx = edge_info["protected_idx"][0]
    psi = evecs[:, idx]
    intensity_A = np.abs(psi[0::2]) ** 2   # A-sublattice sites only
    intensity_A = intensity_A / intensity_A[0]  # normalize to site 0

    n_cells = np.arange(N)
    valid = intensity_A > 1e-12
    if valid.sum() < 3:
        return np.nan

    log_int = np.log(intensity_A[valid] + 1e-30)
    coeffs = np.polyfit(n_cells[valid], log_int, 1)
    return float(-2.0 / coeffs[0])
