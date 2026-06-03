"""
spectrum.py
===========
Eigenvalue analysis, edge-mode identification, and localization diagnostics
for the non-Hermitian SSH Hamiltonian.

For H = H^T (complex-symmetric) the biorthogonal inner product is the
UNCONJUGATED bilinear form, because the left eigenvectors are the transpose of
the right eigenvectors:

    <psi_L^m | psi_R^n> = sum_i vR[i,m] * vR[i,n]      (no complex conjugation)

The biorthogonal density at site i for mode n is therefore

    rho_i^(n) = Re( vR[i,n]^2 ) / Re( sum_j vR[j,n]^2 ).

Edge modes are identified by LOCALIZATION (boundary weight), not by hard-coded
absolute energy windows, so the detector is scale-invariant.
"""

import numpy as np
import scipy.linalg as la


def diagonalize(H: np.ndarray) -> tuple:
    """
    Diagonalize H and sort eigenpairs by Re(E).

    Returns
    -------
    evals : ndarray (dim,) complex, sorted by real part
    evecs : ndarray (dim, dim) complex, right eigenvectors as columns
    """
    evals_raw, evecs_raw = la.eig(H)
    idx = np.argsort(evals_raw.real)
    return evals_raw[idx], evecs_raw[:, idx]


def biorthogonal_density(evecs: np.ndarray) -> np.ndarray:
    """
    Biorthogonal probability density for each eigenstate of a complex-symmetric H.

        rho[i, n] = Re(vR[i,n]^2) / Re(sum_j vR[j,n]^2)

    Returns
    -------
    density : ndarray (dim, dim) real, density[site, eigenstate].

    NOTE: for non-Hermitian systems the biorthogonal density is real but NOT
    sign-definite site-by-site; it sums to 1 over sites (Re part). Callers that
    need a manifestly non-negative spatial profile should use |vR|^2 instead and
    say so. We keep both available rather than silently clamping.
    """
    bio_norms = np.einsum("ij,ij->j", evecs, evecs)         # complex biorthogonal norms
    return np.real(evecs ** 2) / np.real(bio_norms)[np.newaxis, :]


def edge_localization_weights(evecs: np.ndarray, edge_cells: int = 1) -> tuple:
    """
    Conventional (|psi|^2) boundary weights for each eigenstate.

    Valid for localization diagnostics/visualisation; for strict non-Hermitian
    observables use the biorthogonal density.

    Returns
    -------
    left_w, right_w : ndarray (n_states,) weight in the left / right boundary cells
    """
    n_edge_sites = 2 * edge_cells
    w = np.abs(evecs) ** 2
    norm = w.sum(axis=0)
    left_w = w[:n_edge_sites, :].sum(axis=0) / norm
    right_w = w[-n_edge_sites:, :].sum(axis=0) / norm
    return left_w, right_w


def find_edge_modes(evals: np.ndarray, evecs: np.ndarray,
                    edge_weight_min: float = 0.30) -> dict:
    """
    Identify topological edge modes by boundary localization (scale-invariant).

    A state is an *edge mode* if a fraction > `edge_weight_min` of its
    conventional weight lies in the single boundary unit cell at either end
    (bulk states have boundary weight ~ 1/N << edge_weight_min). Edge modes are
    then split by linewidth |Im(E)|:

        'protected' : the smaller-|Im(E)| edge mode(s) (A-sublattice, near-zero loss)
        'lossy'     : the larger-|Im(E)|  edge mode(s) (B-sublattice, linewidth ~ gamma)

    The protected/lossy split is placed at the midpoint of the edge modes'
    |Im(E)| range, so no absolute loss scale is hard-coded. Returns empty index
    arrays in the trivial phase (no localized in-gap modes).

    Returns
    -------
    dict: protected_idx, lossy_idx, protected_evals, lossy_evals, edge_idx
    """
    left_w, right_w = edge_localization_weights(evecs, edge_cells=1)
    boundary = np.maximum(left_w, right_w)
    edge_idx = np.where(boundary > edge_weight_min)[0]

    if edge_idx.size == 0:
        empty = np.array([], dtype=int)
        return {"protected_idx": empty, "lossy_idx": empty, "edge_idx": empty,
                "protected_evals": evals[empty], "lossy_evals": evals[empty]}

    loss = np.abs(evals[edge_idx].imag)
    if loss.max() - loss.min() < 1e-9:
        # Hermitian limit (gamma = 0): all edge modes are "protected".
        split = loss.max() + 1.0
    else:
        split = 0.5 * (loss.min() + loss.max())

    protected = edge_idx[loss < split]
    lossy = edge_idx[loss >= split]
    return {
        "protected_idx":   protected,
        "lossy_idx":       lossy,
        "edge_idx":        edge_idx,
        "protected_evals": evals[protected],
        "lossy_evals":     evals[lossy],
    }


def fit_localization_length(evals: np.ndarray, evecs: np.ndarray,
                            N: int, max_floor: float = 1e-8) -> float:
    """
    Fit the exponential decay length of the protected edge mode on its host
    sublattice, returning the AMPLITUDE localization length in unit cells.

    Robustness measures:
      - selects the protected mode and orients the fit from the boundary it is
        localized at (left or right);
      - fits only the monotonically decaying region above a noise floor
        (`max_floor`), excluding the opposite-edge upturn in finite chains.

    Intensity decays as |psi(n)|^2 ~ exp(-2 n / xi), so the fitted slope s of
    log|psi|^2 gives xi = -2 / s. Returns NaN if no protected mode is found.
    """
    info = find_edge_modes(evals, evecs)
    if info["protected_idx"].size == 0:
        return np.nan

    idx = info["protected_idx"][0]
    psi = evecs[:, idx]
    left_w, right_w = edge_localization_weights(evecs[:, [idx]])
    intensity_A = np.abs(psi[0::2]) ** 2                  # A-sublattice sites
    if right_w[0] > left_w[0]:                            # localized on the right edge
        intensity_A = intensity_A[::-1]
    intensity_A = intensity_A / intensity_A[0]            # normalise to boundary cell

    # Keep the leading monotonically-decreasing run above the noise floor.
    keep = [0]
    for n in range(1, len(intensity_A)):
        if intensity_A[n] < max_floor or intensity_A[n] > intensity_A[n - 1]:
            break
        keep.append(n)
    keep = np.array(keep)
    if keep.size < 3:
        return np.nan

    n_cells = keep.astype(float)
    log_int = np.log(intensity_A[keep])
    slope = np.polyfit(n_cells, log_int, 1)[0]
    return float(-2.0 / slope)
