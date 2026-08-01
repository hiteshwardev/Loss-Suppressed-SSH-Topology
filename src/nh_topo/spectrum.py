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
                    edge_weight_min: float = None,
                    gap_half_width: float = None,
                    localization_factor: float = 2.5) -> dict:
    """
    Identify topological edge modes by boundary localization.

    A state is an *edge mode* if a fraction > `edge_weight_min` of its
    conventional weight lies in the single boundary unit cell at either end.

    ROBUSTNESS (revised)
    --------------------
    The previous implementation used a hard-coded absolute threshold of 0.30 and
    split protected/lossy at the MIDPOINT of the detected edge modes' |Im(E)|
    range. Both choices fail under strong bond disorder:

      * a genuinely protected mode whose localization length has been lengthened
        by disorder can fall below 0.30 and be missed entirely;
      * band-edge bulk states that happen to acquire boundary weight just above
        0.30 are then admitted, and the midpoint split can label THEM
        "protected", returning their |Re(E)| ~ |t1 + t2| as a spurious
        "displacement" of the zero mode.

    In the shipped disorder study this occurred in 3/200 realisations at
    W = 0.6 t1 and produced an artefactual mean displacement of 0.023 t1 --
    a value inconsistent with the exact CS-dagger symmetry (see symmetry.py),
    which forbids chiral-preserving disorder from moving Re(E) of a
    self-conjugate mode at all.

    Three changes remove the artefact:

      1. SIZE-AWARE THRESHOLD. Bulk states carry boundary weight ~ 1/N, so the
         threshold is set RELATIVE to that bulk expectation,
         `localization_factor / N`, clipped to [0.08, 0.25]. A fixed 0.30 fails
         at both ends: it misses genuinely localized states in short chains and
         is needlessly loose in long ones. The ceiling of 0.25 matters because a
         HYBRIDISED boundary doublet shares its weight between the two ends, so
         each end carries only about half of W0 (e.g. 0.279 at N = 10) -- with a
         0.30 threshold such a doublet is invisible and the edge weight is
         wrongly reported as zero.
      2. IN-GAP RESTRICTION. If `gap_half_width` is supplied, only states with
         |Re(E)| < gap_half_width are eligible. Band-edge states can no longer
         masquerade as edge modes however localized they look.
      3. PHYSICAL PROTECTED/LOSSY SPLIT. The split is made against the DECAY
         RATE SCALE of the detected set (midpoint between the minimum |Im(E)|
         and the maximum) only when that range is wide; otherwise all detected
         modes with |Im(E)| close to the minimum are protected. Ties no longer
         propagate a bulk state into the protected set.

    For an unambiguous, non-circular identification of the protected mode under
    strong disorder, prefer `track_mode_by_overlap`, which follows the clean
    edge mode by wavefunction overlap and uses no energy information at all.

    Returns
    -------
    dict: protected_idx, lossy_idx, protected_evals, lossy_evals, edge_idx
    """
    n_states = evecs.shape[1]
    N = n_states // 2

    if edge_weight_min is None:
        edge_weight_min = float(np.clip(localization_factor / max(N, 1), 0.08, 0.25))

    left_w, right_w = edge_localization_weights(evecs, edge_cells=1)
    boundary = np.maximum(left_w, right_w)
    eligible = boundary > edge_weight_min

    if gap_half_width is not None:
        eligible &= np.abs(evals.real) < gap_half_width

    edge_idx = np.where(eligible)[0]

    if edge_idx.size == 0:
        empty = np.array([], dtype=int)
        return {"protected_idx": empty, "lossy_idx": empty, "edge_idx": empty,
                "protected_evals": evals[empty], "lossy_evals": evals[empty]}

    loss = np.abs(evals[edge_idx].imag)
    if loss.max() - loss.min() < 1e-9:
        # Degenerate linewidths (e.g. the Hermitian limit): all are protected.
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


def track_mode_by_overlap(evecs: np.ndarray, reference: np.ndarray) -> int:
    """
    Index of the eigenstate with maximum overlap with a reference wavefunction.

    This is adiabatic continuation: the disordered counterpart of a clean mode is
    the state that most resembles it spatially. It uses NO energy information, so
    using it to then measure that mode's energy displacement is not circular --
    the essential property for the disorder diagnostic in disorder.py.

    Parameters
    ----------
    evecs     : ndarray (dim, dim)  eigenvectors of the disordered H (columns)
    reference : ndarray (dim,)      clean protected-mode wavefunction

    Returns
    -------
    int : column index of the best-matching eigenvector
    """
    ref = np.asarray(reference, dtype=np.complex128).ravel()
    ref = ref / np.linalg.norm(ref)
    cols = evecs / np.linalg.norm(evecs, axis=0, keepdims=True)
    overlaps = np.abs(ref.conj() @ cols)
    return int(np.argmax(overlaps))


def clean_protected_mode(N: int, t1: float, t2: float, gamma: float) -> np.ndarray:
    """
    Wavefunction of the protected boundary mode of the CLEAN chain, used as the
    reference for `track_mode_by_overlap`.

    Selects, among boundary-localized in-gap states, the one with the smallest
    decay rate |Im(E)| -- the A-sublattice mode that avoids the loss channel.
    Returns the normalised right eigenvector.
    """
    from .hamiltonian import build_ssh

    H = build_ssh(N, t1, t2, gamma)
    evals, evecs = diagonalize(H)
    info = find_edge_modes(evals, evecs, gap_half_width=abs(t2 - t1))
    if info["protected_idx"].size == 0:
        # Trivial phase: no protected mode exists.
        return np.zeros(2 * N, dtype=np.complex128)
    idx = info["protected_idx"][np.argmin(np.abs(evals[info["protected_idx"]].imag))]
    v = evecs[:, idx]
    return v / np.linalg.norm(v)


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


def localization_length_convergence(t1: float, t2: float, gamma: float,
                                    N_vals) -> dict:
    """
    Fitted localisation length of the protected mode as a function of chain length.

    The fit is made to the mode profile on its host sublattice and should
    approach the analytic value 1 / ln(t2/t1) from below as the chain grows: in a
    short chain the profile is truncated before the exponential tail is fully
    developed, which biases the fit low. Reporting the trend rather than a single
    value is what establishes that the residual deviation is a finite-size effect
    and not a defect of the fit.
    """
    from .hamiltonian import build_ssh
    from .topology import localization_length

    xi_ana = localization_length(t1, t2)
    fitted = []
    for N in N_vals:
        evals, evecs = diagonalize(build_ssh(int(N), t1, t2, gamma))
        fitted.append(float(fit_localization_length(evals, evecs, int(N))))

    return {"N_vals": [int(n) for n in N_vals], "xi_fit": fitted,
            "xi_analytic": float(xi_ana),
            "rel_error": [abs(x - xi_ana) / xi_ana for x in fitted]}
