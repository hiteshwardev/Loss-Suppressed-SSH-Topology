"""
topology.py
===========
Bulk topological invariants and analytic edge-mode properties of the
non-Hermitian SSH model.

The relevant invariant is the winding number of the off-diagonal Bloch element
q(k) = t1 + t2*exp(-ik). For the loss-only SSH model gamma enters only the
diagonal of H(k) and does not affect q(k), so the invariant is identical to the
Hermitian SSH case. Reciprocal hopping (H_ij = H_ji) ensures the standard
bulk-edge correspondence (GBZ = BZ); no GBZ correction is required.

Convention:
    q(k) = t1 + t2 * exp(-ik)
    nu = -1  for t2 > t1  (topological, |nu| = 1)
    nu =  0  for t2 < t1  (trivial)
"""

import numpy as np


def winding_number(t1: float, t2: float, num_k: int = 4000) -> float:
    """
    Grid-robust winding number of q(k) = t1 + t2*exp(-ik) about the origin.

    Computed as the sum of principal-value phase increments around the CLOSED
    Brillouin-zone loop (the wrap-around segment from the last k-point back to
    the first is included). This makes the result essentially independent of
    `num_k`: it returns -1.0000 (topological) / 0.0000 (trivial) for any
    sufficiently dense grid, unlike the naive (phases[-1]-phases[0]) estimator
    which systematically under-counts by one grid segment.

    Returns the raw signed winding; use `topological_invariant` for |nu|.
    """
    k = np.linspace(-np.pi, np.pi, num_k, endpoint=False)
    q = t1 + t2 * np.exp(-1j * k)
    q_closed = np.append(q, q[0])                      # close the loop
    dphi = np.angle(q_closed[1:] / q_closed[:-1])      # increments in (-pi, pi]
    return float(np.sum(dphi) / (2 * np.pi))


def topological_invariant(t1: float, t2: float, num_k: int = 4000) -> int:
    """Return |nu| (0 = trivial, 1 = topological)."""
    return abs(int(round(winding_number(t1, t2, num_k))))


def phase_boundary(t1: float) -> float:
    """Inter-cell coupling at the topological phase transition: t2 = t1."""
    return t1


def localization_length(t1: float, t2: float) -> float:
    """
    Analytic amplitude localization length of the edge mode (in unit cells):

        xi = 1 / ln(t2 / t1)        (edge amplitude ~ exp(-n/xi)).

    Valid for t2 > t1 (topological phase); diverges at the transition.
    """
    if t2 <= t1:
        return np.inf
    return 1.0 / np.log(t2 / t1)


def analytic_edge_weight(t1: float, t2: float) -> float:
    """
    Analytic spectral weight of the semi-infinite topological zero mode on the
    boundary unit cell (A-sublattice edge site).

    The semi-infinite left-edge zero mode is psi_A(n) ∝ (-t1/t2)^n, n = 0,1,2,...
    Normalising sum_n |psi_A(n)|^2 = 1 / (1 - (t1/t2)^2) gives the edge-site
    weight

        W0 = 1 - (t1/t2)^2.

    This is the eta-INDEPENDENT topological observable benchmarked numerically
    against the biorthogonal edge-pole residue in ldos.edge_mode_spectral_weight.
    Returns 0 in the trivial phase (no protected edge mode).
    """
    if t2 <= t1:
        return 0.0
    return 1.0 - (t1 / t2) ** 2


def qk_trajectory(t1: float, t2: float, num_k: int = 600) -> np.ndarray:
    """Trajectory of q(k) in the complex plane as k sweeps [-pi, pi)."""
    k = np.linspace(-np.pi, np.pi, num_k, endpoint=False)
    return t1 + t2 * np.exp(-1j * k)


def compute_bloch_bands(t1: float, t2: float, gamma: float,
                        num_k: int = 600) -> tuple:
    """
    Two complex Bloch bands over the Brillouin zone.

    Uses the closed-form eigenvalues E(k) = -i*gamma/2 +/- sqrt(|q|^2 -
    (gamma/2)^2) (branch chosen by sign of Re) rather than a per-k numerical
    eigensolve, so the two bands are returned without ordering ambiguity.

    Returns
    -------
    k_vals : ndarray (num_k,)
    bands  : ndarray (num_k, 2) complex, columns = lower/upper band by Re(E)
    """
    k_vals = np.linspace(-np.pi, np.pi, num_k)
    q = t1 + t2 * np.exp(-1j * k_vals)
    root = np.sqrt(np.abs(q) ** 2 - (gamma / 2.0) ** 2 + 0j)
    e_minus = -1j * gamma / 2.0 - root
    e_plus = -1j * gamma / 2.0 + root
    bands = np.stack([e_minus, e_plus], axis=1)
    return k_vals, bands
