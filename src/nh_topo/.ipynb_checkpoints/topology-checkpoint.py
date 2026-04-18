"""
topology.py
===========
Bulk topological invariants for the non-Hermitian SSH model.

The relevant invariant is the winding number of the off-diagonal Bloch
element q(k) = t1 + t2*exp(-ik). For the loss-only SSH model, gamma enters
only the diagonal of H(k) and does not affect q(k), so the topological
invariant is identical to the Hermitian SSH case.

Reciprocal hopping (H_ij = H_ji) ensures the standard bulk-edge
correspondence holds — GBZ corrections are not required here.

Convention used throughout:
    q(k) = t1 + t2 * exp(-ik)
    nu = +1  for t2 < t1  (trivial)
    nu = -1  for t2 > t1  (topological, |nu| = 1)
"""

import numpy as np


def winding_number(t1: float, t2: float, num_k: int = 4000) -> float:
    """
    Compute the winding number of q(k) = t1 + t2*exp(-ik) around the origin.

    Uses phase unwrapping over a dense k-grid. The raw (signed) winding is
    returned; take abs(round(result)) for the topological invariant |nu|.

    Parameters
    ----------
    t1    : float  Intra-cell hopping.
    t2    : float  Inter-cell hopping.
    num_k : int    Number of k-points (default 4000; increase for near t1=t2).

    Returns
    -------
    float  Raw winding (≈ -1 for topological, ≈ 0 for trivial).
    """
    k = np.linspace(-np.pi, np.pi, num_k, endpoint=False)
    q = t1 + t2 * np.exp(-1j * k)
    phases = np.unwrap(np.angle(q))
    return (phases[-1] - phases[0]) / (2 * np.pi)


def topological_invariant(t1: float, t2: float, num_k: int = 4000) -> int:
    """
    Return |nu| (0 = trivial, 1 = topological).
    """
    return abs(int(round(winding_number(t1, t2, num_k))))


def phase_boundary(t1: float) -> float:
    """Coupling ratio at the topological phase transition: t2/t1 = 1."""
    return t1


def localization_length(t1: float, t2: float) -> float:
    """
    Analytical localization length of the edge mode (in unit cells):
        xi = 1 / ln(t2 / t1)

    Valid for t2 > t1 (topological phase). Diverges at the phase transition.
    """
    if t2 <= t1:
        return np.inf
    return 1.0 / np.log(t2 / t1)


def qk_trajectory(t1: float, t2: float, num_k: int = 600) -> np.ndarray:
    """
    Return the trajectory of q(k) in the complex plane as k sweeps [-pi, pi).
    Used to visualise the winding around the origin.
    """
    k = np.linspace(-np.pi, np.pi, num_k, endpoint=False)
    return t1 + t2 * np.exp(-1j * k)


def compute_bloch_bands(t1: float, t2: float, gamma: float,
                        num_k: int = 600) -> tuple:
    """
    Compute the two complex Bloch bands over the Brillouin zone.

    Returns
    -------
    k_vals  : ndarray (num_k,)
    bands   : ndarray (num_k, 2), complex  — eigenvalues of H(k), sorted by Re(E)
    """
    from scipy import linalg as la
    from .hamiltonian import bloch_hamiltonian

    k_vals = np.linspace(-np.pi, np.pi, num_k)
    bands = np.zeros((num_k, 2), dtype=np.complex128)
    for i, k in enumerate(k_vals):
        evs = la.eigvals(bloch_hamiltonian(k, t1, t2, gamma))
        bands[i] = sorted(evs, key=lambda x: x.real)
    return k_vals, bands
