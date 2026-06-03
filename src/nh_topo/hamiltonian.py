"""
hamiltonian.py
==============
Real-space non-Hermitian SSH Hamiltonians.

Model: 1D chain, N unit cells, 2 sites per cell (A = even index, B = odd index).
Non-Hermiticity: loss rate gamma on the B sublattice only.

    H = sum_n [ t1_n |n,A><n,B| + t2_n |n+1,A><n,B| + h.c. ]
      + sum_i eps_i |i><i|
      - i*gamma * sum_n |n,B><n,B|

Structural properties enforced by construction (real, reciprocal hopping +
real on-site disorder + purely imaginary B-loss):

  - H = H^T            (complex-symmetric; left eigvec = (right eigvec)^T)
  - Im(E_n) in [-gamma, 0]   (passive system; LDOS provably >= 0, see ldos.py)
  - No non-Hermitian skin effect (hopping is reciprocal: H_ij = H_ji)

The general builder `build_ssh_general` accepts per-bond couplings and per-site
on-site energies so that bond (chiral-preserving) and on-site (chiral-breaking)
disorder are constructed from the same primitive.
"""

import numpy as np


def build_ssh_general(N, gamma, t1_bonds, t2_bonds, onsite=None):
    """
    General finite non-Hermitian SSH Hamiltonian (OBC) with B-sublattice loss.

    Parameters
    ----------
    N         : int                   Number of unit cells.
    gamma     : float                 Loss rate on every B site (>= 0).
    t1_bonds  : float or array (N,)   Intra-cell hoppings (cell n: A_n <-> B_n).
    t2_bonds  : float or array (N-1,) Inter-cell hoppings (B_n <-> A_{n+1}).
    onsite    : None or array (2N,)    Real on-site energies (default: zeros).

    Returns
    -------
    H : ndarray (2N, 2N), complex128
    """
    dim = 2 * N
    H = np.zeros((dim, dim), dtype=np.complex128)

    t1_bonds = np.broadcast_to(np.asarray(t1_bonds, float), (N,))
    t2_bonds = np.broadcast_to(np.asarray(t2_bonds, float), (max(N - 1, 0),))

    for n in range(N):
        a, b = 2 * n, 2 * n + 1
        H[b, b] = -1j * gamma                      # loss on B sublattice
        H[a, b] = H[b, a] = t1_bonds[n]            # intra-cell hopping
        if n < N - 1:
            H[a + 2, b] = H[b, a + 2] = t2_bonds[n]  # inter-cell hopping

    if onsite is not None:
        onsite = np.asarray(onsite, float)
        if onsite.shape != (dim,):
            raise ValueError(f"onsite must have shape ({dim},), got {onsite.shape}")
        H[np.diag_indices(dim)] += onsite          # real shifts keep H = H^T

    return H


def build_ssh(N: int, t1: float, t2: float, gamma: float) -> np.ndarray:
    """
    Homogeneous non-Hermitian SSH Hamiltonian with OBC and B-sublattice loss.

    Convenience wrapper around `build_ssh_general` with uniform couplings.
    """
    return build_ssh_general(N, gamma, t1, t2, onsite=None)


def build_ssh_pbc(N: int, t1: float, t2: float, gamma: float) -> np.ndarray:
    """
    SSH Hamiltonian with periodic boundary conditions (ring geometry).
    Adds wrap-around hopping B(N-1) <-> A(0) with amplitude t2.
    Used to verify bulk-edge correspondence: PBC has no edge states.
    """
    H = build_ssh(N, t1, t2, gamma).copy()
    H[0, 2 * N - 1] = t2
    H[2 * N - 1, 0] = t2
    return H


def bloch_hamiltonian(k: float, t1: float, t2: float, gamma: float) -> np.ndarray:
    """
    2x2 Bloch Hamiltonian H(k) for the loss-only SSH model.

        H(k) = [[0,         q(k)    ],
                [q*(k),    -i*gamma ]],     q(k) = t1 + t2 * exp(-ik).

    The lower-left element q*(k) follows from H = H^T. gamma enters only the
    diagonal, so it does not affect q(k) (hence the winding number is
    gamma-independent).
    """
    q = t1 + t2 * np.exp(-1j * k)
    return np.array([[0.0 + 0j,   q         ],
                     [q.conj(),  -1j * gamma]], dtype=np.complex128)


def exceptional_point_threshold(t1: float, t2: float) -> float:
    """
    Bulk exceptional-point / PT-breaking loss threshold.

    H(k) eigenvalues are  -i*gamma/2 +/- sqrt(|q(k)|^2 - (gamma/2)^2).
    The radicand is real for all k iff gamma/2 < min_k |q(k)| = |t2 - t1|, so

        gamma_EP = 2 * |t2 - t1|.

    For gamma < gamma_EP both bulk bands share Im(E) = -gamma/2 (unbroken,
    "passive PT"); for gamma > gamma_EP the bands split in Im(E) near the
    Re-gap centre (broken). NOTE this is 2|t2-t1|, not |t2-t1|.
    """
    return 2.0 * abs(t2 - t1)


def verify_properties(H: np.ndarray) -> dict:
    """
    Sanity-check key mathematical properties of the Hamiltonian.

    Returns dict with keys: is_symmetric, all_passive, is_hermitian, dim.
    """
    import scipy.linalg as la
    evals = la.eigvals(H)
    return {
        "is_symmetric":  bool(np.allclose(H, H.T)),
        "all_passive":   bool(np.all(evals.imag <= 1e-9)),
        "is_hermitian":  bool(np.allclose(H, H.conj().T)),
        "dim":           int(H.shape[0]),
    }
