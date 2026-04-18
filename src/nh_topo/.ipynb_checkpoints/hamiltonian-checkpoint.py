"""
hamiltonian.py
==============
Real-space non-Hermitian SSH Hamiltonians.

Model: 1D chain, N unit cells, 2 sites per cell (A=even, B=odd indices).
Non-Hermiticity: loss rate gamma on B sublattice only.

    H = sum_n [ t1 |n,A><n,B| + t2 |n+1,A><n,B| + h.c. ]
      - i*gamma * sum_n |n,B><n,B|

Properties enforced:
  - H = H^T  (complex-symmetric, reciprocal hopping)
  - Im(E_n) <= 0 for all n  (passive system, positive LDOS guaranteed)
  - No non-Hermitian skin effect (reciprocal hoppings)
"""

import numpy as np


def build_ssh(N: int, t1: float, t2: float, gamma: float) -> np.ndarray:
    """
    Finite non-Hermitian SSH Hamiltonian with OBC and sublattice-B loss.

    Parameters
    ----------
    N     : int    Number of unit cells.
    t1    : float  Intra-cell hopping amplitude.
    t2    : float  Inter-cell hopping amplitude.
    gamma : float  Loss rate on B sublattice (>= 0).

    Returns
    -------
    H : ndarray, shape (2N, 2N), dtype complex128
    """
    dim = 2 * N
    H = np.zeros((dim, dim), dtype=np.complex128)
    for n in range(N):
        a, b = 2 * n, 2 * n + 1
        H[b, b] = -1j * gamma          # loss on B sublattice
        H[a, b] = H[b, a] = t1        # intra-cell hopping
        if n < N - 1:
            H[a + 2, b] = H[b, a + 2] = t2  # inter-cell hopping
    return H


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
                [q*(k),    -i*gamma ]]

    where q(k) = t1 + t2 * exp(-ik).

    The lower-left element q*(k) follows from H = H^T (real symmetric hopping
    combined with purely imaginary diagonal).

    Parameters
    ----------
    k     : float   Crystal momentum in [-pi, pi).
    t1    : float   Intra-cell hopping.
    t2    : float   Inter-cell hopping.
    gamma : float   Loss rate on B sublattice.
    """
    q = t1 + t2 * np.exp(-1j * k)
    return np.array([[0.0 + 0j,   q         ],
                     [q.conj(),  -1j * gamma]], dtype=np.complex128)


def verify_properties(H: np.ndarray) -> dict:
    """
    Sanity-check key mathematical properties of the Hamiltonian.

    Returns dict with keys: is_symmetric, all_passive, is_hermitian.
    """
    import scipy.linalg as la
    evals = la.eigvals(H)
    return {
        "is_symmetric":  np.allclose(H, H.T),
        "all_passive":   bool(np.all(evals.imag <= 1e-9)),
        "is_hermitian":  np.allclose(H, H.conj().T),
        "dim":           H.shape[0],
    }

