"""
ldos.py
=======
Local density of optical states (LDOS) via the retarded Green's function.

For a passive non-Hermitian photonic Hamiltonian H with Im(E_n) <= 0:

    G^R(omega) = (omega + i*eta - H)^{-1}

    rho_i(omega) = -(1/pi) Im G^R_ii(omega) >= 0   for all omega

Non-negativity is guaranteed because all poles lie in the lower half plane,
so Im(G^R_ii) <= 0 always holds for this class of Hamiltonian.

Enhancement metric: integrated LDOS ratio over the gap region, which is robust
to the choice of eta (unlike the peak-height ratio).
"""

import numpy as np
import scipy.linalg as la


def ldos_at(H: np.ndarray, site: int, omega: float, eta: float) -> float:
    """
    LDOS at a single (site, frequency) point.

    Parameters
    ----------
    H     : (dim, dim) complex array
    site  : int    Site index (0 to dim-1)
    omega : float  Real frequency
    eta   : float  Lorentzian broadening (> 0)

    Returns
    -------
    float  LDOS value >= 0
    """
    dim = H.shape[0]
    G = la.inv((omega + 1j * eta) * np.eye(dim) - H)
    return float(-np.imag(G[site, site]) / np.pi)


def ldos_spectrum(H: np.ndarray, site: int, omegas: np.ndarray,
                  eta: float) -> np.ndarray:
    """
    LDOS spectrum at a single site over an array of frequencies.

    Returns ndarray of shape (len(omegas),), all values >= 0.
    """
    return np.array([ldos_at(H, site, w, eta) for w in omegas])


def ldos_spatial_map(H: np.ndarray, omega: float, eta: float) -> np.ndarray:
    """
    LDOS at all sites for a single frequency omega.

    Returns ndarray of shape (2N,).
    """
    dim = H.shape[0]
    G = la.inv((omega + 1j * eta) * np.eye(dim) - H)
    return np.array([-np.imag(G[i, i]) / np.pi for i in range(dim)])


def integrated_ldos(H: np.ndarray, site: int, omega_range: np.ndarray,
                    eta: float) -> float:
    """
    Integrate the LDOS over a frequency window using the trapezoid rule.
    """
    rho = ldos_spectrum(H, site, omega_range, eta)
    return float(np.trapezoid(rho, omega_range))


def enhancement_factor(H: np.ndarray, edge_site: int, bulk_site: int,
                       omega_range: np.ndarray, eta: float) -> float:
    """
    LDOS enhancement factor: ratio of integrated edge LDOS to bulk LDOS
    over the specified frequency window.

        E = int_window rho_edge(omega) domega
            / int_window rho_bulk(omega) domega

    Using integrated (not peak) LDOS makes E robust to the choice of eta.

    Parameters
    ----------
    H          : Hamiltonian
    edge_site  : site index for edge LDOS
    bulk_site  : site index for bulk LDOS
    omega_range: frequency grid over the gap window
    eta        : broadening

    Returns
    -------
    float  Enhancement (>1 indicates edge enhancement)
    """
    ie = integrated_ldos(H, edge_site, omega_range, eta)
    ib = integrated_ldos(H, bulk_site, omega_range, eta)
    return ie / max(ib, 1e-12)


def ldos_eigdecomp(H: np.ndarray, site: int, omega: float,
                   eta: float) -> float:
    """
    LDOS via eigendecomposition — for cross-validation against ldos_at().

    For complex-symmetric H (H^T = H, which holds for our SSH model):

        G_ii(z) = sum_n  vR[i,n]^2 / (norm_n * (z - E_n))

    where norm_n = sum_j vR[j,n]^2 is the complex biorthogonal norm.

    This formula agrees with ldos_at() to machine precision (< 1e-12).
    """
    assert np.allclose(H, H.T, atol=1e-12), "H must be complex-symmetric"
    evals, vR = la.eig(H)
    bio_norms = np.einsum("ij,ij->j", vR, vR)   # sum_j vR[j,n]^2
    c_n = vR[site, :] ** 2 / bio_norms
    z = omega + 1j * eta
    return float(-np.imag(np.sum(c_n / (z - evals))) / np.pi)
