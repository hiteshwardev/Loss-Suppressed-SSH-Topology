"""
analytics.py
============
Independent, semi-analytic benchmarks for the non-Hermitian SSH LDOS.

These provide an INDEPENDENT cross-check of the finite-chain Green's-function
LDOS (ldos.py) and the biorthogonal eigendecomposition (ldos.ldos_eigdecomp):

  * `surface_green_function` / `surface_ldos` — the surface (edge) Green's
    function of the SEMI-INFINITE SSH chain via the Lopez-Sancho / Sancho-Rubio
    iterative decimation method. This is a different algorithm (renormalisation
    of principal layers) and a different geometry (truly semi-infinite, hence
    free of finite-size contamination) from the finite-matrix inversion.

  * `bulk_ldos` — the bulk (infinite-chain) LDOS from the analytic k-integrated
    Green's function, used as the in-gap bulk reference.

Principal layer = one unit cell (orbitals A, B). Layer Hamiltonian and the
forward inter-layer coupling V (B_n -> A_{n+1}, amplitude t2):

    h = [[0,  t1],          V = [[0, 0],
         [t1, -i*gamma]]         [t2, 0]]
"""

import numpy as np


def _layer_matrices(t1, t2, gamma):
    h = np.array([[0.0 + 0j, t1], [t1, -1j * gamma]], dtype=np.complex128)
    V = np.array([[0.0, 0.0], [t2, 0.0]], dtype=np.complex128)   # layer n -> n+1
    return h, V


def surface_green_function(z, t1, t2, gamma, tol=1e-14, max_iter=200):
    """
    2x2 surface Green's function g_s(z) of the semi-infinite SSH chain (left edge,
    surface layer = unit cell 0) via Sancho-Rubio decimation.

    Solves the self-consistent surface equation g_s = (z I - h - V g_s V^dag)^{-1}
    by the standard quadratically-convergent iteration. `z = omega + i*eta`.

    Returns
    -------
    g_s : ndarray (2, 2) complex  — g_s[0,0] is the edge A-site GF, g_s[1,1] the
          edge B-site GF.
    """
    h, V = _layer_matrices(t1, t2, gamma)
    I = np.eye(2, dtype=np.complex128)
    Vdag = V.conj().T

    eps_s = h.copy()          # effective surface on-site
    eps = h.copy()            # effective bulk on-site
    alpha = V.copy()          # forward coupling (couples to layers on the right)
    beta = Vdag.copy()        # backward coupling

    for _ in range(max_iter):
        g = np.linalg.inv(z * I - eps)
        agb = alpha @ g @ beta
        bga = beta @ g @ alpha
        eps_s = eps_s + agb
        eps = eps + agb + bga
        alpha = alpha @ g @ alpha
        beta = beta @ g @ beta
        if np.max(np.abs(alpha)) + np.max(np.abs(beta)) < tol:
            break

    return np.linalg.inv(z * I - eps_s)


def surface_ldos(omega, t1, t2, gamma, eta, sublattice=0):
    """
    Edge LDOS of the semi-infinite chain at the surface A (sublattice=0) or
    B (sublattice=1) site: rho = -(1/pi) Im g_s[sublattice, sublattice].
    """
    z = omega + 1j * eta
    g_s = surface_green_function(z, t1, t2, gamma)
    return float(-np.imag(g_s[sublattice, sublattice]) / np.pi)


def surface_ldos_spectrum(omegas, t1, t2, gamma, eta, sublattice=0):
    """Vectorised `surface_ldos` over an array of frequencies."""
    return np.array([surface_ldos(w, t1, t2, gamma, eta, sublattice) for w in omegas])


def bulk_ldos(omega, t1, t2, gamma, eta, sublattice=0):
    """
    Infinite-chain (bulk) LDOS per site from the k-integrated Green's function:

        rho_bulk(omega) = -(1/pi) (1/2pi) int_BZ Im [ (z - H(k))^{-1} ]_{ss} dk.

    Uses a dense Brillouin-zone quadrature; serves as the gap reference that is
    free of any boundary contribution.
    """
    z = omega + 1j * eta
    ks = np.linspace(-np.pi, np.pi, 2001)
    q = t1 + t2 * np.exp(-1j * ks)
    # (z - H(k))^{-1}, H(k) = [[0,q],[q*, -i gamma]]; diagonal elements in closed form.
    a = z                      # z - 0
    d = z + 1j * gamma         # z - (-i gamma)
    det = a * d - np.abs(q) ** 2
    g_ss = (d if sublattice == 0 else a) / det
    integrand = -np.imag(g_ss) / np.pi
    return float(np.trapezoid(integrand, ks) / (2 * np.pi))
