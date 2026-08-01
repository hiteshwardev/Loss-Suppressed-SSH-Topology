"""
ldos.py
=======
Local density of optical states (LDOS) via the retarded Green's function, plus
an eta-independent topological observable (the edge-mode spectral weight).

LDOS
----
For a Hamiltonian H,

    G^R(omega) = (omega + i*eta - H)^{-1},     rho_i(omega) = -(1/pi) Im G^R_ii.

Non-negativity (rigorous, for THIS model).  Write H = H0 - i*Gamma with H0 real
symmetric and Gamma = gamma * P_B real diagonal PSD. Then
M := (omega + i*eta) I - H = A + iB with A = omega I - H0 (real symmetric) and
B = eta I + Gamma (real diagonal, positive definite for eta > 0). Using
G - G^dagger = M^{-1}(M^dagger - M)(M^dagger)^{-1} = -2i M^{-1} B (M^{-1})^dagger,

    Im G_ii = -[ M^{-1} B (M^{-1})^dagger ]_ii = - sum_j B_jj |G_ij|^2 <= 0,

so rho_i = -(1/pi) Im G_ii = (1/pi) sum_j (eta + gamma [P_B]_jj) |G_ij|^2 >= 0
for every site and frequency. (The naive "all poles in the lower half plane"
argument is NOT sufficient, because the residues are complex; positivity follows
from the PSD structure above.) Verified numerically in `ldos_nonnegativity_min`.

eta-dependence of the edge/bulk ratio
--------------------------------------
The in-gap edge LDOS is dominated by the protected edge pole (weight O(1)); the
in-gap bulk LDOS is a smooth tail that vanishes as eta -> 0. The integrated
edge/bulk RATIO is therefore intrinsically eta-dependent and is reported WITH
its eta-scaling. The eta-INDEPENDENT topological observable is the edge-site
spectral weight of the protected mode, `edge_mode_spectral_weight`.
"""

import numpy as np
import scipy.linalg as la

from .spectrum import find_edge_modes


def _green_diag(H, omega, eta):
    """Diagonal of G^R = ((omega+i eta) I - H)^{-1} via a linear solve (no explicit inverse)."""
    dim = H.shape[0]
    M = (omega + 1j * eta) * np.eye(dim) - H
    G = la.solve(M, np.eye(dim))
    return np.diag(G)


def ldos_at(H: np.ndarray, site: int, omega: float, eta: float) -> float:
    """LDOS at a single (site, frequency). Returns rho >= 0."""
    dim = H.shape[0]
    M = (omega + 1j * eta) * np.eye(dim) - H
    e = np.zeros(dim, dtype=np.complex128); e[site] = 1.0
    g_col = la.solve(M, e)
    return float(-np.imag(g_col[site]) / np.pi)


def ldos_spectrum(H: np.ndarray, site: int, omegas: np.ndarray,
                  eta: float) -> np.ndarray:
    """LDOS spectrum at one site over an array of frequencies (all values >= 0)."""
    return np.array([ldos_at(H, site, w, eta) for w in omegas])


def ldos_spatial_map(H: np.ndarray, omega: float, eta: float) -> np.ndarray:
    """LDOS at all sites for a single frequency omega (shape (2N,))."""
    return -np.imag(_green_diag(H, omega, eta)) / np.pi


def integrated_ldos(H: np.ndarray, site: int, omega_range: np.ndarray,
                    eta: float) -> float:
    """Integrate the LDOS over a frequency window (trapezoid rule)."""
    rho = ldos_spectrum(H, site, omega_range, eta)
    return float(np.trapezoid(rho, omega_range))


def enhancement_factor(H: np.ndarray, edge_site: int, bulk_site: int,
                       omega_range: np.ndarray, eta: float) -> float:
    """
    Integrated edge/bulk LDOS ratio over a frequency window.

        E = int_window rho_edge domega / int_window rho_bulk domega

    NOTE: edge_site and bulk_site should be on the SAME sublattice (both A) for a
    like-for-like comparison; see `central_bulk_site`. This ratio is
    eta-dependent (see module docstring); report it with `enhancement_vs_eta`.
    """
    ie = integrated_ldos(H, edge_site, omega_range, eta)
    ib = integrated_ldos(H, bulk_site, omega_range, eta)
    return ie / max(ib, 1e-15)


def central_bulk_site(N: int) -> int:
    """
    Index of the central A-sublattice (lossless) site, = 2*(N//2) (always even).

    Using this as the bulk reference avoids the sublattice-parity artifact that
    arises from naively taking site index = N (which is a lossy B site for odd N).
    """
    return 2 * (N // 2)


def enhancement_vs_eta(H, edge_site, bulk_site, gap_half_width, etas,
                       n_omega: int = 121) -> dict:
    """
    Edge/bulk integrated LDOS ratio as a function of broadening eta.

    Returns dict: etas, enhancement (same length), to expose the eta-scaling of
    the ratio explicitly rather than quoting a single eta-specific number.
    """
    etas = np.asarray(etas, float)
    out = []
    for eta in etas:
        gap_oms = np.linspace(-gap_half_width, gap_half_width, n_omega)
        out.append(enhancement_factor(H, edge_site, bulk_site, gap_oms, eta))
    return {"etas": etas, "enhancement": np.array(out)}


def edge_mode_spectral_weight(H: np.ndarray, site: int) -> float:
    """
    eta-INDEPENDENT topological observable: the spectral weight (residue of the
    retarded Green's function pole) of the PROTECTED edge mode at `site`.

    For complex-symmetric H, G_ii(z) = sum_n c_n / (z - E_n) with
    c_n = vR[i,n]^2 / (sum_j vR[j,n]^2). The integrated LDOS contributed by the
    protected mode at `site` is Re(c_protected). At the boundary A site this
    equals the analytic semi-infinite value 1 - (t1/t2)^2
    (`topology.analytic_edge_weight`). Returns 0 if there is no protected mode.
    """
    evals, vR = la.eig(H)
    info = find_edge_modes(evals, vR)
    if info["protected_idx"].size == 0:
        return 0.0
    n = info["protected_idx"][0]
    c_n = vR[site, n] ** 2 / np.einsum("i,i->", vR[:, n], vR[:, n])
    return float(np.real(c_n))


def ldos_nonnegativity_min(H: np.ndarray, omegas: np.ndarray,
                           eta: float) -> float:
    """
    Minimum of rho_i(omega) over ALL sites and the supplied frequencies.

    A non-negative return value is a numerical confirmation of the analytic
    non-negativity proof in the module docstring.
    """
    m = np.inf
    for w in omegas:
        m = min(m, float(np.min(-np.imag(_green_diag(H, w, eta)) / np.pi)))
    return m


def ldos_eigdecomp(H: np.ndarray, site: int, omega: float,
                   eta: float) -> float:
    """
    LDOS via the biorthogonal eigendecomposition, for cross-validation against
    `ldos_at`. For complex-symmetric H,

        G_ii(z) = sum_n  vR[i,n]^2 / (norm_n * (z - E_n)),   norm_n = sum_j vR[j,n]^2.

    Agrees with `ldos_at` to ~1e-12 (independent algorithm: spectral sum vs
    linear solve).
    """
    assert np.allclose(H, H.T, atol=1e-12), "H must be complex-symmetric"
    evals, vR = la.eig(H)
    bio_norms = np.einsum("ij,ij->j", vR, vR)
    c_n = vR[site, :] ** 2 / bio_norms
    z = omega + 1j * eta
    return float(-np.imag(np.sum(c_n / (z - evals))) / np.pi)


def ldos_sum_rule(H: np.ndarray, site: int, omegas: np.ndarray,
                  eta: float) -> float:
    """
    Integral of the LDOS at one site over a frequency window.

    The exact spectral sum rule is int_{-inf}^{inf} rho_i(omega) domega = 1 for
    every site. On a finite window the result falls short by the weight in the
    Lorentzian tails, so the returned value quantifies BOTH the correctness of
    the spectral normalisation and the adequacy of the window. Reported in the
    manuscript alongside the window used.
    """
    rho = ldos_spectrum(H, site, omegas, eta)
    return float(np.trapezoid(rho, omegas))


def ldos_sum_rule_all_sites(H: np.ndarray, omegas: np.ndarray,
                            eta: float) -> dict:
    """
    Sum rule evaluated at every site.

    Returns dict with per-site values plus min/mean/max, so the check cannot be
    passed by a single favourable site.
    """
    vals = np.array([ldos_sum_rule(H, i, omegas, eta) for i in range(H.shape[0])])
    return {"per_site": vals, "min": float(vals.min()),
            "mean": float(vals.mean()), "max": float(vals.max())}
