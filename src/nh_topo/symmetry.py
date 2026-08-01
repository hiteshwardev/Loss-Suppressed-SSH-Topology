"""
symmetry.py
===========
Symmetry classification of the passive, sublattice-lossy SSH Hamiltonian.

Why this module exists
----------------------
It is tempting to describe the loss-only SSH chain as "chirally symmetric"
because its Hermitian part is. That is wrong, and the error matters: the
protection statement it licenses is not the one the model actually obeys.

Write the Hamiltonian as

    H = H0 - i*gamma*P_B ,

with H0 the (real, symmetric, purely off-diagonal) SSH Hamiltonian and P_B the
projector onto the B sublattice. In the sublattice basis

    sigma_z = diag(+1, -1, +1, -1, ...) ,     P_B = (I - sigma_z)/2 .

HERMITIAN CHIRAL SYMMETRY FAILS
    sigma_z H sigma_z = -H0 - i*gamma*P_B  !=  -H ,
    || sigma_z H sigma_z + H ||_max = 2*gamma .
The anti-Hermitian diagonal violates it by exactly 2*gamma. See
`chiral_residual`.

THE CORRECT STATEMENT (CS-dagger / sublattice symmetry)
Using P_B = (I - sigma_z)/2,

    H = H0 - i*(gamma/2)*I + i*(gamma/2)*sigma_z ,

so define the SHIFTED Hamiltonian

    Htilde := H + i*(gamma/2)*I = H0 + i*(gamma/2)*sigma_z .

The shift removes the mean decay rate: it translates every eigenvalue by
+i*gamma/2 and leaves every eigenvector -- hence the entire spatial LDOS
profile -- unchanged. Using H0^dag = H0 and sigma_z^dag = sigma_z,

    sigma_z Htilde sigma_z = -H0 + i*(gamma/2)*sigma_z = -Htilde^dag .      (*)

Equation (*) is the non-Hermitian sublattice (chiral) symmetry, denoted CS-dagger
or SLS in the Bernard-LeClair / Kawabata-Shiozaki-Ueda-Sato classification
[Phys. Rev. X 9, 041015 (2019)]. See `cs_dagger_residual`.

SPECTRAL CONSEQUENCE -- AND WHY IT DIFFERS FROM THE HERMITIAN CASE
Hermitian chiral symmetry forces eigenvalues into +/- pairs. CS-dagger instead
forces spec(Htilde) to be invariant under

    E  ->  -E*  ,

i.e. symmetric about the IMAGINARY axis. A single eigenvalue is self-conjugate
under this map iff Re(E) = 0. So CS-dagger pins the REAL ENERGY of an unpaired
mode while leaving its decay rate free. Verified by `spectral_symmetry_residual`.

Crucially, a mode can leave the imaginary axis only IN COMPANY: two eigenvalues
at equal Im(E) may pair and move to +/- delta together. That is the only channel
by which the pinning can fail -- and it is the channel that sublattice loss
closes, because loss gives the two boundary modes DIFFERENT decay rates
(Im E ~ 0 on A, Im E ~ -gamma on B), leaving neither with a partner. This is the
mechanism behind the loss-suppressed hybridisation reported in
`disorder.hybridization_study`.

WHICH PERTURBATIONS PRESERVE CS-DAGGER
  * bond (off-diagonal, real)  : H0 stays Hermitian and off-diagonal -> (*) holds
                                 exactly. Residual 0 at any strength.
  * on-site (diagonal, real V) : adds +V to sigma_z Htilde sigma_z and -V to
                                 -Htilde^dag -> breaks (*) by 2V.
This is the precise sense in which the two disorder classes used in this study
are symmetry-distinct. See `symmetry_class_of_perturbation`.
"""

import numpy as np


def sublattice_operator(N: int) -> np.ndarray:
    """
    Sublattice (chiral) operator sigma_z = diag(+1, -1, +1, -1, ...).

    +1 on A sites (even index, lossless), -1 on B sites (odd index, lossy).
    """
    diag = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(2 * N)])
    return np.diag(diag)


def shifted_hamiltonian(H: np.ndarray, gamma: float) -> np.ndarray:
    """
    Htilde = H + i*(gamma/2)*I  -- the Hamiltonian with the mean decay removed.

    This is a rigid translation of the spectrum by +i*gamma/2. Eigenvectors, and
    therefore all spatial observables including the LDOS profile, are unchanged.
    """
    return H + 1j * (gamma / 2.0) * np.eye(H.shape[0], dtype=np.complex128)


def chiral_residual(H: np.ndarray) -> float:
    """
    || sigma_z H sigma_z - (-H) ||_max : the failure of HERMITIAN chiral symmetry.

    Returns 2*gamma for the lossy chain (nonzero => symmetry broken) and 0 in the
    Hermitian limit. This is the quantity a referee will ask about.
    """
    N = H.shape[0] // 2
    sz = sublattice_operator(N)
    return float(np.abs(sz @ H @ sz + H).max())


def cs_dagger_residual(H: np.ndarray, gamma: float) -> float:
    """
    || sigma_z Htilde sigma_z - (-Htilde^dag) ||_max : the CS-dagger test, Eq. (*).

    Returns ~0 (machine precision) whenever the perturbation is real and purely
    off-diagonal, for ANY gamma and ANY bond-disorder strength.
    """
    N = H.shape[0] // 2
    sz = sublattice_operator(N)
    Ht = shifted_hamiltonian(H, gamma)
    return float(np.abs(sz @ Ht @ sz + Ht.conj().T).max())


def spectral_symmetry_residual(H: np.ndarray, gamma: float) -> float:
    """
    max_E dist(-E*, spec(Htilde)) : numerical test that the spectrum is closed
    under E -> -E*, i.e. symmetric about the imaginary axis.

    Returns ~1e-15 when CS-dagger holds.
    """
    ev = np.linalg.eigvals(shifted_hamiltonian(H, gamma))
    return float(max(np.min(np.abs(ev + np.conj(e))) for e in ev))


def count_pinned_modes(H: np.ndarray, tol: float = 1e-8) -> int:
    """
    Number of eigenvalues with |Re(E)| < tol, i.e. modes pinned to the imaginary
    axis (self-conjugate under E -> -E*).

    For the topological phase with sublattice loss this is 2 (the A- and
    B-sublattice boundary modes) and is INDEPENDENT of chiral-preserving bond
    disorder strength -- the sharpest statement of the protection.
    """
    return int(np.sum(np.abs(np.linalg.eigvals(H).real) < tol))


def symmetry_class_of_perturbation(H_clean: np.ndarray, H_pert: np.ndarray,
                                   gamma: float, tol: float = 1e-10) -> str:
    """
    Classify a perturbation by whether it preserves CS-dagger.

    Returns "CS-dagger preserving" (bond-type) or "CS-dagger breaking"
    (on-site-type), decided by the residual of Eq. (*) on the perturbed
    Hamiltonian.
    """
    return ("CS-dagger preserving" if cs_dagger_residual(H_pert, gamma) < tol
            else "CS-dagger breaking")


def symmetry_report(H: np.ndarray, gamma: float) -> dict:
    """
    Full symmetry diagnostic for one Hamiltonian.

    Returns
    -------
    dict with keys
      chiral_residual         : 2*gamma  (Hermitian chiral symmetry is broken)
      cs_dagger_residual      : ~0       (CS-dagger holds)
      spectral_symmetry       : ~0       (spectrum closed under E -> -E*)
      n_pinned                : number of modes with Re(E) = 0
      hermitian_chiral_holds  : bool
      cs_dagger_holds         : bool
    """
    rc = chiral_residual(H)
    rcs = cs_dagger_residual(H, gamma)
    return {
        "chiral_residual":        rc,
        "cs_dagger_residual":     rcs,
        "spectral_symmetry":      spectral_symmetry_residual(H, gamma),
        "n_pinned":               count_pinned_modes(H),
        "hermitian_chiral_holds": bool(rc < 1e-10),
        "cs_dagger_holds":        bool(rcs < 1e-10),
    }


# ---------------------------------------------------------------------------
# Effective two-level model for the boundary doublet
# ---------------------------------------------------------------------------

def bare_hybridization(N: int, t1: float, t2: float) -> float:
    """
    Bare hybridisation splitting delta_0 of the two boundary modes in the
    HERMITIAN chain of N unit cells: delta_0 = min|Re E| at gamma = 0.

    In a finite chain the left and right zero modes overlap and repel, giving
    delta_0 ~ exp(-N/xi) with xi = 1/ln(t2/t1). This sets the scale that loss
    must exceed for the pinning to become exact (see `edge_exceptional_point`).
    """
    from .hamiltonian import build_ssh
    return float(np.min(np.abs(np.linalg.eigvals(build_ssh(N, t1, t2, 0.0)).real)))


def edge_exceptional_point(N: int, t1: float, t2: float) -> float:
    """
    Loss threshold at which the BOUNDARY DOUBLET reaches its exceptional point.

    In the shifted frame the two boundary modes form the effective two-level
    non-Hermitian Hamiltonian

        H_eff = [[ +i*gamma/2 ,  delta_0    ],
                 [  delta_0   , -i*gamma/2  ]] ,

    whose eigenvalues are  E = +/- sqrt(delta_0^2 - gamma^2/4).  Hence

        gamma < 2*delta_0 : radicand positive -> the doublet splits in Re(E)
                            (ordinary Hermitian-like hybridisation);
        gamma > 2*delta_0 : radicand negative -> eigenvalues become purely
                            imaginary, Re(E) is pinned to 0, and the two modes
                            separate along the imaginary axis instead.

    The coalescence at gamma = 2*delta_0 is an exceptional point of the EDGE
    sector. It is distinct from -- and exponentially smaller than -- the bulk EP
    at gamma = 2|t2 - t1| (`hamiltonian.exceptional_point_threshold`), because
    delta_0 ~ exp(-N/xi). For any chain of moderate length even very weak loss
    therefore places the system on the protected side.

    Verified against the full spectrum to five significant figures in
    tests/test_symmetry.py.
    """
    return 2.0 * bare_hybridization(N, t1, t2)


def two_level_splitting(gamma, delta0: float) -> np.ndarray:
    """
    Analytic Re(E) splitting of the boundary doublet, |Re sqrt(delta0^2 - g^2/4)|.

    Returns delta_0 at gamma = 0, falls to zero at the edge exceptional point
    gamma = 2*delta_0, and remains exactly zero beyond it.
    """
    gamma = np.asarray(gamma, float)
    radicand = delta0 ** 2 - (gamma / 2.0) ** 2
    return np.sqrt(np.clip(radicand, 0.0, None))
