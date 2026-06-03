"""
disorder.py
===========
Disorder models and statistical analysis for LDOS-enhancement robustness.

Two physically distinct disorder classes are implemented and analysed side by
side, because they probe DIFFERENT physics:

  "bond"    : random hopping amplitudes t1_n, t2_n -> t + U[-W/2, W/2].
              PRESERVES chiral (sublattice) symmetry. The SSH zero mode stays
              pinned at Re(E) = 0 -> this is the disorder class against which the
              topological edge mode is actually protected.

  "on_site" : random on-site energies eps_i -> U[-W/2, W/2].
              BREAKS chiral symmetry. The zero mode drifts off Re(E) = 0 and the
              protection is lost. Models realistic resonance-frequency
              fabrication scatter.

For each class we track three observables per realisation:
  * the edge/bulk integrated LDOS enhancement,
  * the eta-independent protected-edge spectral weight W0,
  * the protected mode's |Re(E)| displacement off zero (the protection diagnostic).

Every study is paired with a TRIVIAL-phase control (t2 < t1, |nu| = 0) so that
"robust because topological" is separated from "robust because gapped".
"""

import numpy as np

from .hamiltonian import build_ssh_general
from .ldos import enhancement_factor, edge_mode_spectral_weight
from .spectrum import diagonalize, find_edge_modes


# ---------------------------------------------------------------------------
# Disordered Hamiltonian builders
# ---------------------------------------------------------------------------

def add_onsite_disorder(H: np.ndarray, W: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Add uniform on-site disorder U[-W/2, W/2] to all diagonal elements (real shifts)."""
    dim = H.shape[0]
    shifts = W * (rng.random(dim) - 0.5)
    return H + np.diag(shifts.astype(np.complex128))


def make_disordered_ssh(N, t1, t2, gamma, W, kind, rng):
    """
    Build one disordered SSH Hamiltonian of the requested class.

    kind = "bond"    : t1_n, t2_n  <- t + U[-W/2, W/2]   (chiral-preserving)
    kind = "on_site" : eps_i        <-     U[-W/2, W/2]   (chiral-breaking)
    """
    if kind == "bond":
        t1b = t1 + W * (rng.random(N) - 0.5)
        t2b = t2 + W * (rng.random(max(N - 1, 0)) - 0.5)
        return build_ssh_general(N, gamma, t1b, t2b)
    elif kind == "on_site":
        eps = W * (rng.random(2 * N) - 0.5)
        return build_ssh_general(N, gamma, t1, t2, onsite=eps)
    raise ValueError(f"unknown disorder kind: {kind!r}")


def protected_mode_displacement(H) -> float:
    """
    |Re(E)| of the protected edge mode (its displacement off zero). Returns NaN
    if no protected edge mode survives. This is the topological-protection
    diagnostic: ~0 under bond disorder, growing under on-site disorder.
    """
    evals, evecs = diagonalize(H)
    info = find_edge_modes(evals, evecs)
    if info["protected_idx"].size == 0:
        return np.nan
    return float(np.min(np.abs(evals[info["protected_idx"]].real)))


# ---------------------------------------------------------------------------
# Statistical ensembles
# ---------------------------------------------------------------------------

def disorder_ensemble(N, t1, t2, gamma, W, kind,
                      edge_site, bulk_site, gap_omegas, eta,
                      num_realizations, rng) -> dict:
    """
    Monte-Carlo ensemble for one (class, W). Returns per-realisation arrays and
    summary statistics for enhancement, edge spectral weight, and protected-mode
    displacement.
    """
    enh = np.empty(num_realizations)
    w0 = np.empty(num_realizations)
    disp = np.empty(num_realizations)

    for r in range(num_realizations):
        Hd = make_disordered_ssh(N, t1, t2, gamma, W, kind, rng)
        enh[r] = enhancement_factor(Hd, edge_site, bulk_site, gap_omegas, eta)
        w0[r] = edge_mode_spectral_weight(Hd, edge_site)
        disp[r] = protected_mode_displacement(Hd)

    def _stats(a):
        a = a[np.isfinite(a)]
        if a.size == 0:
            return dict(mean=np.nan, std=np.nan, median=np.nan)
        return dict(mean=float(a.mean()), std=float(a.std()), median=float(np.median(a)))

    return {
        "W": W, "kind": kind,
        "enh_all": enh, "w0_all": w0, "disp_all": disp,
        "enh": _stats(enh), "w0": _stats(w0), "disp": _stats(disp),
        "protected_survival": float(np.mean(np.isfinite(disp))),
    }


def run_disorder_study(N, t1, t2, gamma, edge_site, bulk_site,
                       gap_omegas, eta, W_vals, kinds,
                       num_realizations, seed,
                       trivial_t2=None) -> dict:
    """
    Full disorder study: every (kind, W) for the topological point, plus a
    trivial-phase control sweep for each kind. A single seeded RNG stream is
    shared so the whole study is reproducible from `seed`.

    Returns
    -------
    dict with keys:
      'topological' : {kind: {W: ensemble_dict}}
      'trivial'     : {kind: {W: ensemble_dict}}   (if trivial_t2 given)
      'meta'        : parameters used
    """
    rng = np.random.default_rng(seed)
    out = {"topological": {}, "trivial": {},
           "meta": dict(N=N, t1=t1, t2=t2, gamma=gamma, eta=eta,
                        W_vals=list(W_vals), kinds=list(kinds),
                        num_realizations=num_realizations, seed=seed,
                        trivial_t2=trivial_t2)}

    for kind in kinds:
        out["topological"][kind] = {
            W: disorder_ensemble(N, t1, t2, gamma, W, kind,
                                 edge_site, bulk_site, gap_omegas, eta,
                                 num_realizations, rng)
            for W in W_vals
        }

    if trivial_t2 is not None:
        for kind in kinds:
            out["trivial"][kind] = {
                W: disorder_ensemble(N, t1, trivial_t2, gamma, W, kind,
                                     edge_site, bulk_site, gap_omegas, eta,
                                     num_realizations, rng)
                for W in W_vals
            }

    return out
