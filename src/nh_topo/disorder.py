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

from .hamiltonian import build_ssh_general, build_ssh
from .ldos import enhancement_factor, edge_mode_spectral_weight
from .spectrum import (diagonalize, find_edge_modes, track_mode_by_overlap,
                       clean_protected_mode)


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


def protected_mode_displacement(H, reference=None, gap_half_width=None) -> float:
    """
    |Re(E)| of the protected edge mode -- its displacement off the imaginary axis.

    This is the topological-protection diagnostic: exactly 0 under
    CS-dagger-preserving (bond) disorder, growing under CS-dagger-breaking
    (on-site) disorder.

    MODE IDENTIFICATION (revised)
    -----------------------------
    If `reference` (the clean protected-mode wavefunction) is supplied, the mode
    is tracked by wavefunction OVERLAP -- adiabatic continuation. This uses no
    energy information and is therefore not circular, and it is stable at
    disorder strengths where threshold-based detection fails.

    The previous implementation selected the mode with a fixed boundary-weight
    threshold and returned min|Re(E)| over the resulting set. At W = 0.6 t1 that
    admitted band-edge bulk states in 3/200 realisations and reported their
    |Re(E)| ~ 1.2 t1 as the displacement, inflating the ensemble mean to
    0.023 t1. The exact CS-dagger symmetry (symmetry.py) forbids any such
    displacement under bond disorder, so that value was an artefact.

    Falls back to the (now more robust) localization-based detector when no
    reference is given. Returns NaN if no protected mode can be identified.
    """
    evals, evecs = diagonalize(H)

    if reference is not None and np.any(reference):
        idx = track_mode_by_overlap(evecs, reference)
        return float(abs(evals[idx].real))

    info = find_edge_modes(evals, evecs, gap_half_width=gap_half_width)
    if info["protected_idx"].size == 0:
        return np.nan
    idx = info["protected_idx"][np.argmin(np.abs(evals[info["protected_idx"]].imag))]
    return float(abs(evals[idx].real))


# ---------------------------------------------------------------------------
# Statistical ensembles
# ---------------------------------------------------------------------------

def disorder_ensemble(N, t1, t2, gamma, W, kind,
                      edge_site, bulk_site, gap_omegas, eta,
                      num_realizations, rng, reference=None) -> dict:
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
        disp[r] = protected_mode_displacement(Hd, reference=reference,
                                              gap_half_width=abs(t2 - t1))

    def _stats(a):
        a = a[np.isfinite(a)]
        if a.size == 0:
            return dict(mean=np.nan, std=np.nan, median=np.nan,
                        sem=np.nan, ci95=np.nan, n=0)
        n = int(a.size)
        sd = float(a.std(ddof=1)) if n > 1 else 0.0
        sem = sd / np.sqrt(n) if n > 1 else 0.0
        return dict(mean=float(a.mean()), std=sd, median=float(np.median(a)),
                    sem=float(sem), ci95=float(1.96 * sem), n=n)

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

    ref_topo = clean_protected_mode(N, t1, t2, gamma)

    for kind in kinds:
        out["topological"][kind] = {
            W: disorder_ensemble(N, t1, t2, gamma, W, kind,
                                 edge_site, bulk_site, gap_omegas, eta,
                                 num_realizations, rng, reference=ref_topo)
            for W in W_vals
        }

    if trivial_t2 is not None:
        for kind in kinds:
            out["trivial"][kind] = {
                W: disorder_ensemble(N, t1, trivial_t2, gamma, W, kind,
                                     edge_site, bulk_site, gap_omegas, eta,
                                     num_realizations, rng, reference=None)
                for W in W_vals
            }

    return out


# ---------------------------------------------------------------------------
# Loss-suppressed finite-size hybridisation
# ---------------------------------------------------------------------------

def pinned_mode_count(H, tol: float = 1e-8) -> int:
    """Number of eigenvalues with |Re(E)| < tol (modes on the imaginary axis)."""
    from .spectrum import diagonalize
    evals, _ = diagonalize(H)
    return int(np.sum(np.abs(evals.real) < tol))


def edge_doublet_splitting(H) -> float:
    """
    Splitting of the near-zero boundary doublet, min |Re(E)| over the spectrum.

    In a finite HERMITIAN SSH chain the two boundary modes are degenerate at
    E = 0, hybridise, and split to +/- delta with delta exponentially small in N
    but strictly nonzero. Under sublattice loss the two modes remain at
    Re(E) = 0 but acquire DIFFERENT decay rates, so CS-dagger (see symmetry.py)
    gives neither a partner at equal Im(E) to pair with, and the splitting
    channel is closed.

    Returns min|Re(E)|: ~exp(-N/xi) for the Hermitian chain, ~machine precision
    for the lossy chain.
    """
    from .spectrum import diagonalize
    evals, _ = diagonalize(H)
    return float(np.min(np.abs(evals.real)))


def hybridization_study(N, t1, t2, gamma_vals, W_vals,
                        num_realizations, seed, kind: str = "bond") -> dict:
    """
    Compare finite-size hybridisation with and without loss, under IDENTICAL
    disorder ensembles.

    For each gamma the same seed is used, so the two columns see exactly the same
    disorder realisations and the comparison is like-for-like. The diagnostic is
    `edge_doublet_splitting` (min|Re E|), which requires no mode identification
    at all and is therefore free of detector artefacts.

    Returns
    -------
    dict:
      'gamma_vals', 'W_vals',
      'splitting'  : ndarray (n_gamma, n_W) mean min|Re E|
      'sem'        : ndarray (n_gamma, n_W) standard error of that mean
      'meta'
    """
    gamma_vals = np.asarray(gamma_vals, float)
    W_vals = np.asarray(W_vals, float)
    mean = np.zeros((gamma_vals.size, W_vals.size))
    sem = np.zeros_like(mean)

    for i, g in enumerate(gamma_vals):
        for j, W in enumerate(W_vals):
            rng = np.random.default_rng(seed)      # identical ensemble per gamma
            vals = np.empty(num_realizations)
            for r in range(num_realizations):
                Hd = make_disordered_ssh(N, t1, t2, g, W, kind, rng)
                vals[r] = edge_doublet_splitting(Hd)
            mean[i, j] = vals.mean()
            sem[i, j] = vals.std(ddof=1) / np.sqrt(num_realizations) if num_realizations > 1 else 0.0

    return {"gamma_vals": gamma_vals, "W_vals": W_vals,
            "splitting": mean, "sem": sem,
            "meta": dict(N=N, t1=t1, t2=t2, kind=kind, seed=seed,
                         num_realizations=num_realizations)}


def hybridization_vs_size(N_vals, t1, t2, gamma_vals, W,
                          num_realizations, seed, kind: str = "bond") -> dict:
    """
    Size scaling of the doublet splitting at fixed disorder strength.

    The Hermitian splitting should fall exponentially with N (delta ~ exp(-N/xi))
    while the lossy chain stays at machine precision for every N -- the
    sharpest demonstration that the effect is a symmetry protection and not
    merely a large-N asymptote.

    Returns dict: 'N_vals', 'gamma_vals', 'splitting' (n_gamma, n_N), 'sem'.
    """
    N_vals = np.asarray(N_vals, int)
    gamma_vals = np.asarray(gamma_vals, float)
    mean = np.zeros((gamma_vals.size, N_vals.size))
    sem = np.zeros_like(mean)

    for i, g in enumerate(gamma_vals):
        for j, N in enumerate(N_vals):
            rng = np.random.default_rng(seed)
            vals = np.empty(num_realizations)
            for r in range(num_realizations):
                Hd = make_disordered_ssh(int(N), t1, t2, g, W, kind, rng)
                vals[r] = edge_doublet_splitting(Hd)
            mean[i, j] = vals.mean()
            sem[i, j] = vals.std(ddof=1) / np.sqrt(num_realizations) if num_realizations > 1 else 0.0

    return {"N_vals": N_vals, "gamma_vals": gamma_vals,
            "splitting": mean, "sem": sem,
            "meta": dict(t1=t1, t2=t2, W=W, kind=kind, seed=seed,
                         num_realizations=num_realizations)}


def symmetry_residual_study(N, t1, t2, gamma, W_vals,
                            num_realizations, seed) -> dict:
    """
    CS-dagger residual as a function of disorder strength for both classes.

    Quantifies the statement that bond disorder is CS-dagger preserving (residual
    ~0 at every strength) while on-site disorder breaks it in proportion to W.

    Returns dict: 'W_vals', 'bond', 'on_site' (mean residuals), plus 'bond_max'.
    """
    from .symmetry import cs_dagger_residual

    W_vals = np.asarray(W_vals, float)
    out = {"W_vals": W_vals}
    for kind in ("bond", "on_site"):
        means, maxes = [], []
        for W in W_vals:
            rng = np.random.default_rng(seed)
            vals = np.empty(num_realizations)
            for r in range(num_realizations):
                Hd = make_disordered_ssh(N, t1, t2, gamma, W, kind, rng)
                vals[r] = cs_dagger_residual(Hd, gamma)
            means.append(float(vals.mean()))
            maxes.append(float(vals.max()))
        out[kind] = np.array(means)
        out[kind + "_max"] = np.array(maxes)
    out["meta"] = dict(N=N, t1=t1, t2=t2, gamma=gamma, seed=seed,
                       num_realizations=num_realizations)
    return out


def ensemble_convergence(N, t1, t2, gamma, W, kind, edge_site, bulk_site,
                         gap_omegas, eta, max_realizations, seed,
                         checkpoints=None) -> dict:
    """
    Running mean of the enhancement as realisations accumulate.

    Justifies the ensemble size: the reported mean must be stable well before
    the number of realisations actually used. Returns the running mean and its
    95% confidence interval at each checkpoint.
    """
    checkpoints = checkpoints or [5, 10, 25, 50, 100, 200, 400]
    checkpoints = [c for c in checkpoints if c <= max_realizations]
    rng = np.random.default_rng(seed)
    vals = np.empty(max_realizations)
    for r in range(max_realizations):
        Hd = make_disordered_ssh(N, t1, t2, gamma, W, kind, rng)
        vals[r] = enhancement_factor(Hd, edge_site, bulk_site, gap_omegas, eta)

    means, ci = [], []
    for n in checkpoints:
        a = vals[:n]
        means.append(float(a.mean()))
        sd = float(a.std(ddof=1)) if n > 1 else 0.0
        ci.append(float(1.96 * sd / np.sqrt(n)) if n > 1 else 0.0)
    return {"checkpoints": checkpoints, "mean": means, "ci95": ci,
            "final_mean": float(vals.mean()),
            "final_ci95": float(1.96 * vals.std(ddof=1) / np.sqrt(max_realizations))}


def delta0_size_scaling(N_vals, t1, t2) -> dict:
    """
    Bare hybridisation splitting delta_0(N), with an exponential fit.

    Tests the claim delta_0 ~ exp(-N/xi) quantitatively: the fitted decay length
    is compared against the analytic xi = 1/ln(t2/t1).
    """
    from .symmetry import bare_hybridization
    from .topology import localization_length

    N_vals = np.asarray(N_vals, int)
    d0 = np.array([bare_hybridization(int(n), t1, t2) for n in N_vals])
    slope = np.polyfit(N_vals.astype(float), np.log(d0), 1)[0]
    xi_fit = -1.0 / slope
    xi_ana = localization_length(t1, t2)
    return {"N_vals": N_vals, "delta0": d0, "xi_fit": float(xi_fit),
            "xi_analytic": float(xi_ana),
            "rel_error": float(abs(xi_fit - xi_ana) / xi_ana)}


def fraction_below_edge_ep(N, t1, t2, gamma, W, num_realizations, seed) -> dict:
    """
    Fraction of disordered realisations that fall below the edge exceptional
    point, where the pinning is NOT expected to hold.

    The protection condition gamma > 2*delta_0 involves the DISORDERED delta_0,
    which fluctuates upward under bond disorder. In short chains this can push
    individual realisations back into the hybridised regime; this function
    quantifies how often, and is what explains the residual visible at N = 5.
    """
    rng = np.random.default_rng(seed)
    n_below, splits = 0, []
    for _ in range(num_realizations):
        b1 = t1 + W * (rng.random(N) - 0.5)
        b2 = t2 + W * (rng.random(max(N - 1, 0)) - 0.5)
        H0 = build_ssh_general(N, 0.0, b1, b2)          # Hermitian counterpart
        d0 = float(np.min(np.abs(np.linalg.eigvals(H0).real)))
        Hg = build_ssh_general(N, gamma, b1, b2)
        s = float(np.min(np.abs(np.linalg.eigvals(Hg).real)))
        splits.append(s)
        if gamma < 2 * d0:
            n_below += 1
    splits = np.array(splits)
    return {"N": N, "W": W, "gamma": gamma,
            "fraction_below_ep": n_below / num_realizations,
            "mean_splitting": float(splits.mean()),
            "max_splitting": float(splits.max()),
            "frac_pinned": float(np.mean(splits < 1e-12))}


def trivial_control_anomalies(N, t1, t2_trivial, gamma, W_vals, edge_site,
                              bulk_site, gap_omegas, eta, num_realizations,
                              seed, threshold: float = 2.0) -> dict:
    """
    Frequency with which the trivial-phase control develops a spurious boundary
    resonance under strong disorder.

    Strong on-site disorder can localise a state near the boundary by accident,
    with no topological origin. Such a realisation mimics the signature the
    control is meant to exclude, so the rate at which it happens bounds how far
    the control can be trusted. A realisation counts as anomalous when its
    edge-to-bulk ratio exceeds `threshold`, well above the value near unity that
    the trivial phase produces in the absence of disorder.
    """
    out = {"W_vals": list(W_vals), "threshold": threshold}
    for kind in ("bond", "on_site"):
        rates, worst = [], []
        for W in W_vals:
            rng = np.random.default_rng(seed)
            vals = np.empty(num_realizations)
            for r in range(num_realizations):
                Hd = make_disordered_ssh(N, t1, t2_trivial, gamma, W, kind, rng)
                vals[r] = enhancement_factor(Hd, edge_site, bulk_site,
                                             gap_omegas, eta)
            rates.append(float(np.mean(vals > threshold)))
            worst.append(float(vals.max()))
        out[kind] = {"anomaly_rate": rates, "max_enhancement": worst}
    return out
