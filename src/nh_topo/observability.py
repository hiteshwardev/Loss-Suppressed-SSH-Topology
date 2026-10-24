"""
observability.py
================
When does a local probe of finite linewidth resolve the hybridised boundary
doublet of a finite sublattice-lossy SSH chain?

Scope. Equation (2) below is the curvature condition of the standard line shape
of a mode coupled to a lossy partner (cavity QED, coupled resonators). The
distinction between that spectral condition and the exceptional point (EP) is
established for two coupled damped modes; what this module adds is its form for
the SSH edge block and the resulting bound on chain length. A change in the
number of peaks occurs at gamma_m <= sqrt(2) delta0, below the EP, and the edge
LDOS is smooth through the EP, so the peak count does not locate the EP itself.

This module is an extension of the package. It reuses the existing builders
(`hamiltonian.build_ssh`, `disorder.make_disordered_ssh`), the bare splitting
(`symmetry.bare_hybridization`) and the LDOS routines (`ldos.integrated_ldos`),
and adds nothing to the existing modules.

Two-level edge Green function
-----------------------------
Let |L> be the left boundary mode (A sublattice) and |R> the right boundary mode
(B sublattice) of the Hermitian chain. In the basis {|L>, |R>} the doublet block
of H = H0 - i*gamma*P_B is

    h = [[0,      delta0      ],
         [delta0, -i*gamma    ]],

with delta0 the Hermitian half-splitting (`symmetry.bare_hybridization`). The
reduction is exact for every real bond configuration (symmetry.py). The left edge
site overlaps |L> with the exact finite-chain weight W_N (|R> has no amplitude on
an A site), which tends to W0 = 1 - (t1/t2)^2 for N >> xi. With z = omega + i*eta

    G_11(z) = W_N (z + i*gamma) / [ z (z + i*gamma) - delta0^2 ] .           (1)

Writing Gamma = gamma/2 + eta, the edge LDOS rho_1 = -Im G_11 / pi is even in
omega and has a local minimum at omega = 0, hence two resolved peaks, iff

    (gamma + eta)^3 < delta0^2 (2*gamma + 3*eta) .                          (2)

Limits: gamma = 0 gives eta < sqrt(3) delta0 (the Sparrow limit of two equal
Lorentzians); eta = 0 gives gamma < sqrt(2) delta0. The peaks therefore merge at
a loss gamma_m(eta) <= sqrt(2) delta0, below the exceptional point gamma = 2
delta0, and a loss-driven change in the number of peaks occurs only if the
Hermitian doublet is resolved,

    eta < sqrt(3) delta0(N) .                                               (3)

For eta >= sqrt(3) delta0 the margin of Eq. (2) is negative at every gamma.

Peak criterion
--------------
Eq. (2) is a zero-contrast (Sparrow) criterion: it asks only whether omega = 0 is a
local minimum. A measured doublet needs a visible dip. `count_peaks` therefore
counts a maximum only when its prominence exceeds a stated fraction of the
maximum, with the prominence computed so that two peaks of equal height are
referred to the dip between them (`peak_prominences_tie_aware`). The standard
`scipy.signal.peak_prominences` refers each of two exactly equal peaks to the
window edge, which makes any threshold ineffective for the symmetric doublet.
ZERO_CONTRAST (1e-10, a numerical floor) tests Eq. (2) itself; a finite threshold
such as 1e-2 asks for a 1 % dip. `dip_contrast_limit` gives the corresponding
bound eta < c(rel) delta0 for the Hermitian doublet.

Integrated weight
-----------------
The residues of (1) sum to W_N for every gamma, because the doublet subspace is
invariant. This is a consequence of the resolvent sum rule, not a new principle.
The LDOS integrated over a finite window around the doublet is close to, but not
exactly, W_N (Lorentzian tails leave the window). What changes across the EP is
the residue of a single pole: W_N/2 (real part) for each member of the hybridised
doublet, approaching W_N for the protected pole far above threshold.
"""

import numpy as np
import scipy.linalg as la
from scipy.signal import find_peaks

from .hamiltonian import build_ssh, build_ssh_general
from .ldos import integrated_ldos, central_bulk_site
from .symmetry import bare_hybridization
from .topology import analytic_edge_weight, localization_length


# ---------------------------------------------------------------------------
# Closed-form two-level results
# ---------------------------------------------------------------------------

def two_level_edge_green(omega, delta0, gamma, eta, w0=1.0):
    """Edge Green function of the doublet block, Eq. (1). Vectorised in omega."""
    z = np.asarray(omega, float) + 1j * eta
    return w0 * (z + 1j * gamma) / (z * (z + 1j * gamma) - delta0 ** 2)


def two_level_edge_ldos(omega, delta0, gamma, eta, w0=1.0):
    """Edge LDOS of the doublet block, -Im G_11 / pi."""
    return -np.imag(two_level_edge_green(omega, delta0, gamma, eta, w0)) / np.pi


def resolvability_margin(delta0, gamma, eta):
    """
    delta0^2 (2 gamma + 3 eta) - (gamma + eta)^3, Eq. (2).

    Positive: the doublet appears as two peaks. Negative: one peak.
    Exact for the two-level block with zero prominence threshold.
    """
    gamma = np.asarray(gamma, float)
    eta = np.asarray(eta, float)
    return delta0 ** 2 * (2 * gamma + 3 * eta) - (gamma + eta) ** 3


def two_peaks_analytic(delta0, gamma, eta):
    """Boolean form of Eq. (2)."""
    return resolvability_margin(delta0, gamma, eta) > 0


def sparrow_resolved(delta0, gamma, eta):
    """
    Simple Rayleigh-type estimate: two equal Lorentzians of half-width
    Gamma = gamma/2 + eta, centred at +/- s with s = Re sqrt(delta0^2 - gamma^2/4),
    are resolved iff s > Gamma / sqrt(3). It ignores the complex residues that
    the exact Eq. (2) keeps.
    """
    gamma = np.asarray(gamma, float)
    s = np.sqrt(np.clip(delta0 ** 2 - gamma ** 2 / 4.0, 0.0, None))
    return s > (gamma / 2.0 + eta) / np.sqrt(3.0)


def merging_loss(delta0, eta):
    """
    Loss rate gamma_m at which the two peaks merge, the root of Eq. (2) in gamma.

    Returns 0.0 when eta >= sqrt(3) delta0 (never resolved). At eta = 0 the
    value is sqrt(2) delta0.
    """
    if eta >= np.sqrt(3.0) * delta0:
        return 0.0
    lo, hi = 0.0, 2.0 * delta0
    for _ in range(200):                       # bisection; margin(lo) > 0 > margin(hi)
        mid = 0.5 * (lo + hi)
        if resolvability_margin(delta0, mid, eta) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def doublet_resolvable(delta0, eta):
    """
    Eq. (3): the Hermitian (gamma = 0) doublet is resolved, so that loss can
    change the number of peaks. It does not mean the EP itself is detectable.
    """
    return np.asarray(eta, float) < np.sqrt(3.0) * delta0


def max_resolvable_length(t1, t2, eta, N_max=200, rel_contrast=0.0):
    """
    Largest number of unit cells N for which eta < c delta0(N), i.e. the longest
    chain whose Hermitian boundary doublet is resolved at linewidth eta, with the
    exact Hermitian splitting. c = sqrt(3) for zero contrast (Eq. (3)); for a
    required central dip rel_contrast > 0, c = dip_contrast_limit(rel_contrast).
    Returns 0 if no N >= 1 works.
    """
    c = dip_contrast_limit(rel_contrast)
    best = 0
    for N in range(1, N_max + 1):
        if c * bare_hybridization(N, t1, t2) > eta:
            best = N
        else:
            break
    return best


def asymptotic_delta0(N, t1, t2):
    """Large-N splitting delta0 ~ (t2^2 - t1^2)/t2 (t1/t2)^N (semi-infinite modes)."""
    return (t2 ** 2 - t1 ** 2) / t2 * (t1 / t2) ** np.asarray(N, float)


def resolvable_length_asymptotic(t1, t2, eta):
    """N_max from Eq. (3) with the asymptotic splitting: xi ln[sqrt(3)(t2^2-t1^2)/(t2 eta)]."""
    return localization_length(t1, t2) * np.log(np.sqrt(3.0) * (t2 ** 2 - t1 ** 2) / (t2 * eta))


# ---------------------------------------------------------------------------
# Full-chain edge LDOS and peak counting
# ---------------------------------------------------------------------------

def site_residues(H, site):
    """
    Eigenvalues E_n and edge residues c_n = vR[site,n]^2 / (vR^T vR)_n of a
    complex-symmetric H, so that G_ss(z) = sum_n c_n / (z - E_n).
    """
    evals, vR = la.eig(H)
    norms = np.einsum("ij,ij->j", vR, vR)
    return evals, vR[site, :] ** 2 / norms


def ldos_from_residues(evals, residues, omegas, eta):
    """rho(omega) = -(1/pi) Im sum_n c_n / (omega + i eta - E_n), vectorised."""
    z = np.asarray(omegas, float)[:, None] + 1j * eta
    return -np.imag(np.sum(residues[None, :] / (z - evals[None, :]), axis=1)) / np.pi


def edge_ldos_continued_fraction(H, omegas, eta):
    """
    LDOS at site 0 of a tridiagonal complex-symmetric H by the continued fraction

        G_00(z) = 1 / (z - h_0 - t_0^2 / (z - h_1 - t_1^2 / (...))),

    evaluated backwards from the last site, vectorised over omega. Exact, O(N)
    per frequency, and well conditioned at exceptional points, where the
    eigendecomposition is not (the residues diverge there).
    """
    z = np.asarray(omegas, float) + 1j * eta
    d = np.diag(H)
    off = np.diag(H, 1)
    g = 1.0 / (z - d[-1])
    for i in range(d.size - 2, -1, -1):
        g = 1.0 / (z - d[i] - off[i] ** 2 * g)
    return -np.imag(g) / np.pi


def log_symmetric_grid(half_width, smallest=1e-9, n_side=4000):
    """Frequency grid dense near zero: -r[::-1], 0, r with r logarithmic."""
    r = np.logspace(np.log10(smallest), np.log10(half_width), n_side)
    return np.concatenate([-r[::-1], [0.0], r])


ZERO_CONTRAST = 1e-10
"""Numerical floor used as the zero-contrast threshold (tests Eq. (2) itself)."""


def peak_prominences_tie_aware(x, peaks, left_edges, right_edges):
    """
    Prominence of each peak, as in `scipy.signal.peak_prominences`, except that
    the search on each side stops at the first sample that is at least as high
    as the peak (>=, not >). Two peaks of exactly equal height are then each
    referred to the minimum between them, so their prominence is the depth of
    the dip. Plateaus are handled through their edges.
    """
    x = np.asarray(x, float)
    out = np.empty(len(peaks))
    for k, (p, l, r) in enumerate(zip(peaks, left_edges, right_edges)):
        h = x[p]
        left = x[:l]
        hi = np.flatnonzero(left >= h)
        s0 = hi[-1] + 1 if hi.size else 0
        lb = left[s0:].min() if s0 < l else h
        right = x[r + 1:]
        hi = np.flatnonzero(right >= h)
        e0 = hi[0] if hi.size else right.size
        rb = right[:e0].min() if e0 > 0 else h
        out[k] = h - max(lb, rb)
    return out


def peak_contrasts(rho, floor=1e-12):
    """
    Relative tie-aware prominences of the interior local maxima of rho, sorted
    in decreasing order (prominence / max rho). The highest maximum keeps its
    ordinary prominence, so a spectrum with one peak returns one entry close
    to 1 and a symmetric doublet returns (about 1, central dip). Maxima whose ordinary scipy
    prominence is at most `floor` are dropped first (rounding noise on flat
    peak tops); the tie-aware prominence never exceeds the scipy one, so no
    maximum above the floor is lost.
    """
    rho = np.asarray(rho, float)
    q = rho / rho.max()
    peaks, props = find_peaks(q, plateau_size=1, prominence=floor)
    if peaks.size == 0:
        return np.zeros(0)
    pr = peak_prominences_tie_aware(q, peaks, props["left_edges"], props["right_edges"])
    # The highest maximum is a peak by definition: keep its ordinary prominence.
    # Among several maxima of exactly equal height this is the first one; the
    # others keep the dip that separates them from it.
    top = int(np.argmax(q[peaks]))
    pr[top] = props["prominences"][top]
    return np.sort(pr[pr > floor])[::-1]


def doublet_contrast(rho):
    """
    Contrast of the weaker of the two most prominent maxima, as a fraction of
    max rho. For a symmetric doublet this is the central dip
    (rho_peak - rho(0)) / rho_peak. Zero if there are fewer than two maxima.
    """
    pr = peak_contrasts(rho)
    return float(pr[1]) if pr.size >= 2 else 0.0


def count_peaks(rho, rel_prominence=1e-2):
    """
    Number of interior local maxima of rho whose tie-aware prominence exceeds
    rel_prominence times max(rho). A maximum at the grid end is not counted.
    Use ZERO_CONTRAST to test Eq. (2) itself; 1e-2 asks for a 1 % dip.
    """
    return int(np.sum(peak_contrasts(rho) > rel_prominence))


def dip_contrast_limit(rel_contrast):
    """
    Largest eta/delta0 at which the Hermitian doublet (two equal Lorentzians of
    half-width eta at +/- delta0) still shows a central dip larger than
    rel_contrast of the peak. Returns sqrt(3) for rel_contrast <= ZERO_CONTRAST.
    """
    if rel_contrast <= ZERO_CONTRAST:
        return float(np.sqrt(3.0))
    om = np.linspace(0.0, 2.0, 40001)

    def dip(x):
        r = 1.0 / ((om - 1) ** 2 + x ** 2) + 1.0 / ((om + 1) ** 2 + x ** 2)
        return 1.0 - r[0] / r.max()

    lo, hi = 1e-6, np.sqrt(3.0)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if dip(mid) > rel_contrast:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def classify(n_peaks, gamma, delta0):
    """
    1: two resolved peaks (hybridised doublet, W0/2 per pole)
    2: one peak, gamma > 2 delta0 (de-hybridised, W0 in one pole)
    3: one peak, gamma <= 2 delta0 (hybridised but unresolved)
    0: anything else (flagged, should not occur)
    """
    if n_peaks == 2:
        return 1 if gamma <= 2 * delta0 else 0
    if n_peaks > 2:
        return 0
    if n_peaks == 1:
        return 2 if gamma > 2 * delta0 else 3
    return 0


def analytic_class(delta0, gamma, eta):
    """Class predicted by Eq. (2) and the exceptional point."""
    if two_peaks_analytic(delta0, gamma, eta):
        return 1
    return 2 if gamma > 2 * delta0 else 3


def observability_map(N_vals, t1, t2, gammas, etas, gap_half_width=0.15,
                      rel_prominence=ZERO_CONTRAST, n_side=3000):
    """
    Peak-count classification of the full-chain left-edge LDOS on an (N, gamma, eta)
    grid, compared point by point with the two-level prediction.

    The LDOS is sampled on `log_symmetric_grid(gap_half_width, 1e-10, n_side)`, a
    window |omega| <= gap_half_width inside the gap. For every point the doublet
    contrast (`doublet_contrast`) is stored for the full chain and for the
    two-level line shape on the same grid, so that any contrast threshold can be
    applied afterwards without recomputing the LDOS.

    Returns dict with arrays of shape (len(N_vals), len(gammas), len(etas)):
      n_peaks          maxima with relative prominence > rel_prominence
      contrast         full-chain doublet contrast
      contrast_two_level  two-level (Eq. (1)) doublet contrast, same grid
      cls              class from n_peaks (see `classify`)
      cls_analytic     class from Eq. (2) (zero contrast)
      cls_two_level    class from the two-level line shape at rel_prominence
      cls_sparrow      class from the two-Lorentzian estimate
    plus delta0 per N.
    """
    N_vals = np.asarray(N_vals, int)
    gammas = np.asarray(gammas, float)
    etas = np.asarray(etas, float)
    shape = (N_vals.size, gammas.size, etas.size)
    n_peaks = np.zeros(shape, int)
    contrast = np.zeros(shape)
    contrast_2l = np.zeros(shape)
    cls = np.zeros(shape, int)
    cls_a = np.zeros(shape, int)
    cls_2l = np.zeros(shape, int)
    cls_s = np.zeros(shape, int)
    d0 = np.array([bare_hybridization(int(N), t1, t2) for N in N_vals])
    om = log_symmetric_grid(gap_half_width, smallest=1e-10, n_side=n_side)

    for i, N in enumerate(N_vals):
        for j, g in enumerate(gammas):
            H = build_ssh(int(N), t1, t2, g)
            for k, e in enumerate(etas):
                pr = peak_contrasts(edge_ldos_continued_fraction(H, om, e))
                pr2 = peak_contrasts(two_level_edge_ldos(om, d0[i], g, e))
                npk = int(np.sum(pr > rel_prominence))
                n_peaks[i, j, k] = npk
                contrast[i, j, k] = pr[1] if pr.size >= 2 else 0.0
                contrast_2l[i, j, k] = pr2[1] if pr2.size >= 2 else 0.0
                cls[i, j, k] = classify(npk, g, d0[i])
                cls_a[i, j, k] = analytic_class(d0[i], g, e)
                cls_2l[i, j, k] = classify(int(np.sum(pr2 > rel_prominence)), g, d0[i])
                cls_s[i, j, k] = (1 if sparrow_resolved(d0[i], g, e)
                                  else (2 if g > 2 * d0[i] else 3))
    return {"N_vals": N_vals, "gammas": gammas, "etas": etas, "delta0": d0,
            "n_peaks": n_peaks, "contrast": contrast, "contrast_two_level": contrast_2l,
            "cls": cls, "cls_analytic": cls_a, "cls_two_level": cls_2l,
            "cls_sparrow": cls_s, "rel_prominence": rel_prominence}


def classes_at_contrast(omap, rel_contrast):
    """
    Full-chain and two-level classes of an `observability_map` result at another
    contrast threshold, from the stored contrasts (no new LDOS evaluation).
    Points with more than two maxima keep their flag only if the map was built
    at a threshold at or below rel_contrast.
    """
    g = omap["gammas"][None, :, None]
    d0 = omap["delta0"][:, None, None]
    above = (g > 2 * d0) & np.ones(omap["contrast"].shape, bool)

    def cls_of(two):
        return np.where(two, np.where(above, 0, 1), np.where(above, 2, 3))

    return (cls_of(omap["contrast"] > rel_contrast),
            cls_of(omap["contrast_two_level"] > rel_contrast))


# ---------------------------------------------------------------------------
# Peak weights
# ---------------------------------------------------------------------------

def lorentzian_window_weight(E, c, eta, a, b):
    """
    Exact integral over [a, b] of -(1/pi) Im[c / (omega + i eta - E)]:
    -(1/pi) Im{ c [log(b - E') - log(a - E')] },  E' = E - i eta.
    """
    Ep = np.asarray(E) - 1j * eta
    return -np.imag(np.asarray(c) * (np.log(b - Ep) - np.log(a - Ep))) / np.pi


def doublet_pole_weights(H, site=0, gap_half_width=0.15):
    """
    Residues at `site` of the two in-gap poles of H (the boundary doublet).

    Returns dict: E (2,), c (2,) sorted so that index 0 is the pole with the
    smaller decay rate |Im E| (the A-sublattice, protected pole), and the sum.
    """
    ev, c = site_residues(H, site)
    idx = np.where(np.abs(ev.real) < gap_half_width)[0]
    idx = idx[np.argsort(np.abs(ev[idx].imag))][:2]
    return {"E": ev[idx], "c": c[idx], "sum": complex(np.sum(c[idx]))}


def weight_decomposition(N, t1, t2, gammas, eta, gap_half_width=0.15, site=0):
    """
    Per-pole residues and window-integrated edge weights versus gamma.

    Returns dict of arrays over gammas:
      c_protected, c_lossy   complex residues of the two doublet poles
      c_sum                  their sum (equals W0 of the finite chain)
      window_doublet         doublet contribution to the gap-integrated LDOS
      window_total           all poles, gap-integrated LDOS at `site`
      window_protected, window_lossy  per-pole window weights
    """
    out = {k: [] for k in ("c_protected", "c_lossy", "c_sum", "window_doublet",
                           "window_total", "window_protected", "window_lossy")}
    a, b = -gap_half_width, gap_half_width
    for g in np.asarray(gammas, float):
        H = build_ssh(N, t1, t2, g)
        ev, c = site_residues(H, site)
        idx = np.where(np.abs(ev.real) < gap_half_width)[0]
        idx = idx[np.argsort(np.abs(ev[idx].imag))][:2]
        d = {"E": ev[idx], "c": c[idx], "sum": complex(np.sum(c[idx]))}
        wp = float(lorentzian_window_weight(d["E"][0], d["c"][0], eta, a, b))
        wl = float(lorentzian_window_weight(d["E"][1], d["c"][1], eta, a, b))
        out["c_protected"].append(d["c"][0])
        out["c_lossy"].append(d["c"][1])
        out["c_sum"].append(d["sum"])
        out["window_protected"].append(wp)
        out["window_lossy"].append(wl)
        out["window_doublet"].append(wp + wl)
        out["window_total"].append(float(np.sum(lorentzian_window_weight(ev, c, eta, a, b))))
    out = {k: np.asarray(v) for k, v in out.items()}
    out["gammas"] = np.asarray(gammas, float)
    out["W0_analytic"] = analytic_edge_weight(t1, t2)
    return out


# ---------------------------------------------------------------------------
# Enhancement decomposition
# ---------------------------------------------------------------------------

def enhancement_decomposition(N, t1, t2, gammas, gap_omegas, eta, edge_site=0):
    """
    Split the gap-integrated enhancement E = I_edge / I_bulk into numerator and
    denominator, with exactly the quadrature of `ldos.enhancement_factor`.

    Returns dict: gammas, I_edge, I_bulk, enhancement, and the share of the drop
    ln E(gamma_0)/E(gamma) carried by the denominator, for every gamma.
    """
    bulk = central_bulk_site(N)
    ie, ib = [], []
    for g in np.asarray(gammas, float):
        H = build_ssh(N, t1, t2, g)
        ie.append(integrated_ldos(H, edge_site, gap_omegas, eta))
        ib.append(integrated_ldos(H, bulk, gap_omegas, eta))
    ie, ib = np.asarray(ie), np.asarray(ib)
    enh = ie / ib
    drop = np.log(enh[0] / enh)
    with np.errstate(invalid="ignore", divide="ignore"):
        share_bulk = np.where(drop > 0, np.log(ib / ib[0]) / drop, np.nan)
    return {"gammas": np.asarray(gammas, float), "I_edge": ie, "I_bulk": ib,
            "enhancement": enh, "share_bulk": share_bulk}


# ---------------------------------------------------------------------------
# Disorder-resolved resolvability
# ---------------------------------------------------------------------------

def disorder_resolvability(N, t1, t2, W, eta, num_realizations, seed,
                           gap_half_width=None, rel_prominence=ZERO_CONTRAST, n_side=3000):
    """
    Fraction of bond-disordered chains whose Hermitian (gamma = 0) boundary
    doublet appears as two peaks at the left edge for broadening eta, i.e. the
    fraction in which loss can change the number of peaks. Bonds follow
    `disorder.make_disordered_ssh` (U[-W/2, W/2]); a single seeded stream as in
    the rest of the package.

    A chain counts as resolved when its doublet contrast exceeds rel_prominence.
    The prediction is eta < c delta0 with the realization's own delta0 and
    c = dip_contrast_limit(rel_prominence) (sqrt(3) at zero contrast); bond
    disorder keeps the two-level block exact, so this is a direct test.

    Returns dict: delta0 and contrast per realisation, predicted and numerical
    resolvability, the two fractions and their agreement.
    """
    rng = np.random.default_rng(seed)
    if gap_half_width is None:
        gap_half_width = 0.5 * abs(t2 - t1)
    om = log_symmetric_grid(gap_half_width, smallest=1e-12 * max(abs(t1), abs(t2)),
                            n_side=n_side)
    c_lim = dip_contrast_limit(rel_prominence)
    d0, con = [], []
    for _ in range(num_realizations):
        b1 = t1 + W * (rng.random(N) - 0.5)
        b2 = t2 + W * (rng.random(max(N - 1, 0)) - 0.5)
        H0 = build_ssh_general(N, 0.0, b1, b2)
        d0.append(float(np.min(np.abs(np.linalg.eigvalsh(H0.real)))))
        con.append(doublet_contrast(edge_ldos_continued_fraction(H0, om, eta)))
    d0, con = np.asarray(d0), np.asarray(con)
    ana = eta < c_lim * d0
    num = con > rel_prominence
    return {"N": N, "W": W, "eta": eta, "delta0": d0, "contrast": con,
            "rel_prominence": rel_prominence,
            "resolved_analytic": ana, "resolved_numeric": num,
            "fraction_analytic": float(ana.mean()),
            "fraction_numeric": float(num.mean()),
            "agreement": float(np.mean(ana == num))}


# ---------------------------------------------------------------------------
# Resolvability under on-site detuning
# ---------------------------------------------------------------------------

def detuning_resolvability(N, t1, t2, W, eta, num_realizations, seed,
                           gamma=0.0, n_omega=40001, rel_prominence=1e-2):
    """
    Fraction of chains with uniform on-site detuning U[-W/2, W/2] (the
    convention of `disorder.make_disordered_ssh`; W is the full width) whose
    left-edge LDOS shows two peaks at broadening eta.

    On-site detuning breaks Eq. (1) of symmetry.py, so the exact two-level
    reduction no longer holds; this function therefore counts peaks in the full
    edge LDOS (continued fraction, bulk poles included) on a uniform grid over
    |omega| <= |t2 - t1| / 2, half of the gap. The window is adequate while the
    detuning shifts of the boundary modes stay well inside it (W up to about
    |t2 - t1| / 2). A detuning between the two ends that is large compared with
    delta0 localizes the doublet on separate ends, and the left edge then sees
    one peak.

    A chain counts as resolved when its doublet contrast (`doublet_contrast`)
    exceeds rel_prominence; the zero-contrast fraction is returned as well.

    Returns dict: W, eta, fraction (at rel_prominence), fraction_zero_contrast,
    and the per-realization contrasts and peak counts.
    """
    rng = np.random.default_rng(seed)
    hw = 0.5 * abs(t2 - t1)
    om = np.linspace(-hw, hw, n_omega)
    counts = np.empty(num_realizations, int)
    con = np.empty(num_realizations)
    for k in range(num_realizations):
        eps = W * (rng.random(2 * N) - 0.5)
        H = build_ssh_general(N, gamma, t1, t2, onsite=eps)
        pr = peak_contrasts(edge_ldos_continued_fraction(H, om, eta))
        counts[k] = int(np.sum(pr > rel_prominence))
        con[k] = pr[1] if pr.size >= 2 else 0.0
    return {"N": N, "W": W, "eta": eta, "gamma": gamma, "rel_prominence": rel_prominence,
            "fraction": float(np.mean(con > rel_prominence)),
            "fraction_zero_contrast": float(np.mean(con > ZERO_CONTRAST)),
            "contrast": con, "counts": counts}
