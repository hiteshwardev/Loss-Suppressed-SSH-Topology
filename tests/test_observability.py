"""
Tests for nh_topo.observability (extension: observability of the edge
exceptional point). Known limits: gamma = 0, the Hermitian splitting, the
analytic W0, and gamma >> 2 delta0.
"""

import numpy as np
import pytest

import nh_topo as nt
from nh_topo import observability as ob

T1, T2 = 0.8, 1.2


@pytest.fixture(scope="module")
def chain10():
    return 10, nt.bare_hybridization(10, T1, T2)


# --- closed-form two-level results -----------------------------------------

def test_two_level_gamma0_is_two_lorentzians(chain10):
    _, d0 = chain10
    om = np.linspace(-0.05, 0.05, 2001)
    eta = 0.1 * d0
    rho = ob.two_level_edge_ldos(om, d0, 0.0, eta)
    lor = lambda x0: eta / np.pi / ((om - x0) ** 2 + eta ** 2)
    assert np.max(np.abs(rho - 0.5 * (lor(d0) + lor(-d0)))) < 1e-10 * rho.max()


def test_two_level_large_gamma_single_narrow_pole(chain10):
    _, d0 = chain10
    g, eta = 1000 * d0, 1e-6
    om = np.linspace(-1e-4, 1e-4, 4001)
    rho = ob.two_level_edge_ldos(om, d0, g, eta)
    width = eta + d0 ** 2 / g                       # pole at -i delta0^2/gamma
    ref = width / np.pi / (om ** 2 + width ** 2)
    assert np.max(np.abs(rho - ref)) / ref.max() < 2e-3


def test_two_level_matches_full_chain_doublet(chain10):
    N, d0 = chain10
    om = ob.log_symmetric_grid(0.15, 1e-10, 1000)
    for g in (0.0, 0.7 * d0, 5 * d0):
        ev, c = ob.site_residues(nt.build_ssh(N, T1, T2, g), 0)
        sel = np.abs(ev.real) < 0.15
        dbl = ob.ldos_from_residues(ev[sel], c[sel], om, 0.2 * d0)
        two = ob.two_level_edge_ldos(om, d0, g, 0.2 * d0, w0=np.sum(c[sel]).real)
        assert np.max(np.abs(dbl - two)) < 1e-9 * two.max()


def test_margin_limits(chain10):
    _, d0 = chain10
    # gamma = 0: Sparrow limit eta = sqrt(3) delta0
    assert ob.resolvability_margin(d0, 0.0, 0.99 * np.sqrt(3) * d0) > 0
    assert ob.resolvability_margin(d0, 0.0, 1.01 * np.sqrt(3) * d0) < 0
    # eta = 0: merging at gamma = sqrt(2) delta0, below the EP at 2 delta0
    assert ob.merging_loss(d0, 0.0) == pytest.approx(np.sqrt(2) * d0, rel=1e-10)
    assert ob.merging_loss(d0, 2 * d0) == 0.0
    # merging always precedes the exceptional point
    for e in np.linspace(0, 1.7, 9) * d0:
        assert ob.merging_loss(d0, e) < 2 * d0


def test_sparrow_equals_exact_at_gamma0(chain10):
    _, d0 = chain10
    for e in np.linspace(0.05, 3, 40) * d0:
        assert ob.sparrow_resolved(d0, 0.0, e) == ob.two_peaks_analytic(d0, 0.0, e)


def test_hermitian_splitting_peak_positions(chain10):
    N, d0 = chain10
    om = np.linspace(-3 * d0, 3 * d0, 60001)
    rho = ob.edge_ldos_continued_fraction(nt.build_ssh(N, T1, T2, 0.0), om, 1e-3 * d0)
    peak = om[np.argmax(rho * (om > 0))]
    assert peak == pytest.approx(d0, rel=1e-3)


# --- full-chain numerics ----------------------------------------------------

def test_continued_fraction_matches_linear_solve():
    H = nt.build_ssh(12, T1, T2, 0.07)
    om = np.linspace(-0.2, 0.2, 9)
    cf = ob.edge_ldos_continued_fraction(H, om, 0.01)
    ref = np.array([nt.ldos_at(H, 0, w, 0.01) for w in om])
    assert np.max(np.abs(cf - ref)) < 1e-12


def test_continued_fraction_well_conditioned_at_ep(chain10):
    N, d0 = chain10
    H = nt.build_ssh(N, T1, T2, 2 * d0)              # edge exceptional point
    om = np.linspace(-0.05, 0.05, 7)
    cf = ob.edge_ldos_continued_fraction(H, om, 1e-3)
    ref = np.array([nt.ldos_at(H, 0, w, 1e-3) for w in om])
    assert np.max(np.abs(cf - ref)) < 1e-10


def test_count_peaks_ignores_flat_top_noise():
    om = ob.log_symmetric_grid(0.15, 1e-10, 2000)
    rho = 1.0 / (om ** 2 + 0.01 ** 2) * (1 + 1e-15 * np.sin(1e9 * om))
    assert ob.count_peaks(rho) == 1


def test_classification_limits(chain10):
    N, d0 = chain10
    m = ob.observability_map([N], T1, T2, [0.0, 50 * d0], [0.05 * d0, 10 * d0],
                             n_side=1500)
    assert m["cls"][0, 0, 0] == 1          # gamma = 0, eta << delta0: two peaks
    assert m["cls"][0, 0, 1] == 3          # gamma = 0, eta >> delta0: unresolved
    assert m["cls"][0, 1, 0] == 2          # gamma >> 2 delta0: one de-hybridised peak
    assert np.array_equal(m["cls"], m["cls_analytic"])


def test_map_agrees_with_analytic_on_a_coarse_grid():
    gs = np.logspace(-4, np.log10(0.3), 8)
    es = np.logspace(-4, -1, 8)
    m = ob.observability_map([6, 12], T1, T2, gs, es, n_side=1500)
    assert np.array_equal(m["cls"], m["cls_analytic"])
    assert not np.any(m["cls"] == 0)


# --- weights ----------------------------------------------------------------

def test_residue_sum_is_W0_on_both_sides_of_ep(chain10):
    N, d0 = chain10
    wd = ob.weight_decomposition(N, T1, T2, np.array([0.0, 0.5, 1.9, 2.1, 10, 30]) * d0, 0.01)
    w0_N = wd["c_sum"][0].real                      # Hermitian doublet, edge site
    assert w0_N == pytest.approx(nt.analytic_edge_weight(T1, T2), rel=5e-3)
    assert np.allclose(wd["c_sum"], w0_N, atol=1e-10)
    # below the EP each pole carries W0/2 in its real part
    assert np.allclose(wd["c_protected"][:3].real, w0_N / 2, atol=1e-10)
    # far above, the protected pole carries W0 and the lossy pole almost nothing
    assert wd["c_protected"][-1].real == pytest.approx(w0_N, rel=2e-3)
    assert abs(wd["c_lossy"][-1]) < 2e-3


def test_semi_infinite_weight_limit():
    w = nt.edge_mode_spectral_weight(nt.build_ssh(40, T1, T2, 0.3), 0)
    assert w == pytest.approx(nt.analytic_edge_weight(T1, T2), abs=1e-8)


def test_window_weight_constant_across_ep(chain10):
    N, d0 = chain10
    wd = ob.weight_decomposition(N, T1, T2, np.array([0.0, 1.0, 3.0, 30.0]) * d0, 0.05)
    assert np.ptp(wd["window_doublet"]) < 1e-3


def test_lorentzian_window_weight_exact():
    E, c, eta = 0.01 - 0.002j, 0.3 + 0.05j, 0.004
    om = np.linspace(-0.15, 0.15, 400001)
    num = np.trapezoid(-np.imag(c / (om + 1j * eta - E)) / np.pi, om)
    assert ob.lorentzian_window_weight(E, c, eta, -0.15, 0.15) == pytest.approx(num, abs=1e-8)


# --- enhancement decomposition ----------------------------------------------

def test_enhancement_decomposition_reproduces_existing_numbers():
    gap = np.linspace(-0.15, 0.15, 61)
    ed = ob.enhancement_decomposition(20, T1, T2, [0.0, 0.3], gap, 0.05)
    H = nt.build_ssh(20, T1, T2, 0.3)
    ref = nt.enhancement_factor(H, 0, nt.central_bulk_site(20), gap, 0.05)
    assert ed["enhancement"][1] == pytest.approx(ref, rel=1e-12)
    assert ed["enhancement"][0] == pytest.approx(68.507, rel=1e-4)
    assert ed["I_bulk"][1] > 5 * ed["I_bulk"][0]


# --- resolvable length and disorder -----------------------------------------

def test_max_resolvable_length_consistent_with_asymptotic():
    for eta in (1e-4, 1e-3, 1e-2):
        n = ob.max_resolvable_length(T1, T2, eta)
        assert abs(n - ob.resolvable_length_asymptotic(T1, T2, eta)) < 1.0
        assert np.sqrt(3) * nt.bare_hybridization(n, T1, T2) > eta
        assert np.sqrt(3) * nt.bare_hybridization(n + 1, T1, T2) <= eta


def test_disorder_resolvability_clean_limit():
    d0 = nt.bare_hybridization(6, T1, T2)
    vis = ob.disorder_resolvability(6, T1, T2, 0.0, 0.5 * d0, 5, 1)
    hid = ob.disorder_resolvability(6, T1, T2, 0.0, 3.0 * d0, 5, 1)
    assert vis["fraction_numeric"] == 1.0 and hid["fraction_numeric"] == 0.0
    assert vis["agreement"] == 1.0 and hid["agreement"] == 1.0


def test_disorder_resolvability_reproducible():
    a = ob.disorder_resolvability(4, T1, T2, 0.2, 0.01, 20, 7)
    b = ob.disorder_resolvability(4, T1, T2, 0.2, 0.01, 20, 7)
    assert np.array_equal(a["delta0"], b["delta0"])


def test_single_peak_at_all_loss_when_hermitian_doublet_unresolved():
    g = np.logspace(-6, 4, 2000)
    for e in (np.sqrt(3), 2.0, 10.0):
        assert not np.any(ob.two_peaks_analytic(1.0, g, e))


def test_merging_loss_never_exceeds_sqrt2_delta0():
    gm = np.array([ob.merging_loss(1.0, e) for e in np.linspace(0, 1.73, 200)])
    assert gm.max() <= np.sqrt(2) + 1e-12
    assert np.all(np.diff(gm) <= 1e-12)


def test_edge_ldos_smooth_through_ep():
    gg = np.linspace(1.9, 2.1, 201)
    r0 = np.array([ob.two_level_edge_ldos(0.0, 1.0, x, 0.05) for x in gg]).ravel()
    d2 = np.abs(np.diff(r0, 2))
    assert d2.max() < 2 * d2.min()          # no kink or cusp at gamma = 2 delta0


def test_two_level_ldos_has_at_most_two_maxima():
    # d rho/d omega = omega * Q(omega^2) with Q quadratic, leading and linear
    # coefficients negative, constant term proportional to the margin of Eq. (2).
    # Descartes' rule then gives one positive root iff margin > 0. Check on a
    # random sample: the peak count is 2 iff margin > 0, and never exceeds 2.
    rng = np.random.default_rng(2026)
    for _ in range(300):
        g, e = 10 ** rng.uniform(-3, 1, 2)
        om = np.linspace(-6, 6, 120001)
        n = ob.count_peaks(ob.two_level_edge_ldos(om, 1.0, g, e), rel_prominence=0.0)
        assert n <= 2
        m = ob.resolvability_margin(1.0, g, e)
        if abs(m) > 1e-3:                      # away from the boundary itself
            assert (n == 2) == (m > 0)


def test_detuning_resolvability_clean_limit_and_suppression():
    t1, t2 = 14.8, 37.1
    clean = ob.detuning_resolvability(4, t1, t2, 0.0, 1.0, 5, 3, n_omega=8001)
    assert clean["fraction"] == 1.0                 # N = 4, eta = 1 MHz is resolved
    strong = ob.detuning_resolvability(4, t1, t2, 10.0, 1.0, 40, 3, n_omega=8001)
    assert strong["fraction"] < 0.3                 # scatter >> delta0 hides the doublet
    again = ob.detuning_resolvability(4, t1, t2, 10.0, 1.0, 40, 3, n_omega=8001)
    assert np.array_equal(strong["counts"], again["counts"])


def test_peak_count_converged_in_frequency_grid():
    gs = np.logspace(-4, np.log10(0.3), 6)
    es = np.logspace(-4, -1, 6)
    a = ob.observability_map([8, 14], T1, T2, gs, es, n_side=750)
    b = ob.observability_map([8, 14], T1, T2, gs, es, n_side=6000)
    assert np.array_equal(a["n_peaks"] >= 2, b["n_peaks"] >= 2)


# --- added in the final audit ------------------------------------------------

def test_merging_loss_decreases_with_broadening_and_vanishes_at_sqrt3():
    etas = np.linspace(0.0, 1.7, 35)
    gm = np.array([ob.merging_loss(1.0, e) for e in etas])
    assert abs(gm[0] - np.sqrt(2.0)) < 1e-12
    assert np.all(np.diff(gm) < 0)
    for e in (np.sqrt(3.0), 2.0, 10.0):
        g = np.logspace(-4, 3, 400)
        assert np.all(ob.resolvability_margin(1.0, g, e) <= 0)


def test_bulk_poles_stay_outside_the_gap_window():
    # |Re E| of every bulk pole >= sqrt((t2 - t1)^2 - gamma^2/4) > 0.15 for gamma <= 0.5
    for N in (6, 10, 20, 30):
        for g in (0.0, 0.3, 0.5):
            ev = np.linalg.eigvals(nt.build_ssh(N, T1, T2, g))
            bulk = np.sort(np.abs(ev.real))[2:]
            assert bulk.min() >= np.sqrt((T2 - T1) ** 2 - g ** 2 / 4) - 1e-12
            assert bulk.min() > 0.15


def _figure_inputs():
    rng = np.random.default_rng(0)
    W = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 1.0, 1.5])
    Nv = np.array([5, 8, 10, 15, 20, 30])
    x = np.linspace(-2.2, 2.2, 201)
    spectra = [1.0 / ((x - s) ** 2 + 0.05 + 0.1 * k) + 1.0 / ((x + s) ** 2 + 0.05 + 0.1 * k)
               for k, s in enumerate((1.0, 0.7, 0.0, 0.0, 0.0))]
    g = 10 ** rng.uniform(-3, 4, 500)
    e = 10 ** rng.uniform(-3, 4, 500)
    cls = rng.integers(1, 4, 500)
    gr = np.logspace(-2, 1.5, 100)
    cp = 0.28 + 0.1j / (np.abs(gr - 2) + 0.1)
    return dict(W=W, Nv=Nv, x=x, spectra=spectra, g=g, e=e, cls=cls, gr=gr, cp=cp)


def test_extension_figures_keep_legends_and_labels_outside_plot_areas():
    from nh_topo import extension_figures as XF
    from nh_topo import figures as F
    import matplotlib.pyplot as plt
    F.apply_style()            # the notebook applies the original style first
    d = _figure_inputs()
    figs = [
        XF.fig_pinning(d["W"], 1e-15 * np.ones(7), np.linspace(0.01, 0.2, 7), 1e-3 * np.ones(7),
                       d["Nv"], 0.3 * 0.67 ** d["Nv"], 1e-16 * np.ones(6), d["Nv"],
                       0.3 * 0.67 ** d["Nv"]),
        XF.fig_observability(d["x"], d["spectra"], ["0", "1", r"$\sqrt{2}$", "2 (EP)", "6"],
                             d["g"], d["e"], d["cls"], np.linspace(1e-3, 1.4, 50),
                             np.linspace(1.7, 1e-3, 50), np.logspace(-5, -1, 20),
                             [np.arange(28, 8, -1), np.arange(13, -7, -1)], ["1.5", "2.51"]),
        XF.fig_weights(d["gr"], d["cp"], 0.557 - d["cp"], 0.557 * np.ones(100),
                       0.554 * np.ones(100), 0.557, np.linspace(0, 0.8, 30),
                       np.linspace(1, 1.1, 30), np.linspace(1, 13, 30), np.linspace(68, 6, 30)),
        XF.fig_fullwave(np.linspace(0.275, 0.345, 50), np.linspace(3, 1, 50), np.ones(50),
                        np.linspace(2, 0.5, 50), (0.295, 0.32), np.array([0, 0.5, 1, 2]),
                        np.array([7.9, 3.9, 2.6, 1.6])),
    ]
    for f in figs:
        assert XF.layout_overlaps(f) == []
        plt.close(f)


def test_layout_check_flags_a_legend_inside_the_axes():
    from nh_topo import extension_figures as XF
    import matplotlib.pyplot as plt
    XF.apply_style()
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1], [0, 1], label="data")
    ax.legend(loc="upper left")
    issues = XF.layout_overlaps(fig)
    plt.close(fig)
    assert any("plot area" in s for s in issues)


# --- added after the blind review: the peak criterion must use the dip ------

def _two_lorentzians(eta, om):
    return 1.0 / ((om - 1) ** 2 + eta ** 2) + 1.0 / ((om + 1) ** 2 + eta ** 2)


def test_threshold_acts_on_the_dip_of_a_symmetric_doublet():
    om = np.linspace(-4, 4, 16001)                 # symmetric grid, exactly equal twin peaks
    shallow = _two_lorentzians(1.6, om)            # dip between 0 and 1 %
    deep = _two_lorentzians(1.2, om)               # dip above 5 %
    dip = lambda r: 1 - r[om.size // 2] / r.max()
    assert 0 < dip(shallow) < 1e-2 and dip(deep) > 5e-2
    assert ob.count_peaks(shallow, ob.ZERO_CONTRAST) == 2
    assert ob.count_peaks(shallow, 1e-2) == 1      # fails if the threshold is ignored
    assert ob.count_peaks(deep, 1e-2) == 2
    assert ob.doublet_contrast(shallow) == pytest.approx(dip(shallow), rel=1e-6)
    assert ob.count_peaks(_two_lorentzians(2.0, om), ob.ZERO_CONTRAST) == 1


def test_dip_contrast_limit():
    om = np.linspace(-4, 4, 40001)
    assert ob.dip_contrast_limit(0.0) == pytest.approx(np.sqrt(3))
    for rel in (1e-2, 5e-2, 0.2):
        c = ob.dip_contrast_limit(rel)
        assert ob.doublet_contrast(_two_lorentzians(0.995 * c, om)) > rel
        assert ob.doublet_contrast(_two_lorentzians(1.005 * c, om)) < rel
    assert ob.dip_contrast_limit(1e-2) < np.sqrt(3)


def test_max_resolvable_length_with_contrast():
    for eta in (1e-3, 1e-2):
        n0 = ob.max_resolvable_length(T1, T2, eta)
        n5 = ob.max_resolvable_length(T1, T2, eta, rel_contrast=0.05)
        assert n5 <= n0
        c = ob.dip_contrast_limit(0.05)
        assert c * nt.bare_hybridization(n5, T1, T2) > eta >= c * nt.bare_hybridization(n5 + 1, T1, T2)


def test_map_at_one_percent_contrast_matches_two_level_line_shape():
    gs = np.logspace(-4, np.log10(0.3), 8)
    es = np.logspace(-4, -1, 8)
    m = ob.observability_map([6, 12], T1, T2, gs, es, n_side=1500)
    c_full, c_two = ob.classes_at_contrast(m, 1e-2)
    assert np.array_equal(c_full, c_two)
    assert np.sum(c_full == 1) <= np.sum(m["cls"] == 1)


def test_disorder_resolvability_with_disorder_matches_prediction():
    for rel in (ob.ZERO_CONTRAST, 1e-2):
        eta = 1.5 * nt.bare_hybridization(5, T1, T2)
        r = ob.disorder_resolvability(5, T1, T2, 0.3, eta, 40, 11, rel_prominence=rel)
        assert 0.0 < r["fraction_numeric"] < 1.0      # the case is not trivial
        assert r["agreement"] == 1.0


# --- added after the second blind review: pin what the paper reports ---------

def test_doublet_contrast_is_the_second_most_prominent_maximum():
    x = np.linspace(-10, 10, 20001)
    lor = lambda c, w, a: a * w ** 2 / ((x - c) ** 2 + w ** 2)
    rho = lor(-5, 0.3, 1.0) + lor(0, 0.3, 0.6) + lor(5, 0.3, 0.3)   # three separated peaks
    pr = ob.peak_contrasts(rho)
    assert pr.size == 3
    assert ob.doublet_contrast(rho) == pytest.approx(pr[1])
    assert 0.5 < ob.doublet_contrast(rho) < 0.62                      # the 0.6 peak, not the 0.3 one


def test_detuning_fraction_uses_the_stated_contrast():
    r = ob.detuning_resolvability(4, 14.8, 37.1, 1.0, 1.0, 60, 12345, n_omega=8001)
    assert r["fraction"] == pytest.approx(np.mean(r["contrast"] > 1e-2))
    assert r["fraction_zero_contrast"] == pytest.approx(np.mean(r["contrast"] > ob.ZERO_CONTRAST))
    assert r["fraction"] < r["fraction_zero_contrast"]              # the two criteria differ here


def test_map_two_level_contrast_uses_the_same_eta():
    gs, es = np.array([0.3e-2]), np.array([1.2e-2, 2.0e-2])
    m = ob.observability_map([10], T1, T2, gs, es, n_side=1500)
    om = ob.log_symmetric_grid(0.15, 1e-10, 1500)
    for k, e in enumerate(es):
        ref = ob.doublet_contrast(ob.two_level_edge_ldos(om, m["delta0"][0], gs[0], e))
        assert m["contrast_two_level"][0, 0, k] == pytest.approx(ref, rel=1e-12, abs=1e-14)
    assert m["contrast_two_level"][0, 0, 0] > 1e-3                   # a resolved, non-trivial case
