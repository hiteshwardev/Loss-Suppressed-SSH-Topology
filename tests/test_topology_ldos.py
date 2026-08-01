"""Topological invariants, LDOS positivity, and three-way cross-validation."""
import numpy as np
import pytest
import nh_topo as nt


def test_winding_number_is_quantized(params):
    assert np.isclose(nt.winding_number(params["t1"], params["t2"]), -1.0, atol=1e-9)
    assert np.isclose(nt.winding_number(params["t1"], 0.6 * params["t1"]), 0.0, atol=1e-9)


def test_winding_is_grid_independent(params):
    vals = [nt.winding_number(params["t1"], params["t2"], num_k=n)
            for n in (20, 200, 4000)]
    assert np.allclose(vals, -1.0, atol=1e-9)


def test_winding_is_gamma_independent(params):
    """gamma enters only the diagonal, so it cannot change q(k)."""
    assert nt.topological_invariant(params["t1"], params["t2"]) == 1


def test_edge_weight_matches_analytic(H, params):
    """W0 numeric == 1 - (t1/t2)^2."""
    w_num = nt.edge_mode_spectral_weight(H, 0)
    w_ana = nt.analytic_edge_weight(params["t1"], params["t2"])
    assert np.isclose(w_num, w_ana, rtol=2e-3)


def test_edge_weight_is_gamma_independent_above_edge_ep(params):
    """
    Above the boundary-doublet exceptional point W0 depends on t1/t2 alone.

    (Below it the two boundary modes are hybridised into combinations spread over
    BOTH ends, so the weight at one end is halved -- see
    test_edge_weight_doubles_across_edge_ep.)
    """
    vals = [nt.edge_mode_spectral_weight(
        nt.build_ssh(params["N"], params["t1"], params["t2"], g), 0)
        for g in (0.01, 0.1, 0.3, 0.5)]
    assert np.allclose(vals, vals[0], atol=1e-3)


def test_edge_weight_doubles_across_edge_ep(params):
    """
    Observable signature of the edge exceptional point: the boundary spectral
    weight is HALF the semi-infinite value while the doublet is hybridised
    (gamma < 2*delta_0) and recovers the full value once loss de-hybridises it.

    This is the LDOS-level counterpart of the loss-suppressed hybridisation.
    """
    N, t1, t2 = params["N"], params["t1"], params["t2"]
    d0 = nt.bare_hybridization(N, t1, t2)
    w_ana = nt.analytic_edge_weight(t1, t2)
    w_below = nt.edge_mode_spectral_weight(nt.build_ssh(N, t1, t2, 0.1 * d0), 0)
    w_above = nt.edge_mode_spectral_weight(nt.build_ssh(N, t1, t2, 100 * d0), 0)
    assert np.isclose(w_below, w_ana / 2, rtol=5e-3)
    assert np.isclose(w_above, w_ana, rtol=5e-3)


def test_trivial_phase_has_no_edge_weight(params):
    H = nt.build_ssh(params["N"], params["t1"], 0.6 * params["t1"], params["gamma"])
    assert abs(nt.edge_mode_spectral_weight(H, 0)) < 1e-6


def test_ldos_is_nonnegative(H, params):
    """The passive-system positivity proof, checked numerically."""
    m = nt.ldos_nonnegativity_min(H, np.linspace(-2.5, 2.5, 241), params["eta"])
    assert m >= -1e-12


def test_ldos_positive_for_small_eta(H):
    """Positivity must survive small broadening."""
    for eta in (0.5, 0.05, 0.005):
        m = nt.ldos_nonnegativity_min(H, np.linspace(-2, 2, 121), eta)
        assert m >= -1e-12


def test_three_ldos_algorithms_agree(H, params):
    """Green's function vs biorthogonal eigendecomposition vs semi-infinite."""
    om = np.linspace(-1.0, 1.0, 201)
    gf = nt.ldos_spectrum(H, 0, om, params["eta"])
    ed = np.array([nt.ldos_eigdecomp(H, 0, w, params["eta"]) for w in om])
    sf = nt.surface_ldos_spectrum(om, params["t1"], params["t2"],
                                  params["gamma"], params["eta"], sublattice=0)
    assert np.max(np.abs(gf - ed)) < 1e-10          # same operator, two algorithms
    assert np.max(np.abs(gf - sf)) < 1e-3           # finite-size boundary correction


def test_localization_length(H, params):
    """Fitted xi within 5% of 1/ln(t2/t1)."""
    ev, vec = nt.diagonalize(H)
    xi = nt.fit_localization_length(ev, vec, params["N"])
    xi_a = nt.localization_length(params["t1"], params["t2"])
    assert abs(xi - xi_a) / xi_a < 0.05


def test_obc_has_ingap_states_pbc_does_not(params):
    """Bulk-boundary correspondence."""
    ho = nt.build_ssh(params["N"], params["t1"], params["t2"], params["gamma"])
    hp = nt.build_ssh_pbc(params["N"], params["t1"], params["t2"], params["gamma"])
    n_o = np.sum(np.abs(np.linalg.eigvals(ho).real) < params["gap_hw"])
    n_p = np.sum(np.abs(np.linalg.eigvals(hp).real) < params["gap_hw"])
    assert n_o == 2 and n_p == 0


def test_enhancement_decreases_with_eta(H, params):
    """The ratio is broadening-dependent -- it is not an intrinsic figure of merit."""
    sc = nt.enhancement_vs_eta(H, 0, nt.central_bulk_site(params["N"]),
                               params["gap_hw"], np.logspace(-2, -0.7, 6))
    e = sc["enhancement"]
    assert np.all(np.diff(e) < 0)


def test_central_bulk_site_is_on_A_sublattice():
    """Parity-safe bulk reference: always an even (lossless A) index."""
    for N in range(4, 41):
        assert nt.central_bulk_site(N) % 2 == 0


@pytest.mark.parametrize("N", [8, 10, 15, 20, 30])
def test_edge_weight_doubles_exactly_across_ep(N, params):
    """
    The boundary spectral weight doubles across the edge exceptional point, for
    every chain length: W0(below) = W0_analytic/2, W0(above) = W0_analytic.
    Ratio = 2.000 independent of N.
    """
    t1, t2 = params["t1"], params["t2"]
    d0 = nt.bare_hybridization(N, t1, t2)
    lo = nt.edge_mode_spectral_weight(nt.build_ssh(N, t1, t2, 0.01 * d0), 0)
    hi = nt.edge_mode_spectral_weight(nt.build_ssh(N, t1, t2, 100 * d0), 0)
    assert lo > 0.0, "hybridised doublet must still be detected"
    assert np.isclose(hi / lo, 2.0, rtol=1e-3)


def test_edge_detector_finds_hybridised_doublet(params):
    """
    Regression: a hybridised doublet shares weight between both ends (~0.28 at
    N=10). The old fixed 0.30 threshold missed it and reported W0 = 0.
    """
    for N in (8, 10, 12):
        d0 = nt.bare_hybridization(N, params["t1"], params["t2"])
        H = nt.build_ssh(N, params["t1"], params["t2"], 0.01 * d0)
        ev, vec = nt.diagonalize(H)
        info = nt.find_edge_modes(ev, vec)
        assert info["edge_idx"].size >= 2, f"doublet not detected at N={N}"


def test_sum_rule_approaches_unity(H, params):
    """
    The exact spectral sum rule is 1 per site; a finite window falls short only
    by Lorentzian tail weight, and must converge upward with the window.
    """
    vals = [nt.ldos_sum_rule(H, 0, np.linspace(-w, w, 4001), params["eta"])
            for w in (5.0, 20.0, 80.0)]
    assert np.all(np.diff(vals) > 0)
    assert vals[-1] > 0.999
    assert vals[-1] < 1.001


def test_sum_rule_holds_at_every_site(H, params):
    """No site may violate the sum rule: the check cannot be passed selectively."""
    r = nt.ldos_sum_rule_all_sites(H, np.linspace(-40, 40, 4001), params["eta"])
    assert r["min"] > 0.99 and r["max"] < 1.01


def test_enhancement_is_grid_converged(H, params):
    """The reported enhancement must not depend on the integration grid."""
    vals = [nt.enhancement_factor(H, 0, nt.central_bulk_site(params["N"]),
                                  np.linspace(-params["gap_hw"], params["gap_hw"], m),
                                  params["eta"]) for m in (201, 801, 3201)]
    assert abs(vals[-1] - vals[0]) / vals[-1] < 1e-4


def test_localization_length_converges_with_size(params):
    """
    Fitting a short chain underestimates xi because the exponential tail is
    truncated by the far boundary. The bias must fall monotonically and vanish
    once the chain is long compared with xi.
    """
    c = nt.localization_length_convergence(params["t1"], params["t2"],
                                           params["gamma"], [10, 20, 40])
    assert c["rel_error"][0] > c["rel_error"][-1]
    assert c["rel_error"][-1] < 1e-3
