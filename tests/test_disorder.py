"""
Disorder protection, and the regression test for the edge-mode detection bug.

BACKGROUND
----------
An earlier version identified the protected mode with a fixed boundary-weight
threshold (0.30) and no in-gap restriction, then reported min|Re E| over the
resulting set. At W = 0.6 t1 that admitted band-edge bulk states in 3/200
realisations and reported |Re E| ~ 1.2 t1 as the "displacement", inflating the
ensemble mean to 0.023 t1 -- a value forbidden by the exact CS-dagger symmetry.
`test_bond_disorder_displacement_is_machine_zero` locks the fix in.
"""
import numpy as np
import pytest
import nh_topo as nt


@pytest.mark.parametrize("W", [0.0, 0.2, 0.4, 0.6, 1.0])
def test_bond_disorder_displacement_is_machine_zero(params, W):
    """
    REGRESSION TEST. CS-dagger forbids chiral-preserving disorder from moving
    Re E of a self-conjugate mode. The displacement must be machine zero at
    every strength -- never the 0.023 t1 previously reported.
    """
    rng = np.random.default_rng(42)
    ref = nt.clean_protected_mode(params["N"], params["t1"], params["t2"], params["gamma"])
    for _ in range(50):
        Hd = nt.make_disordered_ssh(params["N"], params["t1"], params["t2"],
                                    params["gamma"], W, "bond", rng)
        d = nt.protected_mode_displacement(Hd, reference=ref)
        assert d < 1e-10, f"bond disorder moved Re E by {d:g} at W={W}"


@pytest.mark.parametrize("W", [0.0, 0.2, 0.4, 0.6, 1.0])
def test_two_modes_stay_pinned_under_bond_disorder(params, W):
    """
    Independent of any mode identification: Re E = 0 stays occupied.

    CS-dagger pins self-conjugate modes to the imaginary axis, so the count is
    always EVEN and at least 2. At very strong disorder (W >~ t1) the chain can
    fragment into weakly connected segments, each contributing its own pinned
    pair; we observed 4 pinned modes in 6/200 realisations at W = 1.0 t1. That is
    additional protected modes, not a loss of protection.
    """
    rng = np.random.default_rng(7)
    for _ in range(30):
        Hd = nt.make_disordered_ssh(params["N"], params["t1"], params["t2"],
                                    params["gamma"], W, "bond", rng)
        n = nt.count_pinned_modes(Hd)
        assert n >= 2 and n % 2 == 0


def test_onsite_disorder_displaces_the_mode(params):
    """Symmetry-breaking disorder does move the mode, monotonically in W."""
    ref = nt.clean_protected_mode(params["N"], params["t1"], params["t2"], params["gamma"])
    means = []
    for W in (0.1, 0.2, 0.4, 0.6):
        rng = np.random.default_rng(3)
        vals = [nt.protected_mode_displacement(
            nt.make_disordered_ssh(params["N"], params["t1"], params["t2"],
                                   params["gamma"], W, "on_site", rng), reference=ref)
            for _ in range(60)]
        means.append(np.mean(vals))
    assert means[0] > 1e-4
    assert np.all(np.diff(means) > 0)


def test_edge_detector_rejects_band_edge_states(params):
    """
    The in-gap restriction must exclude band-edge states however localized they
    appear -- the specific failure mode of the old detector.
    """
    rng = np.random.default_rng(42)
    for _ in range(200):
        Hd = nt.make_disordered_ssh(params["N"], params["t1"], params["t2"],
                                    params["gamma"], 0.6, "bond", rng)
        ev, vec = nt.diagonalize(Hd)
        info = nt.find_edge_modes(ev, vec, gap_half_width=abs(params["t2"] - params["t1"]))
        if info["protected_idx"].size:
            assert np.all(np.abs(ev[info["protected_idx"]].real) < abs(params["t2"] - params["t1"]))


def test_overlap_tracking_is_stable(params):
    """Adiabatic continuation returns a boundary-localized state at strong disorder."""
    rng = np.random.default_rng(11)
    ref = nt.clean_protected_mode(params["N"], params["t1"], params["t2"], params["gamma"])
    for _ in range(30):
        Hd = nt.make_disordered_ssh(params["N"], params["t1"], params["t2"],
                                    params["gamma"], 0.6, "bond", rng)
        ev, vec = nt.diagonalize(Hd)
        idx = nt.track_mode_by_overlap(vec, ref)
        lw, rw = nt.edge_localization_weights(vec[:, [idx]], edge_cells=1)
        assert max(lw[0], rw[0]) > 0.05


def test_trivial_control_shows_no_enhancement(params):
    """The control that separates 'topological' from merely 'gapped'."""
    Ht = nt.build_ssh(params["N"], params["t1"], 0.6 * params["t1"], params["gamma"])
    om = np.linspace(-params["gap_hw"], params["gap_hw"], 121)
    e = nt.enhancement_factor(Ht, 0, nt.central_bulk_site(params["N"]), om, params["eta"])
    assert 0.5 < e < 1.5


def test_loss_suppresses_hybridisation(params):
    """
    Central claim: at equal length and identical disorder, the Hermitian chain
    splits while the lossy chain stays pinned.
    """
    hs = nt.hybridization_study(20, params["t1"], params["t2"],
                                [0.0, params["gamma"]], [0.0, 0.4], 20, 42)
    herm, lossy = hs["splitting"][0], hs["splitting"][1]
    assert np.all(herm > 1e-6)
    assert np.all(lossy < 1e-12)
    assert np.all(herm / np.maximum(lossy, 1e-300) > 1e6)


def test_hybridisation_decays_with_size(params):
    """Hermitian splitting ~ exp(-N/xi); lossy stays pinned for every N."""
    hz = nt.hybridization_vs_size([8, 12, 20], params["t1"], params["t2"],
                                  [0.0, params["gamma"]], 0.0, 1, 42)
    herm = hz["splitting"][0]
    assert np.all(np.diff(herm) < 0)
    assert np.all(hz["splitting"][1] < 1e-12)


def test_disorder_study_is_reproducible(params):
    """Same seed => identical results."""
    om = np.linspace(-params["gap_hw"], params["gap_hw"], 41)
    kw = dict(N=params["N"], t1=params["t1"], t2=params["t2"], gamma=params["gamma"],
              edge_site=0, bulk_site=nt.central_bulk_site(params["N"]),
              gap_omegas=om, eta=params["eta"], W_vals=[0.2], kinds=["bond"],
              num_realizations=5, seed=123)
    a = nt.run_disorder_study(**kw)["topological"]["bond"][0.2]["enh"]["mean"]
    b = nt.run_disorder_study(**kw)["topological"]["bond"][0.2]["enh"]["mean"]
    assert a == b


def test_trivial_control_loses_discriminating_power(params):
    """
    The control only means something while the trivial phase stays featureless.
    Strong bond disorder localises states near the boundary by accident, so the
    anomaly rate must be negligible at moderate disorder and appreciable at
    strong disorder. Quantifying this bounds the range over which the control
    argument carries weight.
    """
    om = np.linspace(-params["gap_hw"], params["gap_hw"], 61)
    a = nt.trivial_control_anomalies(params["N"], params["t1"], 0.6 * params["t1"],
                                     params["gamma"], [0.2, 1.5], 0,
                                     nt.central_bulk_site(params["N"]), om,
                                     params["eta"], 60, 42, threshold=2.0)
    assert a["bond"]["anomaly_rate"][0] < 0.05
    assert a["bond"]["anomaly_rate"][1] > 0.15


def test_protection_survives_to_strong_bond_disorder(params):
    """
    Pinning holds far beyond the moderate disorder usually tested, and the
    breakdown when it finally comes is a fragmentation effect rather than a
    symmetry violation: the CS-dagger residual stays exactly zero throughout.
    """
    ref = nt.clean_protected_mode(params["N"], params["t1"], params["t2"], params["gamma"])
    rng = np.random.default_rng(5)
    for _ in range(40):
        Hd = nt.make_disordered_ssh(params["N"], params["t1"], params["t2"],
                                    params["gamma"], 1.0, "bond", rng)
        assert nt.cs_dagger_residual(Hd, params["gamma"]) < 1e-12
        assert nt.protected_mode_displacement(Hd, reference=ref) < 1e-10
