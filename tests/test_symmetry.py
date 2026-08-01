"""CS-dagger classification -- the core physics correction."""
import numpy as np
import pytest
import nh_topo as nt


def test_hermitian_chiral_symmetry_is_broken(H, params):
    """sigma_z H sigma_z != -H; the residual is exactly 2*gamma."""
    assert np.isclose(nt.chiral_residual(H), 2 * params["gamma"], rtol=1e-12)


def test_cs_dagger_symmetry_holds(H, params):
    """sigma_z Htilde sigma_z = -Htilde^dagger to machine precision."""
    assert nt.cs_dagger_residual(H, params["gamma"]) < 1e-12


def test_spectrum_symmetric_about_imaginary_axis(H, params):
    """CS-dagger => spec(Htilde) closed under E -> -E*."""
    assert nt.spectral_symmetry_residual(H, params["gamma"]) < 1e-10


def test_two_modes_pinned(H):
    """Exactly two eigenvalues sit at Re E = 0 in the topological phase."""
    assert nt.count_pinned_modes(H) == 2


def test_hermitian_limit_recovers_chiral_symmetry(params):
    """At gamma = 0 the ordinary chiral symmetry is restored."""
    H0 = nt.build_ssh(params["N"], params["t1"], params["t2"], 0.0)
    assert nt.chiral_residual(H0) < 1e-12


@pytest.mark.parametrize("W", [0.2, 0.6, 1.0, 2.0])
def test_bond_disorder_preserves_cs_dagger(params, W):
    """Off-diagonal disorder preserves CS-dagger exactly, at ANY strength."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        Hd = nt.make_disordered_ssh(params["N"], params["t1"], params["t2"],
                                    params["gamma"], W, "bond", rng)
        assert nt.cs_dagger_residual(Hd, params["gamma"]) < 1e-12


@pytest.mark.parametrize("W", [0.2, 0.6])
def test_onsite_disorder_breaks_cs_dagger(params, W):
    """Diagonal disorder breaks CS-dagger by ~2V."""
    rng = np.random.default_rng(2)
    Hd = nt.make_disordered_ssh(params["N"], params["t1"], params["t2"],
                                params["gamma"], W, "on_site", rng)
    assert nt.cs_dagger_residual(Hd, params["gamma"]) > 1e-3


def test_shift_preserves_eigenvectors(H, params):
    """
    The imaginary shift is rigid: every eigenpair (v, E) of H satisfies
    Htilde v = (E + i*gamma/2) v exactly. Stated this way the check is free of
    eigenvalue-ordering ambiguity, which matters here because two eigenvalues are
    degenerate in Re(E).
    """
    Ht = nt.shifted_hamiltonian(H, params["gamma"])
    evals, evecs = np.linalg.eig(H)
    shift = 1j * params["gamma"] / 2
    for n in range(evals.size):
        v = evecs[:, n]
        assert np.allclose(Ht @ v, (evals[n] + shift) * v, atol=1e-10)


def test_edge_exceptional_point_matches_two_level_model(params):
    """
    The boundary doublet follows E = +/- sqrt(delta0^2 - gamma^2/4) below its
    exceptional point at gamma = 2*delta0, to five significant figures.
    """
    N, t1, t2 = 10, params["t1"], params["t2"]
    d0 = nt.bare_hybridization(N, t1, t2)
    assert np.isclose(nt.edge_exceptional_point(N, t1, t2), 2 * d0)
    for frac in (0.0, 0.25, 0.5, 0.9, 1.5, 1.9):
        g = frac * d0
        obs = np.min(np.abs(np.linalg.eigvals(nt.build_ssh(N, t1, t2, g)).real))
        pred = nt.two_level_splitting(g, d0)
        assert np.isclose(obs, float(pred), rtol=1e-4)


def test_above_edge_ep_real_part_is_pinned(params):
    """Beyond the edge EP the real part collapses to machine zero."""
    N, t1, t2 = 10, params["t1"], params["t2"]
    d0 = nt.bare_hybridization(N, t1, t2)
    for frac in (2.5, 5.0, 20.0):
        obs = np.min(np.abs(np.linalg.eigvals(nt.build_ssh(N, t1, t2, frac * d0)).real))
        assert obs < 1e-12


def test_delta0_decays_exponentially_with_length(params):
    """
    delta_0 ~ exp(-N/xi): the fitted decay length must match the analytic
    localisation length 1/ln(t2/t1).
    """
    ds = nt.delta0_size_scaling([6, 8, 10, 12, 15, 18, 20, 25],
                                params["t1"], params["t2"])
    assert ds["rel_error"] < 0.02


def test_ep_condition_predicts_which_realisations_split(params):
    """
    The validity condition gamma > 2*delta_0, evaluated with the DISORDERED
    delta_0, predicts exactly the fraction of realisations that lose the pinning.
    """
    d = nt.fraction_below_edge_ep(5, params["t1"], params["t2"],
                                  params["gamma"], 0.4, 300, 42)
    assert np.isclose(d["fraction_below_ep"], 1.0 - d["frac_pinned"], atol=0.01)


def test_long_chains_are_always_pinned(params):
    """For N >= 6 at moderate disorder no realisation falls below the edge EP."""
    for N in (6, 8, 10, 20):
        d = nt.fraction_below_edge_ep(N, params["t1"], params["t2"],
                                      params["gamma"], 0.4, 100, 42)
        assert d["frac_pinned"] == 1.0
