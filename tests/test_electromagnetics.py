"""Full-wave solver: verification against analytic results and the physics it is used for."""
import numpy as np
import pytest

from nh_topo import electromagnetics as em


def test_homogeneous_purcell_approaches_unity():
    """
    An emitter in unbounded vacuum has Purcell factor one by definition, so this
    fixes the normalisation of the whole calculation.
    """
    bench = em.validate_homogeneous_medium(resolutions=(20, 40))
    assert bench["purcell_error"][-1] < 1e-3
    assert bench["purcell_error"][-1] < bench["purcell_error"][0]


def test_solver_is_second_order():
    """Halving the grid spacing should reduce the error roughly fourfold."""
    bench = em.validate_homogeneous_medium(resolutions=(20, 40))
    ratio = bench["purcell_error"][0] / bench["purcell_error"][-1]
    assert 2.5 < ratio < 6.0


def test_field_matches_hankel_function():
    """The radial profile away from the source must follow (i/4) H_0^(1)(k r)."""
    bench = em.validate_homogeneous_medium(resolutions=(40,))
    assert bench["profile_error"][0] < 1e-3


def test_permittivity_is_built_correctly():
    """Rods carry the specified permittivity, with loss on the B sublattice only."""
    grid = em.Grid(nx=120, ny=80, h=1 / 20, pml_cells=10)
    centres = em.rod_positions(3, 0.6, 0.4)
    eps = em.build_permittivity(grid, centres, 0.15, 12.0, loss_b=0.5)

    assert np.isclose(eps.real.max(), 12.0, atol=1e-6)
    assert np.isclose(eps.real.min(), 1.0, atol=1e-6)
    assert eps.imag.max() > 0.4          # loss present on B rods
    assert eps.imag.min() >= -1e-12      # and nowhere negative (no gain)

    a_centre = grid.index(centres[0], 0.0)
    b_centre = grid.index(centres[1], 0.0)
    assert eps[a_centre].imag == pytest.approx(0.0, abs=1e-12)
    assert eps[b_centre].imag > 0.4


def test_dimerised_geometry_alternates():
    """The two gap lengths must alternate, which is what makes the chain an SSH chain."""
    centres = em.rod_positions(4, 0.6, 0.4)
    gaps = np.diff(centres)
    assert np.allclose(gaps[0::2], 0.6)
    assert np.allclose(gaps[1::2], 0.4)


@pytest.mark.slow
def test_topological_geometry_enhances_and_trivial_does_not():
    """
    The full-wave counterpart of the tight-binding statement: a boundary
    enhancement appears only when the chain is dimerised into the topological
    phase. The control has the same period, hence the same bulk bands.
    """
    study = em.photonic_ldos_study(n_cells=6, loss_b=0.0, resolution=20, n_freq=9,
                                   freq_range=(0.295, 0.320))
    assert study["topological"]["gap_enhancement"] > 3.0
    assert study["trivial"]["gap_enhancement"] < 1.5
