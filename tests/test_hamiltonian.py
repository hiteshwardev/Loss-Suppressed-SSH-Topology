"""Structural properties of the Hamiltonian."""
import numpy as np
import nh_topo as nt


def test_complex_symmetric(H):
    """Reciprocal hopping => H = H^T (not Hermitian). Basis of the biorthogonal treatment."""
    assert np.allclose(H, H.T)
    assert not np.allclose(H, H.conj().T)


def test_passive_spectrum(H, params):
    """Loss-only => every eigenvalue in the lower half plane, Im E in [-gamma, 0]."""
    ev = np.linalg.eigvals(H)
    assert np.all(ev.imag <= 1e-9)
    assert np.all(ev.imag >= -params["gamma"] - 1e-9)


def test_no_skin_effect(params):
    """Reciprocity: |H_ij| = |H_ji| for all i,j, so there is no skin effect."""
    H = nt.build_ssh(params["N"], params["t1"], params["t2"], params["gamma"])
    assert np.allclose(np.abs(H), np.abs(H.T))


def test_loss_only_on_B(params):
    """Imaginary part lives exclusively on odd (B) sites."""
    H = nt.build_ssh(params["N"], params["t1"], params["t2"], params["gamma"])
    d = np.diag(H)
    assert np.allclose(d[0::2].imag, 0.0)
    assert np.allclose(d[1::2].imag, -params["gamma"])


def test_bulk_exceptional_point(params):
    """gamma_EP = 2|t2-t1|, NOT |t2-t1|; operating gamma is below it."""
    ep = nt.exceptional_point_threshold(params["t1"], params["t2"])
    assert np.isclose(ep, 2 * abs(params["t2"] - params["t1"]))
    assert params["gamma"] < ep


def test_onsite_disorder_keeps_symmetry(params):
    """Real on-site shifts preserve H = H^T (they break CS-dagger, not symmetry of H)."""
    rng = np.random.default_rng(0)
    Hd = nt.build_ssh_general(params["N"], params["gamma"], params["t1"],
                              params["t2"], onsite=rng.normal(size=2 * params["N"]))
    assert np.allclose(Hd, Hd.T)
