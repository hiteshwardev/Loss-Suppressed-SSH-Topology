"""
Finite-difference frequency-domain solution of Maxwell's equations in two
dimensions, used to test the tight-binding predictions against a full-wave
calculation in a fabricable dielectric structure.

Formulation
-----------
For transverse-magnetic polarisation the electric field has a single component
E_z(x, y) and Maxwell's equations reduce to the scalar Helmholtz equation

    d2E/dx2 + d2E/dy2 + w^2 eps(x, y) E = -S ,

written in units with c = eps_0 = mu_0 = 1, so that the free-space wavenumber is
k_0 = w and lengths are measured in units of the mean lattice period a.
Frequencies are quoted as a/lambda = w / 2pi.

Outgoing-wave boundaries are imposed by complex coordinate stretching. Inside an
absorbing layer of thickness d the coordinate is scaled by

    s(x) = 1 + i sigma(x) / w ,    sigma(x) = sigma_max ((d - x) / d)^m ,

with the polynomial grading exponent m and the peak conductivity chosen from the
target round-trip reflectance R0,

    sigma_max = -(m + 1) ln(R0) / (2 d) .

Discretising the stretched Laplacian on a uniform grid with spacing h gives, for
the x direction,

    (1 / s_i h) [ (E_{i+1} - E_i) / (s_{i+1/2} h) - (E_i - E_{i-1}) / (s_{i-1/2} h) ] ,

and similarly for y. The resulting sparse system is factorised directly.

Local density of states
-----------------------
A unit line current at r_0 produces, in a homogeneous medium of permittivity
eps, the field E_z(r) = (i/4) H_0^(1)(w sqrt(eps) |r - r_0|). The real part
diverges logarithmically as r -> r_0, but the imaginary part is finite,

    Im E_z(r_0) = 1/4 ,

independent of frequency and permittivity. The power delivered by the source is
proportional to Im E_z(r_0), so the Purcell factor is the ratio of that quantity
in the structure to its homogeneous-medium value. Taking a ratio on the same
grid also cancels the grid-dependent part of the self-field, which is what makes
the quantity numerically well defined. `validate_homogeneous_medium` checks the
solver against the analytic value above.
"""

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import hankel1


# ---------------------------------------------------------------------------
# Grid and absorbing boundary
# ---------------------------------------------------------------------------

@dataclass
class Grid:
    """Uniform Cartesian grid with an absorbing layer of `pml_cells` on each side."""

    nx: int
    ny: int
    h: float
    pml_cells: int = 20

    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)

    def __post_init__(self):
        self.x = (np.arange(self.nx) - self.nx / 2 + 0.5) * self.h
        self.y = (np.arange(self.ny) - self.ny / 2 + 0.5) * self.h

    @property
    def shape(self):
        return (self.nx, self.ny)

    def index(self, xv, yv):
        """Grid indices of the point closest to the physical coordinates (xv, yv)."""
        return int(np.argmin(np.abs(self.x - xv))), int(np.argmin(np.abs(self.y - yv)))


def _stretch(n, h, npml, omega, m=3.0, reflectance=1e-8):
    """
    Coordinate-stretching factors s at grid points and at half-grid points.

    Returns two arrays of length n and n + 1 respectively, both equal to one
    outside the absorbing layer.
    """
    if npml <= 0:
        return np.ones(n, complex), np.ones(n + 1, complex)

    d = npml * h
    sigma_max = -(m + 1.0) * np.log(reflectance) / (2.0 * d)

    def sigma(u):
        """Conductivity as a function of position measured from the left edge."""
        s = np.zeros_like(u)
        left = u < d
        s[left] = sigma_max * ((d - u[left]) / d) ** m
        total = (n - 1) * h
        right = u > total - d
        s[right] = sigma_max * ((u[right] - (total - d)) / d) ** m
        return s

    u_full = np.arange(n) * h
    u_half = (np.arange(n + 1) - 0.5) * h
    return (1.0 + 1j * sigma(u_full) / omega,
            1.0 + 1j * sigma(u_half) / omega)


def _laplacian_1d(n, h, s, s_half):
    """
    Stretched second-derivative operator with Dirichlet closure.

    The absorbing layer makes the outer boundary invisible, so the choice of
    closure there is immaterial.
    """
    inv = 1.0 / (s * h)
    inv_half = 1.0 / (s_half * h)

    lower = inv[1:] * inv_half[1:-1]
    upper = inv[:-1] * inv_half[1:-1]
    diag = -(inv * (inv_half[:-1] + inv_half[1:]))

    return sp.diags([lower, diag, upper], [-1, 0, 1], format="csr")


def helmholtz_operator(grid: Grid, omega: float, eps: np.ndarray) -> sp.csr_matrix:
    """Assemble the sparse operator for d2/dx2 + d2/dy2 + omega^2 eps."""
    sx, sx_h = _stretch(grid.nx, grid.h, grid.pml_cells, omega)
    sy, sy_h = _stretch(grid.ny, grid.h, grid.pml_cells, omega)

    lx = _laplacian_1d(grid.nx, grid.h, sx, sx_h)
    ly = _laplacian_1d(grid.ny, grid.h, sy, sy_h)

    ix = sp.identity(grid.nx, format="csr")
    iy = sp.identity(grid.ny, format="csr")

    laplacian = sp.kron(lx, iy, format="csr") + sp.kron(ix, ly, format="csr")
    return (laplacian + sp.diags(omega ** 2 * eps.ravel())).tocsc()


def solve_ez(grid: Grid, omega: float, eps: np.ndarray,
             source_index) -> np.ndarray:
    """
    Field radiated by a unit line source at `source_index = (i, j)`.

    The source amplitude is scaled by 1 / h^2 so that it represents a delta
    function of unit weight on the discrete grid, which is what makes the
    computed self-field comparable with the analytic result.
    """
    operator = helmholtz_operator(grid, omega, eps)
    rhs = np.zeros(grid.nx * grid.ny, dtype=complex)
    rhs[np.ravel_multi_index(source_index, grid.shape)] = -1.0 / grid.h ** 2
    return spla.spsolve(operator, rhs).reshape(grid.shape)


# ---------------------------------------------------------------------------
# Local density of states
# ---------------------------------------------------------------------------

#: Imaginary part of the two-dimensional Green function evaluated at the source.
HOMOGENEOUS_IM_G = 0.25


def purcell_factor(grid: Grid, omega: float, eps: np.ndarray,
                   source_index) -> float:
    """
    Purcell factor of an emitter at `source_index`.

    Normalised so that an emitter in unbounded vacuum returns unity.
    """
    ez = solve_ez(grid, omega, eps, source_index)
    return float(np.imag(ez[source_index]) / HOMOGENEOUS_IM_G)


def purcell_spectrum(grid: Grid, omegas, eps: np.ndarray,
                     source_index, progress=None) -> np.ndarray:
    """Purcell factor over a frequency sweep."""
    out = np.empty(len(omegas))
    for k, w in enumerate(omegas):
        out[k] = purcell_factor(grid, w, eps, source_index)
        if progress is not None:
            progress(k, len(omegas))
    return out


# ---------------------------------------------------------------------------
# Structure definition
# ---------------------------------------------------------------------------

def rod_positions(n_cells: int, d_intra: float, d_inter: float):
    """
    Centres of a dimerised chain of 2 * n_cells rods, centred on the origin.

    Rods alternate between the A and B sublattices; `d_intra` separates the two
    rods within a unit cell and `d_inter` separates neighbouring cells.
    """
    xs, x = [], 0.0
    for cell in range(n_cells):
        xs.append(x)                      # A rod
        x += d_intra
        xs.append(x)                      # B rod
        if cell < n_cells - 1:
            x += d_inter
    xs = np.asarray(xs)
    return xs - xs.mean()


def build_permittivity(grid: Grid, centres, radius: float,
                       eps_rod: float, eps_background: float = 1.0,
                       loss_b: float = 0.0) -> np.ndarray:
    """
    Permittivity of a chain of circular rods.

    Material loss is applied to the B sublattice only, the direct analogue of
    the sublattice-selective loss of the tight-binding model. Rod edges are
    antialiased by supersampling each cell, which removes the staircase noise
    that otherwise dominates the frequency dependence at moderate resolution.
    """
    eps = np.full(grid.shape, eps_background, dtype=complex)
    sub = np.array([-0.25, 0.25]) * grid.h
    ox, oy = np.meshgrid(sub, sub, indexing="ij")

    xg = grid.x[:, None, None, None] + ox[None, None, :, :]
    yg = grid.y[None, :, None, None] + oy[None, None, :, :]

    for k, cx in enumerate(centres):
        inside = ((xg - cx) ** 2 + yg ** 2) <= radius ** 2
        fill = inside.mean(axis=(2, 3))
        value = eps_rod + (1j * loss_b if k % 2 else 0.0)
        eps = eps * (1.0 - fill) + value * fill
    return eps


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def validate_homogeneous_medium(omega: float = 2 * np.pi * 0.30,
                                eps_background: float = 1.0,
                                resolutions=(20, 30, 40, 60)) -> dict:
    """
    Check the solver against the analytic two-dimensional Green function.

    The computed Purcell factor in a homogeneous medium must approach unity, and
    the radial profile of the field must follow (i/4) H_0^(1)(k r). Returns the
    error in both quantities at each resolution.
    """
    purcell, profile_error, spacings = [], [], []

    for res in resolutions:
        h = 1.0 / res
        n = int(round(6.0 / h))
        grid = Grid(nx=n, ny=n, h=h, pml_cells=max(int(round(1.0 / h)), 8))
        eps = np.full(grid.shape, eps_background, dtype=complex)
        src = (grid.nx // 2, grid.ny // 2)

        ez = solve_ez(grid, omega, eps, src)
        purcell.append(float(np.imag(ez[src]) / HOMOGENEOUS_IM_G))

        # Radial cut away from both the source singularity and the absorber.
        k = omega * np.sqrt(eps_background)
        i0, j0 = src
        offsets = np.arange(max(int(round(0.5 / h)), 3), int(round(2.0 / h)))
        r = offsets * h
        numeric = ez[i0 + offsets, j0]
        exact = 0.25j * hankel1(0, k * r)
        profile_error.append(float(np.max(np.abs(numeric - exact))
                                   / np.max(np.abs(exact))))
        spacings.append(h)

    return {"resolutions": list(resolutions), "spacings": spacings,
            "purcell": purcell,
            "purcell_error": [abs(p - 1.0) for p in purcell],
            "profile_error": profile_error}


# ---------------------------------------------------------------------------
# Photonic realisation of the tight-binding model
# ---------------------------------------------------------------------------

#: Geometry realising t2 / t1 = 1.5 through the evanescent-coupling calibration
#: of `nanophotonics.calibrate_cmt`: the inter-cell gap is the smaller of the
#: two, so the inter-cell coupling is the stronger one.
TOPOLOGICAL_GEOMETRY = dict(d_intra=0.6, d_inter=0.4)
TRIVIAL_GEOMETRY = dict(d_intra=0.4, d_inter=0.6)

#: Silicon-like rods in air, radius 0.15 a.
ROD_RADIUS = 0.15
ROD_PERMITTIVITY = 12.0

#: Frequency window of the photonic band gap for the geometry above, in a / lambda.
GAP_WINDOW = (0.295, 0.320)


def _chain_grid(resolution: int, n_cells: int) -> Grid:
    """Computational domain sized to hold the chain with room for the absorber."""
    h = 1.0 / resolution
    span_x = n_cells + 4.0
    span_y = 6.0
    return Grid(nx=int(round(span_x / h)), ny=int(round(span_y / h)), h=h,
                pml_cells=int(round(1.0 / h)))


def photonic_chain(n_cells: int, geometry: dict, loss_b: float,
                   resolution: int = 32):
    """
    Assemble the rod chain and return the grid, permittivity and probe indices.

    The emitter positions are the centres of the outermost and the central A
    rods, the two sites whose tight-binding counterparts define the edge-to-bulk
    ratio. Both lie on the lossless sublattice.
    """
    grid = _chain_grid(resolution, n_cells)
    centres = rod_positions(n_cells, geometry["d_intra"], geometry["d_inter"])
    eps = build_permittivity(grid, centres, ROD_RADIUS, ROD_PERMITTIVITY,
                             loss_b=loss_b)
    edge = grid.index(centres[0], 0.0)
    bulk = grid.index(centres[2 * (n_cells // 2)], 0.0)
    return grid, eps, edge, bulk, centres


def photonic_ldos_study(n_cells: int = 8, loss_b: float = 0.0,
                        resolution: int = 32, n_freq: int = 41,
                        freq_range=(0.275, 0.345), progress=None) -> dict:
    """
    Full-wave edge and bulk Purcell spectra for the topological chain and its
    trivial control.

    Returns the spectra, the gap-averaged enhancement for both phases, and the
    peak edge response. The trivial control is the same structure with the two
    gaps exchanged; it has the same period, hence the same bulk bands, and
    differs only in the termination, so any difference between the two is a
    boundary effect rather than a difference of bulk optical response.
    """
    freqs = np.linspace(*freq_range, n_freq)
    omegas = 2 * np.pi * freqs
    out = {"frequencies": freqs, "n_cells": n_cells, "loss_b": loss_b,
           "resolution": resolution}

    for name, geom in (("topological", TOPOLOGICAL_GEOMETRY),
                       ("trivial", TRIVIAL_GEOMETRY)):
        grid, eps, edge, bulk, _ = photonic_chain(n_cells, geom, loss_b, resolution)
        report = (lambda k, n, nm=name: progress(nm, k, n)) if progress else None
        f_edge = purcell_spectrum(grid, omegas, eps, edge, report)
        f_bulk = purcell_spectrum(grid, omegas, eps, bulk, report)

        in_gap = (freqs >= GAP_WINDOW[0]) & (freqs <= GAP_WINDOW[1])
        out[name] = {
            "edge": f_edge, "bulk": f_bulk,
            "gap_enhancement": float(np.trapezoid(f_edge[in_gap], freqs[in_gap])
                                     / np.trapezoid(f_bulk[in_gap], freqs[in_gap])),
            "peak_edge": float(f_edge.max()),
            "peak_frequency": float(freqs[int(f_edge.argmax())]),
        }
    return out


def photonic_loss_series(loss_values, n_cells: int = 8, resolution: int = 32,
                         n_freq: int = 25, freq_range=(0.280, 0.320)) -> dict:
    """
    Gap-averaged enhancement of the topological chain as material loss increases.

    Tests the tight-binding statement that sublattice loss attenuates the
    magnitude of the boundary response in a real dielectric structure.
    """
    freqs = np.linspace(*freq_range, n_freq)
    omegas = 2 * np.pi * freqs
    enhancement, peak = [], []

    for loss in loss_values:
        grid, eps, edge, bulk, _ = photonic_chain(n_cells, TOPOLOGICAL_GEOMETRY,
                                                  loss, resolution)
        f_edge = purcell_spectrum(grid, omegas, eps, edge)
        f_bulk = purcell_spectrum(grid, omegas, eps, bulk)
        enhancement.append(float(np.trapezoid(f_edge, freqs)
                                 / np.trapezoid(f_bulk, freqs)))
        peak.append(float(f_edge.max()))

    return {"loss_values": np.asarray(loss_values, float),
            "gap_enhancement": np.asarray(enhancement),
            "peak_edge": np.asarray(peak)}


def photonic_resolution_convergence(resolutions=(16, 20, 24, 32),
                                    n_cells: int = 8,
                                    frequency: float = 0.300) -> dict:
    """
    Edge and bulk Purcell factors at a fixed in-gap frequency versus grid
    resolution, establishing that the reported enhancement is converged.
    """
    omega = 2 * np.pi * frequency
    edge_vals, bulk_vals, ratio = [], [], []

    for res in resolutions:
        grid, eps, edge, bulk, _ = photonic_chain(n_cells, TOPOLOGICAL_GEOMETRY,
                                                  0.0, res)
        fe = purcell_factor(grid, omega, eps, edge)
        fb = purcell_factor(grid, omega, eps, bulk)
        edge_vals.append(fe)
        bulk_vals.append(fb)
        ratio.append(fe / fb)

    return {"resolutions": list(resolutions), "frequency": frequency,
            "edge": edge_vals, "bulk": bulk_vals, "ratio": ratio}
