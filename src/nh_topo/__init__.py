"""
nh_topo — Non-Hermitian Topological Nanophotonics
==================================================
A reproducible Python package for studying topological LDOS enhancement in a
non-Hermitian (lossy) SSH photonic lattice, and the symmetry conditions under
which that enhancement is protected against disorder.

Submodules
----------
hamiltonian  : build_ssh, build_ssh_general, build_ssh_pbc, bloch_hamiltonian,
               exceptional_point_threshold
topology     : winding_number, topological_invariant, localization_length,
               analytic_edge_weight, compute_bloch_bands
spectrum     : diagonalize, biorthogonal_density, find_edge_modes,
               fit_localization_length
ldos         : ldos_spectrum, enhancement_factor, central_bulk_site,
               edge_mode_spectral_weight, enhancement_vs_eta, ldos_eigdecomp,
               ldos_nonnegativity_min
analytics    : surface_green_function, surface_ldos, bulk_ldos  (independent benchmarks)
disorder     : make_disordered_ssh, run_disorder_study, protected_mode_displacement
nanophotonics: calibrate_cmt, evanescent_coupling, extract_ssh_couplings
sweeps       : enhancement_map, edge_weight_map, finite_size_convergence
plotting     : publication figure functions
config       : load_all, get_physics_params
"""

from .hamiltonian import (build_ssh, build_ssh_general, build_ssh_pbc,
                          bloch_hamiltonian, exceptional_point_threshold,
                          verify_properties)
from .topology import (winding_number, topological_invariant, localization_length,
                       analytic_edge_weight, compute_bloch_bands, qk_trajectory,
                       phase_boundary)
from .ldos import (ldos_at, ldos_spectrum, ldos_spatial_map, integrated_ldos,
                   enhancement_factor, central_bulk_site, edge_mode_spectral_weight,
                   enhancement_vs_eta, ldos_eigdecomp, ldos_nonnegativity_min)
from .analytics import (surface_green_function, surface_ldos,
                        surface_ldos_spectrum, bulk_ldos)
from .spectrum import (diagonalize, biorthogonal_density, edge_localization_weights,
                       find_edge_modes, fit_localization_length)
from .disorder import (add_onsite_disorder, make_disordered_ssh,
                       run_disorder_study, disorder_ensemble,
                       protected_mode_displacement)
from .nanophotonics import (calibrate_cmt, evanescent_coupling, mode_profile,
                            build_resonator_chain, extract_ssh_couplings)
from .sweeps import enhancement_map, edge_weight_map, finite_size_convergence
from .config import load_all, get_physics_params, get_figure_dir
from . import plotting

__version__ = "2.0.0"
__all__ = [
    "build_ssh", "build_ssh_general", "build_ssh_pbc", "bloch_hamiltonian",
    "exceptional_point_threshold", "verify_properties",
    "winding_number", "topological_invariant", "localization_length",
    "analytic_edge_weight", "compute_bloch_bands", "qk_trajectory", "phase_boundary",
    "ldos_at", "ldos_spectrum", "ldos_spatial_map", "integrated_ldos",
    "enhancement_factor", "central_bulk_site", "edge_mode_spectral_weight",
    "enhancement_vs_eta", "ldos_eigdecomp", "ldos_nonnegativity_min",
    "surface_green_function", "surface_ldos", "surface_ldos_spectrum", "bulk_ldos",
    "diagonalize", "biorthogonal_density", "edge_localization_weights",
    "find_edge_modes", "fit_localization_length",
    "add_onsite_disorder", "make_disordered_ssh", "run_disorder_study",
    "disorder_ensemble", "protected_mode_displacement",
    "calibrate_cmt", "evanescent_coupling", "mode_profile",
    "build_resonator_chain", "extract_ssh_couplings",
    "enhancement_map", "edge_weight_map", "finite_size_convergence",
    "load_all", "get_physics_params", "get_figure_dir", "plotting",
]
