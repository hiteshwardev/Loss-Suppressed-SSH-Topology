"""
nh_topo — Non-Hermitian Topological Nanophotonics
==================================================
A Python package for studying LDOS enhancement via topological edge modes
in non-Hermitian SSH photonic lattices.

Submodules
----------
hamiltonian  : build_ssh, build_ssh_pbc, bloch_hamiltonian
topology     : winding_number, topological_invariant, compute_bloch_bands
ldos         : ldos_at, ldos_spectrum, ldos_spatial_map, enhancement_factor
spectrum     : diagonalize, biorthogonal_density, find_edge_modes
disorder     : run_disorder_sweep
nanophotonics: build_resonator_chain, extract_ssh_couplings
sweeps       : enhancement_map, finite_size_convergence
plotting     : all figure functions (see plotting.py)
config       : load_all, get_physics_params
"""

from .hamiltonian    import build_ssh, build_ssh_pbc, bloch_hamiltonian, verify_properties
from .topology       import winding_number, topological_invariant, localization_length, compute_bloch_bands, qk_trajectory
from .ldos           import ldos_at, ldos_spectrum, ldos_spatial_map, integrated_ldos, enhancement_factor, ldos_eigdecomp
from .spectrum       import diagonalize, biorthogonal_density, edge_localization_weights, find_edge_modes, fit_localization_length
from .disorder       import add_onsite_disorder, run_disorder_sweep
from .nanophotonics  import build_resonator_chain, extract_ssh_couplings
from .sweeps         import enhancement_map, finite_size_convergence
from .config         import load_all, get_physics_params, get_figure_dir
from . import plotting

__version__ = "1.0.0"
__all__ = [
    "build_ssh", "build_ssh_pbc", "bloch_hamiltonian", "verify_properties",
    "winding_number", "topological_invariant", "localization_length",
    "compute_bloch_bands", "qk_trajectory",
    "ldos_at", "ldos_spectrum", "ldos_spatial_map", "integrated_ldos",
    "enhancement_factor", "ldos_eigdecomp",
    "diagonalize", "biorthogonal_density", "edge_localization_weights",
    "find_edge_modes", "fit_localization_length",
    "add_onsite_disorder", "run_disorder_sweep",
    "build_resonator_chain", "extract_ssh_couplings",
    "enhancement_map", "finite_size_convergence",
    "load_all", "get_physics_params", "get_figure_dir",
    "plotting",
]
