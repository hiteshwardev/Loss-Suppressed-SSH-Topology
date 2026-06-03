"""
config.py
=========
YAML configuration loader for the NH Topo Nanophotonics project.
"""

import yaml
from pathlib import Path


_DEFAULT_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def load_all(config_dir=None):
    """
    Load all four configuration files.

    Parameters
    ----------
    config_dir : Path or str, optional
        Directory containing the YAML files. Defaults to <project_root>/config.

    Returns
    -------
    dict with keys: 'lattice', 'simulation', 'plotting', 'global'
    """
    d = Path(config_dir) if config_dir else _DEFAULT_CONFIG_DIR

    def _load(fname):
        with open(d / fname) as f:
            return yaml.safe_load(f)

    return {
        "lattice":     _load("lattice_params.yaml"),
        "simulation":  _load("simulation_params.yaml"),
        "plotting":    _load("plotting_params.yaml"),
        "global":      _load("global_config.yaml"),
    }


def get_physics_params(cfg: dict) -> dict:
    """
    Extract the core physics parameters into a flat dict for easy use in notebooks.
    """
    lc = cfg["lattice"]
    sc = cfg["simulation"]
    return {
        "N":       lc["lattice"]["num_unit_cells"],
        "t1":      lc["couplings"]["intra_cell"],
        "t2":      lc["couplings"]["inter_cell"],
        "gamma":   lc["non_hermiticity"]["sublattice_loss"]["strength"],
        "eta":     sc["ldos"]["eta"],
        "gap_hw":  sc["ldos"]["gap_integration_half_width"],
        "edge_site": sc["ldos"]["edge_site"],
        "bulk_site": sc["ldos"]["bulk_site"],
        "om_min":  sc["ldos"]["omega_min"],
        "om_max":  sc["ldos"]["omega_max"],
        "n_omega": sc["ldos"]["num_omega"],
        "a":       lc["geometry"]["lattice_constant"],
        "d_intra": lc["geometry"]["intra_cell_spacing"],
        "d_inter": lc["geometry"]["inter_cell_spacing"],
        "W_vals":  sc["disorder"]["disorder_strengths"],
        "n_real":  sc["disorder"]["num_realizations"],
        "seed":    sc["disorder"]["seed"],
        "disorder_classes": sc["disorder"]["classes"],
        "trivial_t2_over_t1": sc["disorder"]["trivial_t2_over_t1"],
        "N_vals_fs": sc["finite_size"]["N_vals"],
    }


def get_figure_dir(cfg: dict, notebook_key: str) -> str:
    """Return the output directory for a given notebook's figures."""
    return cfg["plotting"]["notebook_figures"][notebook_key]["output_dir"]
