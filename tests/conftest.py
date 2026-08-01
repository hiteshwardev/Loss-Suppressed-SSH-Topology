"""Shared fixtures: put src/ on the path and expose the operating point."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(scope="session")
def params():
    """Operating point used throughout the manuscript."""
    return dict(N=20, t1=0.8, t2=1.2, gamma=0.3, eta=0.05, gap_hw=0.15)


@pytest.fixture(scope="session")
def H(params):
    import nh_topo as nt
    return nt.build_ssh(params["N"], params["t1"], params["t2"], params["gamma"])
