# Loss-Suppressed Hybridisation and Sublattice-Symmetry Protection of the Boundary Local Density of States in a Non-Hermitian SSH Lattice

**Author**

**Hitesh Kumar Singh**  
Department of Physics  
Kurukshetra University  
Kurukshetra, Haryana, India

📧 **Email:** hiteshwardevthakur@gmail.com  
🔗 **ORCID:** https://orcid.org/0009-0008-1583-4848

---

Repository accompanying the research article:

> **Loss-suppressed hybridisation and sublattice-symmetry protection of the boundary local density of states in a non-Hermitian Su–Schrieffer–Heeger lattice**

---

## Overview

This repository contains the complete computational framework used in the accompanying study on passive non-Hermitian topology in the Su–Schrieffer–Heeger (SSH) model with sublattice-selective loss.

The work demonstrates that dissipation can *stabilize*, rather than degrade, topological boundary modes. By correctly classifying the system within the non-Hermitian **CS† symmetry class**, the study shows that sublattice loss suppresses finite-size hybridisation, produces an edge exceptional point, pins the real energy of the protected boundary mode, and doubles its measurable boundary spectral weight.

The repository is fully reproducible. Running the notebook from a clean environment regenerates every figure, table, and numerical result reported in the accompanying manuscript.

---

## Scientific Highlights

This repository reproduces the principal results of the paper:

- Correct classification of the passive lossy SSH chain using non-Hermitian **CS†** symmetry.
- Demonstration that Hermitian chiral symmetry fails by exactly **2γ**.
- Analytical and numerical description of the edge exceptional point.
- Loss-induced suppression of finite-size hybridisation.
- Exact pinning of the protected edge-mode real energy.
- Doubling of the measurable boundary spectral weight.
- Local density of states (LDOS) calculations using Green-function methods.
- Symmetry-resolved disorder studies.
- Full-wave electromagnetic validation using Maxwell's equations.
- Comprehensive numerical verification and regression testing.

---

# Repository Structure

```text
.
├── 00_LAB.ipynb
│   Complete executable study containing the theory, methods,
│   numerical computations, validation, analysis, and figure generation.
│
├── src/
│   └── nh_topo/
│       ├── hamiltonian.py
│       │   SSH model construction and non-Hermitian Hamiltonians.
│       │
│       ├── symmetry.py
│       │   CS† symmetry classification, effective two-level model,
│       │   and edge exceptional-point theory.
│       │
│       ├── spectrum.py
│       │   Eigenvalue calculations, edge-mode identification,
│       │   and overlap-based mode tracking.
│       │
│       ├── ldos.py
│       │   Green-function LDOS calculations, positivity,
│       │   spectral-weight evaluation, and sum-rule verification.
│       │
│       ├── analytics.py
│       │   Semi-infinite surface Green-function calculations
│       │   using iterative decimation.
│       │
│       ├── topology.py
│       │   Winding-number calculations and analytical
│       │   edge-state properties.
│       │
│       ├── disorder.py
│       │   Bond and on-site disorder ensembles,
│       │   hybridisation studies, and control analyses.
│       │
│       ├── sweeps.py
│       │   Parameter sweeps, finite-size scaling,
│       │   and phase-diagram generation.
│       │
│       ├── nanophotonics.py
│       │   Coupled-mode calibration linking the SSH lattice
│       │   to dielectric photonic structures.
│       │
│       ├── electromagnetics.py
│       │   Finite-Difference Frequency-Domain (FDFD)
│       │   Maxwell solver for full-wave validation.
│       │
│       └── figures.py
│           Publication-quality figure generation
│           under an enforced layout contract.
│
├── tests/
│   Automated verification and regression test suite.
│
├── figures/
│   Generated publication-quality figures
│   (vector PDF and 400 dpi PNG).
│
├── results/
│   Generated numerical outputs including
│   results.json containing all reported values.
│
├── requirements.txt
│   Python dependencies.
│
└── README.md
```

---

# Numerical Methods

The study combines several complementary computational approaches:

- Dense LAPACK eigensolvers
- Green-function calculations
- Biorthogonal eigendecomposition
- Semi-infinite iterative decimation
- Topological invariant calculations
- Statistical disorder averaging
- Finite-size scaling
- Coupled-mode theory
- Finite-Difference Frequency-Domain (FDFD) Maxwell simulations

---

# Validation

The repository includes an automated verification suite covering:

- Non-Hermitian CS† symmetry
- Spectrum pairing under **E → −E***
- Two-level model predictions
- Green-function consistency
- LDOS positivity
- Spectral sum rules
- Localisation length
- Hybridisation suppression
- Boundary spectral-weight doubling
- Numerical convergence

These tests reproduce the numerical tolerances reported in the manuscript.

---

# Reproducibility

Executing the notebook from beginning to end reproduces

- all manuscript figures,
- all numerical tables,
- all reported values,
- all validation metrics,
- `results/results.json`.

Every published number is generated directly from the code contained in this repository.

---

# Installation

Clone the repository

```bash
git clone https://github.com/hiteshwardev/Loss-Suppressed-SSH-Topology.git
```

Move into the project directory

```bash
cd Loss-Suppressed-SSH-Topology
```

Install the required Python packages

```bash
pip install -r requirements.txt
```
---

# Running the Study

Launch Jupyter Notebook

```bash
jupyter notebook
```

Open

```text
00_LAB.ipynb
```

Run every cell sequentially.

The notebook performs the complete workflow:

- theoretical derivations,
- model construction,
- spectrum calculations,
- topological analysis,
- LDOS calculations,
- disorder studies,
- parameter sweeps,
- finite-size scaling,
- full-wave electromagnetic simulations,
- validation tests,
- publication-quality figure generation,
- generation of `results/results.json`.

---

# Output

Running the notebook automatically generates

- Publication-quality figures (PDF and PNG)
- Numerical datasets
- Validation summaries
- `results/results.json`

---

# Requirements

- Python 3.11+
- NumPy
- SciPy
- Matplotlib
- Jupyter Notebook

Additional dependencies are listed in `requirements.txt`.

---

# Citation

If you use this repository in your research, please cite the accompanying paper:

> **Hitesh Kumar Singh**  
> *Loss-suppressed hybridisation and sublattice-symmetry protection of the boundary local density of states in a non-Hermitian Su–Schrieffer–Heeger lattice.*

(Journal and DOI will be added after publication.)

---

# License

This project is intended for academic and research use.

If this repository contributes to published work, please cite the accompanying article.

---

## Acknowledgements

The author gratefully acknowledges the open-source scientific Python ecosystem, including **NumPy**, **SciPy**, **Matplotlib**, and **Jupyter**, which made this research possible.
