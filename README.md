# Loss-Suppressed Hybridization and Symmetry Protection of the Boundary Local Density of States in a Non-Hermitian Su-Schrieffer-Heeger Lattice

**Author:** Hitesh Kumar Singh · [ORCID: 0009-0008-1583-4848](https://orcid.org/0009-0008-1583-4848)
*Department of Physics, Kurukshetra University, Kurukshetra, Haryana, India*

---

## Abstract

In a finite Su-Schrieffer-Heeger (SSH) chain, topological boundary modes hybridize and split due to finite-size effects. This work investigates how sublattice-selective loss suppresses this hybridization and identifies the symmetry that protects the boundary-state energy. The key result is an exact reduction: for arbitrary real bond configurations, the shifted Hamiltonian H + iγ/2 satisfies H̃² = H₀² − γ²/4. Each pair of Hermitian eigenvalues ±s becomes −iγ/2 ± √(s² − γ²/4), giving a boundary exceptional point at γ = 2δ₀, where δ₀ is the finite-size hybridization splitting. The symmetry σ<sub>z</sub> H σ<sub>z</sub> = −H† pins the real energy of an unpaired mode and explains its robustness against real bond disorder, whereas on-site disorder breaks the protection. For a chain of ten unit cells, the boundary exceptional point lies 34.5 times below the bulk exceptional point. Crossing it doubles the boundary spectral weight, with numerical agreement within 2 × 10⁻⁴ for chains of 8–30 unit cells. Finite-difference frequency-domain simulations of a dielectric-rod chain show a gap-averaged boundary Purcell enhancement of 7.88 in the topological geometry, compared with 0.69 in a trivial control; sublattice-selective material loss reduces it to 1.62.

## Core Objectives

- Explain why and how sublattice-selective loss suppresses finite-size hybridization of SSH boundary modes.
- Identify the symmetry that protects the boundary-mode energy in a lossy chain and test it against symmetry-preserving and symmetry-breaking disorder.
- Connect the boundary physics to a measurable observable, the local density of states (LDOS) and emitter Purcell enhancement, in a full-wave photonic structure.

## Scientific Significance

Absorption and radiation are unavoidable in photonic structures, so effective Hamiltonians are non-Hermitian. Recovery of zero modes in finite non-Hermitian lattices has been observed, but what protects them under disorder, and what the recovery does to an emitter's response, was less settled. This project gives an exact, realization-by-realization description that does not rely on PT symmetry or perturbation theory, and links it to emission signatures relevant to photonic design.

## Methodological Approach

- **Analytical:** symmetry classification (the chiral symmetry σ<sub>z</sub> H σ<sub>z</sub> = −H† of the non-Hermitian classification), and an exact reduction to two-level blocks via the shifted Hamiltonian.
- **Tight-binding numerics:** LDOS computed three ways (direct resolvent, biorthogonal eigendecomposition, independent semi-infinite surface Green function), with 200 seeded realizations per disorder strength and trivial-phase controls whose reliability is measured.
- **Full-wave validation:** a 2D TM finite-difference frequency-domain solver with complex coordinate stretching, validated against the Hankel-function Green function with second-order grid convergence.

## Contributions

1. An exact (not merely effective) two-level description, valid for any real bond configuration, that places the boundary exceptional point at γ = 2δ₀ in each disordered realization.
2. A symmetry explanation of the pinning (chiral symmetry in the non-Hermitian sense), separating bond disorder (preserves it) from on-site disorder (breaks it).
3. A parameter-free LDOS signature of the transition: the boundary spectral weight doubles from W₀/2 to W₀ = 1 − (t₁/t₂)².
4. A full-wave demonstration of the topological boundary enhancement, its absence in a trivial control and its attenuation by sublattice loss.

5. **Positioning relative to prior work.** An effective two-state description of the edge-state exceptional point in PT-symmetric SSH chains is already established (e.g. arXiv:2204.06932; arXiv:2511.13088). The contribution here is the exact reduction for arbitrary real bond configurations, its application to a passive chain with loss on one sublattice, the symmetry analysis of the pinning, the LDOS weight doubling and the full-wave validation.

## Key Findings

- Hybridization splitting is suppressed from about 10⁻⁴ t̄ to machine precision (≈ 10⁻¹⁶ t̄) under bond disorder; the boundary exceptional point lies 34.5 times below the bulk one at ten unit cells.
- Real energy displacement stays at machine zero under bond disorder up to W = t̄ and grows from 0.0150 t̄ to 0.2145 t̄ under on-site disorder.
- The edge-to-bulk LDOS enhancement falls by a factor of 6.09 between γ = 0 and the operating point (a broadening-dependent measure, stated at η = 0.05 t̄).
- Full-wave gap-averaged edge-to-bulk Purcell enhancement: 7.88 (topological), 0.69 (trivial), 1.62 with strong sublattice loss.

## Scope and Limitations

Most results use a tight-binding model. The full-wave calculation does not resolve the edge exceptional point and is compared with the lattice model qualitatively. Site-dependent loss, nonreciprocity and two dimensions are open questions stated in the manuscript. The Supplemental Material contains the convergence and verification studies.

## Roles

Hitesh Kumar Singh is the sole author. Hitesh Kumar Singh conceived the study, derived the exact reduction and symmetry analysis, implemented the numerical and full-wave calculations, carried out the validation and wrote the manuscript.

## Reproducibility and Availability

The repository contains a notebook that reproduces every figure and numerical result. The manuscript reports 78 automated tests that encode its claims.
