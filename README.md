# Symmetry protection of the boundary LDOS in a lossy SSH lattice

Loss on one sublattice of a Su–Schrieffer–Heeger chain is usually treated as a nuisance that
degrades an otherwise topological boundary mode. For the *stability* of that mode the opposite
turns out to be true. This project identifies the symmetry responsible, works out its consequences,
and confirms the picture with a full-wave solution of Maxwell's equations in a dielectric structure.

## Results

Energies are measured in units of the mean hopping t̄ = (t₁ + t₂)/2; the operating point
t₁ = 0.8, t₂ = 1.2 (t₂/t₁ = 1.5), γ = 0.3, N = 20 sets t̄ = 1.

**The chain is not chirally symmetric in the Hermitian sense.** For `H = H₀ − iγP_B` the relation
`σ_z H σ_z = −H` fails by exactly `2γ`. The operator instead obeys `σ_z H σ_z = −H†`, the chiral
symmetry of the Kawabata–Shiozaki–Ueda–Sato classification (written CS† here), which makes the
spectrum symmetric about the imaginary axis rather than forcing eigenvalues into ± pairs. It pins
the real energy of a self-conjugate mode and constrains its decay rate not at all. Bond disorder
preserves it exactly; on-site disorder breaks it.

**Every eigenvalue follows a square-root law.** The shifted operator `H̃ = H + i(γ/2)𝟙 = H₀ + i(γ/2)σ_z`
is the balanced gain–loss chain, and because `H₀` anticommutes with `σ_z` for any real bond
configuration, `H̃² = H₀² − (γ/2)²𝟙` exactly. Every eigenvalue of `H` is therefore
`−iγ/2 ± √(ε² − γ²/4)` with `ε ≥ 0` an eigenvalue of `H₀`: the bulk exceptional point
(`ε = |t₂ − t₁|`) and the boundary-doublet exceptional point (`ε = δ₀`) are two instances of one
formula, and the two-level description of the doublet is exact for every bond-disordered chain.

**Loss suppresses finite-size hybridisation.** Once `γ > 2δ₀`, with `δ₀ ~ exp(−N/ξ)` the bare
Hermitian splitting, the doublet stays at `Re E = 0`. Under identical bond-disorder ensembles at
N = 20 the Hermitian chain splits by 2.0–2.8 × 10⁻⁴ t̄ while the lossy chain stays at 2–4 × 10⁻¹⁶ t̄.
At N = 10 the edge exceptional point sits at 2.3 × 10⁻² t̄, some 35 times below the bulk one.

**Crossing it doubles the boundary spectral weight**, from `W₀/2` to `W₀ = 1 − (t₁/t₂)²`. The
measured ratio is 2.0000 for every chain length from N = 8 to 30.

**A full-wave calculation confirms the boundary enhancement.** In a finite-difference
frequency-domain solution for a chain of silicon-like rods, the gap-averaged edge-to-bulk Purcell
ratio is 7.88 in the topological geometry against 0.69 in the trivial control, and material loss
on the B rods reduces it to 1.62 at Im ε = 2.

## Running the study

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebook.ipynb
```

`notebook.ipynb` *is* the study. Running it from top to bottom performs every calculation, writes
every figure to `figures/` in vector and raster form, stores all numerical results in
`results/results.json`, and prints a consolidated summary at the end. It takes roughly seven
minutes on a single core, most of that in the disorder ensembles and the full-wave sweeps.

To check the implementation:

```bash
pytest                      # 78 tests
pytest -m "not slow"        # skip the full-wave solves
```

## Layout

```
notebook.ipynb   the complete study: theory, methods, computation, results
src/nh_topo/     reusable routines called from the notebook
  hamiltonian.py      model construction
  symmetry.py         CS-dagger classification, two-level model, edge exceptional point
  spectrum.py         diagonalisation, mode identification, overlap tracking
  ldos.py             Green-function LDOS, positivity, spectral weight, sum rule
  analytics.py        semi-infinite surface Green function by iterative decimation
  topology.py         winding number and analytic edge properties
  disorder.py         disorder ensembles, hybridisation and control studies
  sweeps.py           parameter maps and finite-size scans
  nanophotonics.py    coupled-mode calibration of the lattice parameters
  electromagnetics.py finite-difference frequency-domain Maxwell solver
  figures.py          figure generation under an enforced layout contract
tests/           verification and regression suite
figures/         generated figures (vector PDF and 400 dpi PNG)
results/         generated results.json
```

## Figures and the manuscript

The file names in `figures/` follow the order of the notebook. In the manuscript they appear as
follows (S = Supplemental Material).

| File | Paper | File | Paper |
|---|---|---|---|
| fig01_symmetry_residual | Fig. S1 | fig14_displacement | Fig. 8 |
| fig02_complex_spectrum | Fig. 1 | fig15_enhancement_disorder | Fig. 9 |
| fig03_edge_profile | Fig. 2 | fig16_trivial_control_anomalies | Fig. S7 |
| fig04_localization_fit | Fig. S2 | fig17_enhancement_map | Fig. 10 |
| fig05_localization_convergence | Fig. S3 | fig18_gamma_slices | Fig. S8 |
| fig06_ldos_spectra | Fig. 3 | fig19_cross_validation | Fig. S9 |
| fig07_eta_scaling | Fig. S4 | fig20_obc_vs_pbc | Fig. S10 |
| fig08_edge_weight | Fig. S5 | fig21_sum_rule | Fig. S11 |
| fig09_hybridisation_vs_disorder | Fig. 4 | fig22_grid_convergence | Fig. S12 |
| fig10_hybridisation_vs_size | Fig. 5 | fig23_ensemble_convergence | Fig. S13 |
| fig11_delta0_scaling | Fig. S6 | fig24_fdfd_benchmark | Fig. S14 |
| fig12_edge_exceptional_point | Fig. 6 | fig25_photonic_spectra | Fig. 11 |
| fig13_edge_weight_vs_gamma | Fig. 7 | fig26_photonic_loss | Fig. 12 |

## Notes on method

**Mode identification.** The disorder diagnostic tracks the boundary mode by overlap with the clean
mode rather than by proximity to zero energy, so measuring that mode's energy is not circular. A
fixed boundary-weight threshold fails in two opposite ways: at strong disorder a stretched protected
mode drops below it while band-edge states rise above it, and a hybridised doublet shares its weight
between both ends so each end carries only half of `W₀`. Both behaviours are covered by regression
tests.

**Controls.** Every robustness claim is paired with a trivial-phase control, because a quantity can
be insensitive to disorder simply because a gap is large. The control has its own limits: beyond
about `W = 0.4 t̄` strong bond disorder begins to localise states near the boundary by accident, so
the anomaly rate is measured rather than assumed.

**Figures.** The layout engine measures every element with the renderer and shrinks the axes until
legends fit, then audits the result and raises if anything is clipped or overlapping. Nothing but
data appears inside the plot box.

**Reproducibility.** Every stochastic study is driven by one seed defined at the top of the
notebook. All linear algebra is either dense on matrices of at most 120 × 120 or a sparse direct
solve, so no iterative tolerances enter anywhere.

## Citation

If this code is useful in your work, please cite the accompanying paper.
Hitesh Kumar Singh, Department of Physics, Kurukshetra University, Kurukshetra, Haryana, India.

## License

MIT; see `LICENSE`.
