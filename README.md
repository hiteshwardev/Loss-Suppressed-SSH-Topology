# Sublattice-symmetry protection of the boundary LDOS in a lossy SSH lattice

Loss on one sublattice of a Su–Schrieffer–Heeger chain is usually treated as a nuisance that
degrades an otherwise topological boundary mode. For the *stability* of that mode the opposite
turns out to be true. This project identifies the symmetry responsible, works out its consequences,
and confirms the picture with a full-wave solution of Maxwell's equations in a dielectric structure.

## Results

**The chain is not chirally symmetric.** For `H = H₀ − iγP_B` the Hermitian relation
`σ_z H σ_z = −H` fails by exactly `2γ`. Once the mean decay rate is removed by a uniform imaginary
shift, the operator instead obeys the non-Hermitian sublattice symmetry

```
σ_z H̃ σ_z = −H̃†,        H̃ = H + i(γ/2)𝟙
```

which makes the spectrum symmetric about the imaginary axis rather than forcing eigenvalues into
± pairs. It pins the real energy of a self-conjugate mode and constrains its decay rate not at all.

**Loss suppresses finite-size hybridisation.** A mode can leave the imaginary axis only by pairing
with a partner at equal decay rate. Loss separates the two boundary modes in the imaginary
direction and removes that partner, closing the channel. Under identical bond-disorder ensembles at
N = 20 the Hermitian chain splits by 2–3 × 10⁻⁴ while the lossy chain stays at 2–4 × 10⁻¹⁶.

**The boundary doublet has its own exceptional point** at `γ = 2δ₀`, where `δ₀ ~ exp(−N/ξ)` is the
bare Hermitian splitting. At N = 10 this sits at 2.3 × 10⁻², some 35 times below the bulk
exceptional point at `2|t₂ − t₁|`. The two-level form `±√(δ₀² − γ²/4)` reproduces the full spectrum
to 7 × 10⁻¹⁴.

**Crossing it doubles the boundary spectral weight**, from `W₀/2` to `W₀ = 1 − (t₁/t₂)²`. The
measured ratio is 2.0000 for every chain length from N = 8 to 30 — a parameter-free signature that
needs no absolute calibration to observe.

**A full-wave calculation confirms it.** In a chain of silicon-like rods the gap-averaged boundary
enhancement is 7.88 in the topological geometry against 0.69 in the trivial control, and material
loss attenuates it by 2.7×, the same trade-off the lattice model predicts.

## Running the study

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebook.ipynb
```

`notebook.ipynb` *is* the study. Running it from top to bottom performs every calculation, writes
every figure to `figures/` in vector and raster form, stores all numerical results in
`results/results.json`, and prints a consolidated summary at the end. It takes roughly twelve
minutes on a single core, most of that in the disorder ensembles and the full-wave sweeps.

To check the implementation:

```bash
pytest                      # 72 tests
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

## Notes on method

**Mode identification.** The disorder diagnostic tracks the boundary mode by overlap with the clean
mode rather than by proximity to zero energy, so measuring that mode's energy is not circular. A
fixed boundary-weight threshold fails in two opposite ways: at strong disorder a stretched protected
mode drops below it while band-edge states rise above it, and a hybridised doublet shares its weight
between both ends so each end carries only half of `W₀`. Both behaviours are covered by regression
tests.

**Controls.** Every robustness claim is paired with a trivial-phase control, because a quantity can
be insensitive to disorder simply because a gap is large. The control has its own limits: beyond
about `W = 0.4 t₁` strong bond disorder begins to localise states near the boundary by accident, so
the anomaly rate is measured rather than assumed.

**Figures.** The layout engine measures every element with the renderer and shrinks the axes until
legends fit, then audits the result and raises if anything is clipped or overlapping. Nothing but
data appears inside the plot box.

**Reproducibility.** Every stochastic study is driven by one seed defined at the top of the
notebook. All linear algebra is either dense on matrices of at most 120 × 120 or a sparse direct
solve, so no iterative tolerances enter anywhere.

## Citation

If this code is useful in your work, please cite the accompanying paper.
Hitesh Kumar Singh, Department of Physics, MNS Government College, Bhiwani, Haryana, India.

## License

MIT — see `LICENSE`.
