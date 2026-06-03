# Topological LDOS Enhancement in a Non-Hermitian (Lossy) SSH Lattice

A fully reproducible Python research project on how the **topological edge mode**
of a one-dimensional SSH photonic lattice with **sublattice-selective loss**
enhances the **local density of optical states (LDOS)** at the boundary — and on
the **symmetry conditions** under which that enhancement is genuinely
*topologically protected* against disorder.

> **Honesty statement.** This is a tight-binding / coupled-mode-theory study. It
> does **not** solve Maxwell's equations and makes **no** claim of full-wave
> electromagnetic validation. Quantitative nanophotonic predictions (absolute
> Purcell factors, radiative rates) are explicitly out of scope. What *is*
> established is validated by three independent numerical methods.

---

## 1. Scientific motivation

Topological photonics is usually framed at the level of band structure and
protected eigenmodes. Realistic photonic systems are lossy, so a natural
question is whether topological features survive in a **physically measurable
optical observable** — the LDOS, which sets the spontaneous-emission
(Purcell) rate of an embedded emitter. We ask, and answer with controls:

1. Does the topological edge mode enhance the in-gap LDOS at the boundary? **Yes** — and a trivial-phase control shows the enhancement is topological in origin.
2. Is that enhancement *protected* against disorder? **Only against chiral-symmetric (bond) disorder**, not against chiral-breaking (on-site) disorder — exactly as the BDI symmetry class predicts.
3. What is the role of non-Hermitian loss? It **splits** the edge mode into a protected (lossless) and a lossy partner, and it **monotonically attenuates** the enhancement (the effect is maximal in the Hermitian limit).

## 2. Physical model

A finite SSH chain of `N` unit cells (sites A = even index, B = odd index):

```
H = Σ_n [ t1 |n,A⟩⟨n,B| + t2 |n+1,A⟩⟨n,B| + h.c. ]  −  i·γ Σ_n |n,B⟩⟨n,B|
```

* `t1` (intra-cell), `t2` (inter-cell): real, **reciprocal** hoppings.
* `−iγ`: loss on the **B sublattice only** (passive non-Hermiticity).
* Operating point: `t1 = 0.8`, `t2 = 1.2` (ratio 1.5 → topological), `γ = 0.3`.

Because the hopping is reciprocal, `H = Hᵀ` (complex-symmetric, **not** Hermitian),
the spectrum is passive (`Im E ∈ [−γ, 0]`), and there is **no non-Hermitian skin
effect**. The bulk exceptional point sits at `γ_EP = 2|t2−t1| = 0.8` (the
operating `γ = 0.3` is well inside the unbroken phase).

## 3. Mathematical framework

* **Topological invariant:** winding of `q(k) = t1 + t2 e^{−ik}`; `γ` does not
  enter `q`, so `|ν|` is the Hermitian SSH value. Bulk–edge correspondence holds
  (GBZ = BZ).
* **Biorthogonal formalism:** for complex-symmetric `H`, left = (right)ᵀ, so the
  inner product is the unconjugated bilinear form `Σ_i v_R[i,m] v_R[i,n]`.
* **LDOS:** retarded Green's function `Gᴿ = (ω + iη − H)⁻¹`,
  `ρ_i(ω) = −(1/π) Im Gᴿ_ii ≥ 0` (positivity proved in `src/nh_topo/ldos.py`).
* **η-independent topological observable:** the boundary spectral weight of the
  protected mode, `W₀ = 1 − (t1/t2)²` (analytic), benchmarked numerically.

## 4. Repository structure

```
config/         YAML parameters (lattice, simulation, plotting, global)
src/nh_topo/    the package — all physics lives here
  hamiltonian.py   builders, exceptional-point threshold
  topology.py      winding number, analytic edge weight, Bloch bands
  spectrum.py      diagonalisation, biorthogonal density, edge-mode finder
  ldos.py          Green's-function LDOS, η-independent edge weight, positivity
  analytics.py     independent benchmarks (semi-infinite surface GF, bulk LDOS)
  disorder.py      bond vs on-site disorder, trivial controls, statistics
  nanophotonics.py self-consistent coupled-mode-theory mapping
  sweeps.py        parameter maps, parity-safe finite-size scaling
  plotting.py      publication-quality figures
  config.py        configuration loader
notebooks/      00–08 thin drivers over the package (executed, with outputs)
figures/        generated PDF (vector) + 300-dpi PNG
VALIDATION.md   evidence table for every claim
```

## 5. Installation

```bash
git clone <repo-url> && cd Topological-and-Non-Hermitian-Nanophotonic-Lattices
python -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -r requirements.txt
```

Requires Python ≥ 3.10. Core dependencies: NumPy, SciPy, Matplotlib, PyYAML.

## 6. Figure index

| Figure | Content |
|---|---|
| `01_theory/01_complex_spectrum` | protected (A) vs lossy (B) edge modes in the complex plane |
| `01_theory/02_edge_profiles` | exponential edge-mode profiles per sublattice |
| `02_topology/01_invariant_edgeweight` | `|ν|`, analytic `W₀=1−(t1/t2)²`, `q(k)` winding |
| `03_bands/01_bloch_bands` | complex Bloch bands (Re, Im) |
| `03_bands/02_obc_spectrum` | OBC spectrum coloured by edge weight |
| `03_bands/03_finite_size` | **parity-safe** finite-size scaling (enhancement, `W₀`, gap) |
| `04_effective_mode/01_cmt_mapping` | self-consistent evanescent-coupling schematic |
| `05_ldos/01_ldos_spectra` | edge vs bulk LDOS + semi-infinite overlay |
| `05_ldos/02_eta_scaling` | explicit `η`-dependence of the enhancement ratio |
| `05_ldos/03_ldos_spatial` | spatial LDOS map at `ω = 0` |
| `06_disorder/01_protection` | **bond vs on-site protection + trivial controls** |
| `07_parameter_sweep/01_maps` | enhancement & `W₀` maps; loss suppresses `E` |
| `08_validation/01_cross_validation` | three independent LDOS algorithms agree |

## 7. Key results & their status

he lossy/passive SSH model is well established experimentally (e.g. Zeuner
*et al.* PRL 2015; Weimann *et al.* Nat. Mater. 2017). The **defensible
contribution** here is the *clean, control-equipped* demonstration that a
**measurable optical observable (LDOS/Purcell)** inherits the BDI chiral
protection in a **symmetry-class-specific** way — protected against bond
disorder, not on-site — with trivial-phase controls and three-way numerical
validation, plus an explicit `η`-independent topological order parameter `W₀`.

## License

MIT — see [`LICENSE`](LICENSE).
