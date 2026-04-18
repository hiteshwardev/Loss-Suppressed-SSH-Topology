import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# ── Style ───────────────────────────────────────────────────────────────────
STYLE = {
    "font.family":        "serif",
    "mathtext.fontset":   "dejavuserif",
    "font.size":          11,
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    9,
    "legend.framealpha":  0.95,
    "legend.edgecolor":   "0.75",
    "lines.linewidth":    1.8,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
}

COLORS = {
    "edge":        "#1f77b4",
    "bulk":        "#d62728",
    "topological": "#2ca02c",
    "trivial":     "#ff7f0e",
    "protected":   "#2ca02c",
    "lossy_edge":  "#9467bd",
    "bulk_mode":   "#1f77b4",
    "neutral":     "#7f7f7f",
}

# Legend anchored BELOW the axis (multi-panel figures)
_LEG_BELOW = dict(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  borderaxespad=0.0, framealpha=0.95)

# Legend anchored to the RIGHT of the axis (single-panel figures)
_LEG_RIGHT = dict(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  borderaxespad=0.0, framealpha=0.95)


def apply_style():
    matplotlib.rcParams.update(STYLE)


def save_fig(fig, output_dir, name: str):
    """Save as PDF (vector) and PNG (raster preview)."""
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    fig.savefig(p / f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(p / f"{name}.png", dpi=150, bbox_inches="tight")


# ══════════════════════════════════════════════════════════════════════════════
# NB01 – Theory
# ══════════════════════════════════════════════════════════════════════════════

def plot_complex_spectrum(evals, edge_info, N):
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)

    protected = edge_info["protected_idx"]
    lossy     = edge_info["lossy_idx"]
    bulk_mask = np.ones(len(evals), dtype=bool)
    bulk_mask[protected] = False
    bulk_mask[lossy]     = False

    ax.scatter(evals[bulk_mask].real, evals[bulk_mask].imag,
               s=15, color=COLORS["bulk_mode"], alpha=0.8,
               label="Bulk modes", zorder=3)
    if len(protected):
        ax.scatter(evals[protected].real, evals[protected].imag,
                   s=130, color=COLORS["protected"], marker="*",
                   label="Protected edge mode (A sublattice)", zorder=5)
    if len(lossy):
        ax.scatter(evals[lossy].real, evals[lossy].imag,
                   s=80, color=COLORS["lossy_edge"], marker="s",
                   label="Lossy edge mode (B sublattice)", zorder=5)

    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.axvline(0, color="gray", lw=0.7, ls="--")
    ax.set_xlabel(r"$\mathrm{Re}(E)\;[t_1]$")
    ax.set_ylabel(r"$\mathrm{Im}(E)\;[t_1]$")
    ax.set_title(f"Complex Spectrum — Non-Hermitian SSH ($N={N}$)")
    ax.legend(**_LEG_RIGHT)
    return fig, ax


def plot_edge_characterization(evals, evecs, left_w, right_w,
                               bio_density, edge_info, N):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.8), constrained_layout=True)

    protected = edge_info["protected_idx"]
    lossy     = edge_info["lossy_idx"]
    edge_bias = left_w - right_w
    bulk_mask = np.ones(len(evals), dtype=bool)
    bulk_mask[list(protected) + list(lossy)] = False

    ax = axes[0]
    ax.scatter(evals[bulk_mask].real, edge_bias[bulk_mask],
               s=12, color=COLORS["bulk_mode"], alpha=0.7, label="Bulk")
    if len(protected):
        ax.scatter(evals[protected].real, edge_bias[protected],
                   s=120, color=COLORS["protected"], marker="*",
                   zorder=5, label="Protected edge")
    if len(lossy):
        ax.scatter(evals[lossy].real, edge_bias[lossy],
                   s=80, color=COLORS["lossy_edge"], marker="s",
                   zorder=5, label="Lossy edge")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"$\mathrm{Re}(E)\;[t_1]$")
    ax.set_ylabel(r"$W_\mathrm{left} - W_\mathrm{right}$")
    ax.set_title("Boundary bias (biorthogonal)")
    ax.legend(ncol=1, **_LEG_BELOW)

    ax = axes[1]
    ax.scatter(left_w[bulk_mask], right_w[bulk_mask],
               s=12, color=COLORS["bulk_mode"], alpha=0.7, label="Bulk")
    if len(protected):
        ax.scatter(left_w[protected], right_w[protected],
                   s=120, color=COLORS["protected"], marker="*",
                   zorder=5, label="Protected edge")
    if len(lossy):
        ax.scatter(left_w[lossy], right_w[lossy],
                   s=80, color=COLORS["lossy_edge"], marker="s",
                   zorder=5, label="Lossy edge")
    ax.plot([0, 0.8], [0, 0.8], "k--", lw=0.8, label="Diagonal")
    ax.set_xlabel(r"$W_\mathrm{left}$ (biorthogonal)")
    ax.set_ylabel(r"$W_\mathrm{right}$ (biorthogonal)")
    ax.set_title("Left vs. right edge weight")
    ax.set_xlim(-0.05, 0.85); ax.set_ylim(-0.05, 0.85)
    ax.legend(ncol=2, **_LEG_BELOW)

    ax = axes[2]
    if len(protected):
        idx     = protected[0]
        density = np.real(bio_density[:, idx])
        density = np.maximum(density, 0)
        s = density.sum()
        if s > 0:
            density /= s
        ax.bar(np.arange(0, 2*N, 2), density[0::2],
               color=COLORS["protected"], alpha=0.85, width=0.8,
               label="A sublattice (lossless)")
        ax.bar(np.arange(1, 2*N, 2), density[1::2],
               color=COLORS["lossy_edge"], alpha=0.65, width=0.8,
               label="B sublattice (lossy)")
        ax.set_xlabel("Site index $i$")
        ax.set_ylabel(r"Biorthogonal density $\rho_i$")
        ax.set_title("Protected edge mode profile")
        ax.legend(ncol=2, **_LEG_BELOW)

    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
# NB02 – Topology
# ══════════════════════════════════════════════════════════════════════════════

def plot_phase_diagram(t1, t2, nu_current, t2_scan, nu_scan, qk_cases):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)

    ax = axes[0]
    ax.plot(t2_scan / t1, nu_scan, lw=2, color=COLORS["bulk_mode"],
            label=r"$|\nu|(t_2/t_1)$")
    ax.axvline(1.0, color="k", ls="--", lw=1.2,
               label=r"Phase boundary $t_2/t_1 = 1$")
    ax.scatter([t2 / t1], [nu_current], color=COLORS["bulk"], s=130,
               zorder=5, label=f"Operating point ($t_2/t_1={t2/t1:.2f}$)")
    ax.set_xlabel(r"$t_2/t_1$")
    ax.set_ylabel(r"Topological invariant $|\nu|$")
    ax.set_title("Topological Phase Diagram")
    ax.set_yticks([0, 1]); ax.set_ylim(-0.15, 1.45)
    ax.legend(ncol=1, **_LEG_BELOW)

    ax = axes[1]
    for label, color, q_vals in qk_cases:
        ax.plot(q_vals.real, q_vals.imag, lw=1.8, label=label, color=color)
    ax.scatter([0], [0], color="k", s=60, zorder=5, label="Origin")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"$\mathrm{Re}[q(k)]$")
    ax.set_ylabel(r"$\mathrm{Im}[q(k)]$")
    ax.set_title(r"Trajectory of $q(k) = t_1 + t_2 e^{-ik}$")
    ax.set_aspect("equal")
    ax.legend(ncol=2, **_LEG_BELOW)

    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
# NB03 – Band structures
# ══════════════════════════════════════════════════════════════════════════════

def plot_bloch_bands(k_vals, bands, gamma):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(k_vals / np.pi, bands[:, 0].real,
            color=COLORS["bulk_mode"], lw=1.8, label=r"Band $-$")
    ax.plot(k_vals / np.pi, bands[:, 1].real,
            color=COLORS["bulk_mode"], lw=1.8, ls="--", label=r"Band $+$")
    ax.set_xlabel(r"$k\,/\,\pi$")
    ax.set_ylabel(r"$\mathrm{Re}(E)\;[t_1]$")
    ax.set_title("Bloch Band Dispersion (Real Part)")
    ax.legend(ncol=2, **_LEG_BELOW)

    ax = axes[1]
    ax.plot(k_vals / np.pi, bands[:, 0].imag,
            color=COLORS["bulk"], lw=1.8, label=r"Band $-$ (Im)")
    ax.plot(k_vals / np.pi, bands[:, 1].imag,
            color=COLORS["lossy_edge"], lw=1.8, ls="--", label=r"Band $+$ (Im)")
    ax.axhline(-gamma, color="k", ls=":", lw=1.2,
               label=r"$-\gamma$ (B-site loss rate)")
    ax.axhline(0, color="gray", ls="-", lw=0.5)
    ax.set_xlabel(r"$k\,/\,\pi$")
    ax.set_ylabel(r"$\mathrm{Im}(E)\;[t_1]$")
    ax.set_title("Bloch Band Decay Rates (Imaginary Part)")
    ax.legend(ncol=1, **_LEG_BELOW)

    return fig, axes


def plot_obc_spectrum(evals_obc, left_w, N, gap_hw):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)

    ax = axes[0]
    sc = ax.scatter(np.arange(len(evals_obc)), evals_obc.real,
                    c=left_w, cmap="plasma", s=22, vmin=0, vmax=0.55, zorder=3)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, aspect=30)
    cbar.set_label(r"Left-edge weight $\sum_{i<4}|\psi_i|^2$")
    ax.axhspan(-gap_hw, gap_hw, color="gold", alpha=0.2,
               label=f"Bulk gap region $|\\omega| < {gap_hw}\\,t_1$")
    ax.set_xlabel("Eigenstate index (sorted by $\\mathrm{Re}(E)$)")
    ax.set_ylabel(r"$\mathrm{Re}(E)\;[t_1]$")
    ax.set_title(f"OBC Spectrum ($N={N}$) — colour = left-edge weight")
    ax.legend(ncol=1, **_LEG_BELOW)

    ax = axes[1]
    sc2 = ax.scatter(evals_obc.real, evals_obc.imag,
                     c=left_w, cmap="plasma", s=22, vmin=0, vmax=0.55)
    cbar2 = fig.colorbar(sc2, ax=ax, pad=0.02, aspect=30)
    cbar2.set_label("Left-edge weight")
    ax.set_xlabel(r"$\mathrm{Re}(E)\;[t_1]$")
    ax.set_ylabel(r"$\mathrm{Im}(E)\;[t_1]$")
    ax.set_title("OBC Complex Spectrum (colour = left-edge weight)")

    return fig, axes


def plot_finite_size_convergence(N_vals, enh_vals, gap_obc, xi_theory):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(N_vals, enh_vals, "o-", color=COLORS["edge"], lw=1.8,
            ms=7, label=r"Integrated LDOS enhancement $\mathcal{E}$")
    ax.axvline(3 * xi_theory, color="gray", ls="--", lw=1.2,
               label=rf"$N = 3\xi \approx {3*xi_theory:.0f}$ (convergence threshold)")
    ax.set_xlabel("Number of unit cells $N$")
    ax.set_ylabel(r"Enhancement $\mathcal{E}$")
    ax.set_title("Finite-Size Convergence of LDOS Enhancement")
    ax.legend(ncol=1, **_LEG_BELOW)

    ax = axes[1]
    ax.plot(N_vals, gap_obc, "s-", color=COLORS["topological"], lw=1.8,
            ms=7, label="OBC band-edge gap (numerical)")
    ax.set_xlabel("Number of unit cells $N$")
    ax.set_ylabel(r"Band-edge gap $[t_1]$")
    ax.set_title("Convergence of OBC Band Gap")
    ax.legend(ncol=1, **_LEG_BELOW)

    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
# NB04 – Nanophotonics
# ══════════════════════════════════════════════════════════════════════════════

def plot_coupling_extraction(x, a, centers, modes, coupling_data,
                             d_intra, d_inter):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    ax = axes[0]
    n_show = min(3, len(modes) // 2)
    blues  = plt.cm.Blues(np.linspace(0.5, 0.9, n_show))
    reds   = plt.cm.Reds(np.linspace(0.5, 0.9, n_show))
    for n in range(n_show):
        ax.plot(x / a, modes[2*n],     color=blues[n], lw=1.5)
        ax.plot(x / a, modes[2*n + 1], color=reds[n],  lw=1.5, ls="--")
    for c in centers[:2 * n_show]:
        ax.axvline(c / a, color="gray", lw=0.6, ls=":")
    ax.set_xlabel(r"Position $x / a$")
    ax.set_ylabel("Mode amplitude (normalised)")
    ax.set_title("Localised Gaussian Mode Profiles")
    ax.set_xlim(x[0] / a, centers[2 * n_show - 1] / a + 1)
    ax.legend(handles=[
        Line2D([0],[0], color="#4393c3", lw=1.5,
               label="A-sublattice sites (lossless)"),
        Line2D([0],[0], color="#d6604d", lw=1.5, ls="--",
               label="B-sublattice sites (lossy)"),
    ], ncol=1, **_LEG_BELOW)

    ax = axes[1]
    n_b  = min(10, len(coupling_data["t1_vals"]))
    x_t1 = np.arange(n_b) * 2
    x_t2 = np.arange(n_b) * 2 + 0.5
    ax.bar(x_t1, coupling_data["t1_vals"][:n_b], width=0.4,
           color=COLORS["edge"], alpha=0.85,
           label=rf"$t_1$ (intra-cell, $d_1={d_intra:.2f}a$)")
    ax.bar(x_t2, coupling_data["t2_vals"][:n_b], width=0.4,
           color=COLORS["bulk"], alpha=0.75,
           label=rf"$t_2$ (inter-cell, $d_2={d_inter:.2f}a$)")
    ax.axhline(coupling_data["t1_mean"], color=COLORS["edge"],
               ls="--", lw=1.2, label=r"$\langle t_1 \rangle$ (mean)")
    ax.axhline(coupling_data["t2_mean"], color=COLORS["bulk"],
               ls="--", lw=1.2, label=r"$\langle t_2 \rangle$ (mean)")
    ax.set_xlabel("Bond index")
    ax.set_ylabel("Normalised coupling strength")
    ax.set_title(f"SSH Coupling Structure ($t_2/t_1 = {coupling_data['ratio']:.3f}$)")
    ax.legend(ncol=2, **_LEG_BELOW)

    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
# NB05 – LDOS
# ══════════════════════════════════════════════════════════════════════════════

def plot_ldos_edge_vs_bulk(omegas, rho_edge, rho_bulk, gap_omegas,
                           rho_e_gap, rho_b_gap, E_int,
                           edge_site, bulk_site):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    ax = axes[0]
    ax.plot(omegas, rho_edge, color=COLORS["edge"], lw=1.8,
            label=f"Edge LDOS (site {edge_site}, A sublattice)")
    ax.plot(omegas, rho_bulk, color=COLORS["bulk"], lw=1.8, ls="--",
            label=f"Bulk LDOS (site {bulk_site}, A sublattice)")
    ax.axvspan(gap_omegas[0], gap_omegas[-1], color="gold",
               alpha=0.2, label="Gap integration window")
    ax.set_xlabel(r"Frequency $\omega\;[t_1]$")
    ax.set_ylabel(r"LDOS $\rho_i(\omega)\;[t_1^{-1}]$")
    ax.set_title("Edge vs. Bulk LDOS")
    ax.legend(ncol=1, **_LEG_BELOW)

    ax = axes[1]
    ax.fill_between(gap_omegas, rho_e_gap, alpha=0.25, color=COLORS["edge"])
    ax.plot(gap_omegas, rho_e_gap, color=COLORS["edge"], lw=1.8,
            label=f"Edge LDOS (site {edge_site})")
    ax.fill_between(gap_omegas, rho_b_gap, alpha=0.25, color=COLORS["bulk"])
    ax.plot(gap_omegas, rho_b_gap, color=COLORS["bulk"], lw=1.8, ls="--",
            label=f"Bulk LDOS (site {bulk_site})")
    ax.set_xlabel(r"Frequency $\omega\;[t_1]$")
    ax.set_ylabel(r"LDOS $\rho_i(\omega)\;[t_1^{-1}]$")
    ax.set_title(rf"Gap Region — Enhancement $\mathcal{{E}} = {E_int:.1f}\times$")
    ax.legend(ncol=2, **_LEG_BELOW)

    return fig, axes


def plot_ldos_spatial(rho_spatial, N, edge_site, bulk_site):
    apply_style()
    # Extra right margin for the 4-item legend
    fig, ax = plt.subplots(figsize=(10.0, 4.5), constrained_layout=True)

    ax.bar(np.arange(0, 2*N, 2), rho_spatial[0::2], width=0.8,
           color=COLORS["edge"], alpha=0.85, label="A sublattice (lossless)")
    ax.bar(np.arange(1, 2*N, 2), rho_spatial[1::2], width=0.8,
           color=COLORS["bulk"], alpha=0.70, label="B sublattice (lossy)")
    ax.axvline(edge_site - 0.5, color="#2ca02c", ls="--", lw=1.4,
               label=f"Edge emitter site (site {edge_site})")
    ax.axvline(bulk_site - 0.5, color="gray", ls="--", lw=1.4,
               label=f"Bulk emitter site (site {bulk_site})")
    ax.set_xlabel("Site index $i$")
    ax.set_ylabel(r"$\rho_i(\omega=0)\;[t_1^{-1}]$")
    ax.set_title(r"Spatial Map of LDOS at $\omega = 0$")
    ax.legend(**_LEG_RIGHT)

    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
# NB06 – Disorder
# ══════════════════════════════════════════════════════════════════════════════

def plot_disorder_averaged_ldos(omegas, results, W_vals, gap_hw):
    apply_style()
    n     = min(len(W_vals), 6)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5.5 * nrows),
                             constrained_layout=True)
    axes = np.array(axes).flatten()

    for ax, W in zip(axes[:n], W_vals[:n]):
        r = results[W]
        ax.plot(omegas, r["edge_avg"], color=COLORS["edge"], lw=1.8,
                label="Edge LDOS (disorder avg.)")
        ax.plot(omegas, r["bulk_avg"], color=COLORS["bulk"], lw=1.8,
                ls="--", label="Bulk LDOS (disorder avg.)")
        ax.axvspan(-gap_hw, gap_hw, color="gold", alpha=0.2,
                   label="Gap window")
        ax.set_title(rf"$W = {W:.1f}\,t_1$ — "
                     rf"$\langle\mathcal{{E}}\rangle = {r['enh_mean']:.1f}$")
        ax.set_xlabel(r"$\omega\;[t_1]$")
        ax.set_ylabel(r"LDOS (disorder avg.)")
        ax.legend(ncol=1, **_LEG_BELOW)

    for ax in axes[n:]:
        ax.set_visible(False)

    return fig, axes


def plot_enhancement_statistics(W_vals, results, gap_size):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    W_arr    = np.array(W_vals)
    enh_mean = np.array([results[W]["enh_mean"]   for W in W_vals])
    enh_std  = np.array([results[W]["enh_std"]    for W in W_vals])
    enh_med  = np.array([results[W]["enh_median"] for W in W_vals])

    ax = axes[0]
    ax.errorbar(W_arr, enh_mean, yerr=enh_std, fmt="o-",
                color=COLORS["edge"], lw=1.8, ms=7, capsize=5,
                label=r"Mean $\pm$ std")
    ax.plot(W_arr, enh_med, "s--", color=COLORS["topological"],
            lw=1.4, ms=6, label="Median")
    ax.axhline(1.0, color="gray", ls=":", lw=1.2,
               label=r"$\mathcal{E}=1$ (no enhancement)")
    ax.axvline(gap_size / 2, color="k", ls="--", lw=1.2,
               label=r"$W = \Delta/2$ (gap half-width)")
    ax.set_xlabel(r"Disorder strength $W\;[t_1]$")
    ax.set_ylabel(r"Enhancement $\mathcal{E}$")
    ax.set_title("LDOS Enhancement vs. Disorder Strength")
    ax.legend(ncol=2, **_LEG_BELOW)

    ax = axes[1]
    bplot = ax.boxplot(
        [results[W]["enh_all"] for W in W_vals],
        labels=[f"{W:.1f}" for W in W_vals],
        patch_artist=True,
        medianprops=dict(color="k", lw=2),
        whiskerprops=dict(lw=1.4),
        capprops=dict(lw=1.4),
        flierprops=dict(marker="o", ms=4, alpha=0.5),
    )
    for patch in bplot["boxes"]:
        patch.set_facecolor("#9ecae1")
        patch.set_alpha(0.85)
    ax.axhline(1.0, color="gray", ls=":", lw=1.2,
               label=r"$\mathcal{E}=1$ (no enhancement)")
    ax.set_xlabel(r"Disorder strength $W\;[t_1]$")
    ax.set_ylabel(r"Enhancement $\mathcal{E}$")
    ax.set_title(f"Distribution — {len(results[W_vals[0]]['enh_all'])} realizations")
    ax.legend(ncol=1, **_LEG_BELOW)

    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
# NB07 – Parameter sweep
# ══════════════════════════════════════════════════════════════════════════════

def plot_enhancement_map(r_vals, gamma_vals, enh, g_slices, r_slices):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), constrained_layout=True)

    ax = axes[0]
    vmax = float(np.percentile(enh, 97))
    im = ax.imshow(enh, origin="lower", aspect="auto",
                   extent=[r_vals[0], r_vals[-1],
                           gamma_vals[0], gamma_vals[-1]],
                   cmap="viridis", vmin=0, vmax=vmax)
    ax.axvline(1.0, color="w", ls="--", lw=1.8,
               label="Phase boundary $t_2/t_1 = 1$")
    ax.set_xlabel(r"Coupling ratio $t_2/t_1$")
    ax.set_ylabel(r"Loss rate $\gamma/t_1$")
    ax.set_title(r"LDOS Enhancement $\mathcal{E}(t_2/t_1,\,\gamma/t_1)$")
    cbar = fig.colorbar(im, ax=ax, pad=0.02, aspect=30)
    cbar.set_label(r"Enhancement $\mathcal{E}$")
    ax.legend(ncol=1, **_LEG_BELOW)

    ax = axes[1]
    for g, color in g_slices:
        idx = np.argmin(np.abs(gamma_vals - g))
        ax.plot(r_vals, enh[idx, :], lw=1.8,
                label=rf"$\gamma/t_1 = {g:.1f}$", color=color)
    ax.axvline(1.0, color="k", ls="--", lw=1.2,
               label="Phase boundary")
    ax.axhline(1.0, color="gray", ls=":", lw=1.0,
               label=r"$\mathcal{E}=1$")
    ax.set_xlabel(r"Coupling ratio $t_2/t_1$")
    ax.set_ylabel(r"Enhancement $\mathcal{E}$ (log scale)")
    ax.set_title(r"Enhancement — slices at fixed $\gamma/t_1$")
    ax.set_yscale("log"); ax.set_ylim(0.5, None)
    ax.legend(ncol=2, **_LEG_BELOW)

    ax = axes[2]
    for r, color in r_slices:
        jdx = np.argmin(np.abs(r_vals - r))
        ax.plot(gamma_vals, enh[:, jdx], lw=1.8,
                label=rf"$t_2/t_1 = {r:.1f}$", color=color)
    ax.axhline(1.0, color="gray", ls=":", lw=1.0,
               label=r"$\mathcal{E}=1$")
    ax.set_xlabel(r"Loss rate $\gamma/t_1$")
    ax.set_ylabel(r"Enhancement $\mathcal{E}$")
    ax.set_title(r"Enhancement — slices at fixed $t_2/t_1$")
    ax.legend(ncol=2, **_LEG_BELOW)

    return fig, axes


# ══════════════════════════════════════════════════════════════════════════════
# NB08 – Validation
# ══════════════════════════════════════════════════════════════════════════════

def plot_pbc_obc(evals_obc, evals_pbc, N, gap_hw):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)

    for ax, (evs, label, color) in zip(
            axes,
            [(evals_obc, "Open Boundary Conditions (OBC)", COLORS["edge"]),
             (evals_pbc, "Periodic Boundary Conditions (PBC)", COLORS["bulk"])]):
        ax.scatter(np.arange(len(evs)), evs, s=18, color=color,
                   alpha=0.8, label="Eigenvalues $\\mathrm{Re}(E_n)$")
        ax.axhspan(-gap_hw, gap_hw, color="gold", alpha=0.25,
                   label=f"Gap region $|\\omega| < {gap_hw}\\,t_1$")
        ax.set_title(f"{label} ($N={N}$)")
        ax.set_xlabel("Eigenstate index (sorted by $\\mathrm{Re}(E)$)")
        ax.set_ylabel(r"$\mathrm{Re}(E)\;[t_1]$")
        ax.legend(ncol=1, **_LEG_BELOW)

    return fig, axes


def plot_localization_length(n_cells, intensity_A, xi_theory):
    apply_style()
    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)

    fit_curve = np.exp(-2 * n_cells / xi_theory)
    ax.semilogy(n_cells, intensity_A, "o", color=COLORS["edge"],
                ms=7, label=r"Numerical $|\psi_A(n)|^2$ (A sublattice)")
    ax.semilogy(n_cells, fit_curve, "--", color=COLORS["bulk"], lw=1.8,
                label=(rf"Analytical $e^{{-2n/\xi}}$,"
                       rf" $\xi={xi_theory:.2f}$ unit cells"))
    ax.set_xlabel("Unit cell index $n$")
    ax.set_ylabel(r"$|\psi_A(n)|^2$ (normalised to $n=0$)")
    ax.set_title("Edge-Mode Exponential Spatial Decay")
    ax.legend(**_LEG_RIGHT)

    return fig, ax


def plot_gf_vs_eigdecomp(omegas, rho_gf, rho_ed, site):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.plot(omegas, rho_gf, color=COLORS["edge"], lw=1.8,
            label="Green's function LDOS")
    ax.plot(omegas, rho_ed, color=COLORS["bulk"], lw=1.5, ls="--",
            label="Eigendecomposition LDOS\n(complex-symmetric formula)")
    ax.set_xlabel(r"$\omega\;[t_1]$")
    ax.set_ylabel(r"LDOS $\rho(\omega)\;[t_1^{-1}]$")
    ax.set_title(f"GF vs. Eigendecomposition — site {site}")
    ax.legend(**_LEG_RIGHT)

    ax = axes[1]
    ax.semilogy(omegas, np.abs(rho_gf - rho_ed) + 1e-20,
                color=COLORS["neutral"], lw=1.5)
    ax.set_xlabel(r"$\omega\;[t_1]$")
    ax.set_ylabel(r"$|\rho_\mathrm{GF} - \rho_\mathrm{ED}|$")
    ax.set_title("Pointwise Absolute Agreement (machine precision)")

    return fig, axes
