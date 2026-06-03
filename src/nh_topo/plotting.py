"""
plotting.py
===========
Publication-quality figure functions. Each takes pre-computed data and returns
(fig, axes); saving is handled by `save_fig` (vector PDF + 300-dpi PNG).

Design: serif typography, colorblind-safe palette (Wong 2011), legends placed
outside the data region, fully-labelled axes with physical units.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

STYLE = {
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "legend.framealpha": 0.95, "legend.edgecolor": "0.75",
    "lines.linewidth": 1.8, "figure.dpi": 130, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.axisbelow": True,
}

# Wong (2011) colorblind-safe palette
C = {
    "edge": "#0072B2", "bulk": "#D55E00", "protected": "#009E73",
    "lossy": "#CC79A7", "neutral": "#555555", "accent": "#E69F00",
    "blue2": "#56B4E9", "trivial": "#E69F00", "topo": "#009E73",
}

_LEG_BELOW = dict(loc="upper center", bbox_to_anchor=(0.5, -0.22), borderaxespad=0.0)
_LEG_RIGHT = dict(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)


def apply_style():
    matplotlib.rcParams.update(STYLE)


def save_fig(fig, output_dir, name: str):
    p = Path(output_dir); p.mkdir(parents=True, exist_ok=True)
    fig.savefig(p / f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(p / f"{name}.png", dpi=300, bbox_inches="tight")


# ── NB01 theory ───────────────────────────────────────────────────────────────
def plot_complex_spectrum(evals, edge_info, N, gamma):
    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
    prot, lossy = edge_info["protected_idx"], edge_info["lossy_idx"]
    bulk = np.ones(len(evals), bool); bulk[list(prot) + list(lossy)] = False
    ax.scatter(evals[bulk].real, evals[bulk].imag, s=20, color=C["edge"],
               alpha=0.8, label="Bulk modes", zorder=3)
    if len(prot):
        ax.scatter(evals[prot].real, evals[prot].imag, s=150, color=C["protected"],
                   marker="*", edgecolor="k", linewidth=0.4,
                   label="Protected edge (A): Im$(E)\\approx0$", zorder=5)
    if len(lossy):
        ax.scatter(evals[lossy].real, evals[lossy].imag, s=90, color=C["lossy"],
                   marker="s", edgecolor="k", linewidth=0.4,
                   label="Lossy edge (B): Im$(E)\\approx-\\gamma$", zorder=5)
    ax.axhline(0, color="0.6", lw=0.7, ls="--"); ax.axhline(-gamma, color="0.6", lw=0.7, ls=":")
    ax.axvline(0, color="0.6", lw=0.7, ls="--")
    ax.set_xlabel(r"$\mathrm{Re}(E)\;[t_1]$"); ax.set_ylabel(r"$\mathrm{Im}(E)\;[t_1]$")
    ax.set_title(f"Complex spectrum — non-Hermitian SSH ($N={N}$, $\\gamma={gamma}\\,t_1$)")
    ax.legend(**_LEG_RIGHT)
    return fig, ax


def plot_edge_profile(bio_density, evecs, edge_info, N):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    prot = edge_info["protected_idx"]
    sites = np.arange(2 * N)
    ax = axes[0]
    if len(prot):
        idx = prot[0]
        dens = np.abs(evecs[:, idx]) ** 2; dens /= dens.sum()
        ax.bar(sites[0::2], dens[0::2], width=0.85, color=C["protected"],
               alpha=0.9, label="A sublattice (lossless)")
        ax.bar(sites[1::2], dens[1::2], width=0.85, color=C["lossy"],
               alpha=0.7, label="B sublattice (lossy)")
    ax.set_xlabel("Site index $i$"); ax.set_ylabel(r"$|\psi_i|^2$ (normalised)")
    ax.set_title("Protected edge-mode profile"); ax.set_yscale("log")
    ax.set_ylim(1e-8, 1); ax.legend(**_LEG_BELOW)
    ax = axes[1]
    lossy = edge_info["lossy_idx"]
    if len(lossy):
        idx = lossy[0]; dens = np.abs(evecs[:, idx]) ** 2; dens /= dens.sum()
        ax.bar(sites[0::2], dens[0::2], width=0.85, color=C["protected"], alpha=0.9,
               label="A sublattice (lossless)")
        ax.bar(sites[1::2], dens[1::2], width=0.85, color=C["lossy"], alpha=0.7,
               label="B sublattice (lossy)")
    ax.set_xlabel("Site index $i$"); ax.set_ylabel(r"$|\psi_i|^2$ (normalised)")
    ax.set_title("Lossy edge-mode profile"); ax.set_yscale("log")
    ax.set_ylim(1e-8, 1); ax.legend(**_LEG_BELOW)
    return fig, axes


# ── NB02 topology ─────────────────────────────────────────────────────────────
def plot_topology(t1, t2, r_scan, nu_scan, w0_scan, qk_cases):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(r_scan, nu_scan, lw=2, color=C["edge"])
    ax.axvline(1.0, color="k", ls="--", lw=1.2, label=r"transition $t_2/t_1=1$")
    ax.scatter([t2 / t1], [1], color=C["bulk"], s=120, zorder=5,
               label=f"operating point ({t2/t1:.2f})")
    ax.set_xlabel(r"$t_2/t_1$"); ax.set_ylabel(r"$|\nu|$")
    ax.set_yticks([0, 1]); ax.set_ylim(-0.15, 1.4)
    ax.set_title("Topological invariant"); ax.legend(**_LEG_BELOW)
    ax = axes[1]
    ax.plot(r_scan, w0_scan, lw=2, color=C["protected"])
    ax.axvline(1.0, color="k", ls="--", lw=1.2)
    ax.scatter([t2 / t1], [1 - (t1 / t2) ** 2], color=C["bulk"], s=120, zorder=5,
               label=fr"$W_0={1-(t1/t2)**2:.3f}$")
    ax.set_xlabel(r"$t_2/t_1$")
    ax.set_ylabel(r"edge weight $W_0 = 1-(t_1/t_2)^2$")
    ax.set_title("Edge spectral weight ($\\eta$-independent)"); ax.legend(**_LEG_BELOW)
    ax = axes[2]
    for label, color, q in qk_cases:
        ax.plot(q.real, q.imag, lw=1.8, color=color, label=label)
    ax.scatter([0], [0], color="k", s=55, zorder=5, label="origin")
    ax.axhline(0, color="0.7", lw=0.5); ax.axvline(0, color="0.7", lw=0.5)
    ax.set_xlabel(r"$\mathrm{Re}\,q(k)$"); ax.set_ylabel(r"$\mathrm{Im}\,q(k)$")
    ax.set_aspect("equal"); ax.set_title(r"$q(k)=t_1+t_2e^{-ik}$"); ax.legend(**_LEG_BELOW)
    return fig, axes


# ── NB03 bands ────────────────────────────────────────────────────────────────
def plot_bloch_bands(k, bands, gamma, t1, t2):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    ax = axes[0]
    ax.plot(k / np.pi, bands[:, 0].real, color=C["edge"], lw=1.8, label="band $-$")
    ax.plot(k / np.pi, bands[:, 1].real, color=C["bulk"], lw=1.8, ls="--", label="band $+$")
    ax.axhspan(-abs(t2 - t1), abs(t2 - t1), color="0.85", alpha=0.5,
               label=fr"Re-gap $|E|<|t_2-t_1|={abs(t2-t1):.1f}$")
    ax.set_xlabel(r"$k/\pi$"); ax.set_ylabel(r"$\mathrm{Re}(E)\;[t_1]$")
    ax.set_title("Bloch bands — real part"); ax.legend(**_LEG_BELOW)
    ax = axes[1]
    ax.plot(k / np.pi, bands[:, 0].imag, color=C["edge"], lw=1.8, label="band $-$")
    ax.plot(k / np.pi, bands[:, 1].imag, color=C["bulk"], lw=1.8, ls="--", label="band $+$")
    ax.axhline(-gamma / 2, color="k", ls=":", lw=1.2, label=r"$-\gamma/2$ (passive PT)")
    ax.set_xlabel(r"$k/\pi$"); ax.set_ylabel(r"$\mathrm{Im}(E)\;[t_1]$")
    ax.set_title("Bloch bands — imaginary part"); ax.legend(**_LEG_BELOW)
    return fig, axes


def plot_obc_spectrum(evals, left_w, N, gap_hw):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ax = axes[0]
    sc = ax.scatter(np.arange(len(evals)), evals.real, c=left_w, cmap="viridis",
                    s=26, vmin=0, vmax=0.6, zorder=3)
    fig.colorbar(sc, ax=ax, pad=0.02, aspect=30, label="left-edge weight")
    ax.axhspan(-gap_hw, gap_hw, color=C["accent"], alpha=0.2,
               label=fr"gap window $|E|<{gap_hw}\,t_1$")
    ax.set_xlabel("eigenstate index (sorted by Re$E$)")
    ax.set_ylabel(r"$\mathrm{Re}(E)\;[t_1]$")
    ax.set_title(f"OBC spectrum ($N={N}$)"); ax.legend(**_LEG_BELOW)
    ax = axes[1]
    sc2 = ax.scatter(evals.real, evals.imag, c=left_w, cmap="viridis", s=26, vmin=0, vmax=0.6)
    fig.colorbar(sc2, ax=ax, pad=0.02, aspect=30, label="left-edge weight")
    ax.set_xlabel(r"$\mathrm{Re}(E)\;[t_1]$"); ax.set_ylabel(r"$\mathrm{Im}(E)\;[t_1]$")
    ax.set_title("OBC complex spectrum")
    return fig, axes


def plot_finite_size(N_vals, enh, w0, gap_obc, xi, w0_analytic):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    N_vals = np.array(N_vals)
    ax = axes[0]
    ax.plot(N_vals, enh, "o-", color=C["edge"], ms=7)
    ax.axvline(3 * xi, color="0.5", ls="--", lw=1.2, label=fr"$N=3\xi\approx{3*xi:.0f}$")
    ax.set_xlabel("system size $N$"); ax.set_ylabel(r"enhancement $\mathcal{E}$")
    ax.set_title("LDOS enhancement vs $N$ (parity-safe bulk)"); ax.legend(**_LEG_BELOW)
    ax = axes[1]
    ax.plot(N_vals, w0, "s-", color=C["protected"], ms=7, label="numerical $W_0$")
    ax.axhline(w0_analytic, color="k", ls="--", lw=1.2,
               label=fr"analytic $1-(t_1/t_2)^2={w0_analytic:.3f}$")
    ax.set_xlabel("system size $N$"); ax.set_ylabel(r"edge weight $W_0$")
    ax.set_title(r"Edge weight vs $N$ ($\eta$-independent)"); ax.legend(**_LEG_BELOW)
    ax = axes[2]
    ax.plot(N_vals, gap_obc, "^-", color=C["bulk"], ms=7)
    ax.axhline(abs(1.2 - 0.8), color="k", ls=":", lw=1.0)
    ax.set_xlabel("system size $N$"); ax.set_ylabel(r"OBC band-edge gap $[t_1]$")
    ax.set_title("Band-edge gap vs $N$")
    return fig, axes


# ── NB04 nanophotonics ────────────────────────────────────────────────────────
def plot_cmt(x, a, centers, modes, cal, d_intra, d_inter):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ax = axes[0]
    n_show = min(4, len(modes))
    for n in range(n_show):
        col = C["protected"] if n % 2 == 0 else C["lossy"]
        ax.plot(x / a, modes[n], color=col, lw=1.4)
        ax.axvline(centers[n] / a, color="0.7", lw=0.6, ls=":")
    ax.set_xlim(centers[0] / a - 1, centers[n_show - 1] / a + 1)
    ax.set_xlabel(r"position $x/a$"); ax.set_ylabel("mode amplitude (a.u.)")
    ax.set_title(r"Evanescent modes $\sim e^{-|x-x_0|/L}$")
    ax.legend(handles=[Line2D([0], [0], color=C["protected"], label="A sites"),
                       Line2D([0], [0], color=C["lossy"], label="B sites")], **_LEG_BELOW)
    ax = axes[1]
    d = np.linspace(0.2, 1.0, 200)
    ax.plot(d, cal["t0"] * np.exp(-d / cal["L"]), color=C["edge"], lw=2,
            label=fr"$t(d)=t_0e^{{-d/L}},\ L={cal['L']:.3f}a$")
    ax.scatter([d_intra], [cal["t1_reproduced"]], color=C["protected"], s=110, zorder=5,
               label=fr"$t_1(d={d_intra:.2f}a)={cal['t1_reproduced']:.2f}$")
    ax.scatter([d_inter], [cal["t2_reproduced"]], color=C["bulk"], s=110, zorder=5,
               label=fr"$t_2(d={d_inter:.2f}a)={cal['t2_reproduced']:.2f}$")
    ax.set_xlabel(r"edge-to-edge gap $d/a$"); ax.set_ylabel(r"coupling $[t_1]$")
    ax.set_title(fr"Self-consistent CMT mapping ($t_2/t_1={cal['ratio_geom']:.2f}$)")
    ax.legend(**_LEG_BELOW)
    return fig, axes


# ── NB05 LDOS ─────────────────────────────────────────────────────────────────
def plot_ldos(omegas, rho_edge, rho_bulk, rho_surf, gap_hw, E_int, eta,
              edge_site, bulk_site):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(omegas, rho_edge, color=C["edge"], lw=1.8, label=f"edge (site {edge_site}, A)")
    ax.plot(omegas, rho_bulk, color=C["bulk"], lw=1.8, ls="--", label=f"bulk (site {bulk_site}, A)")
    ax.plot(omegas, rho_surf, color=C["protected"], lw=1.2, ls=":",
            label="semi-infinite edge (Sancho–Rubio)")
    ax.axvspan(-gap_hw, gap_hw, color=C["accent"], alpha=0.2, label="gap window")
    ax.set_xlabel(r"$\omega\;[t_1]$"); ax.set_ylabel(r"LDOS $\rho(\omega)\;[t_1^{-1}]$")
    ax.set_title(fr"Edge vs bulk LDOS ($\eta={eta}\,t_1$)"); ax.legend(**_LEG_BELOW)
    ax = axes[1]
    m = np.abs(omegas) <= gap_hw
    ax.fill_between(omegas[m], rho_edge[m], color=C["edge"], alpha=0.25)
    ax.plot(omegas[m], rho_edge[m], color=C["edge"], lw=1.8, label="edge")
    ax.fill_between(omegas[m], rho_bulk[m], color=C["bulk"], alpha=0.25)
    ax.plot(omegas[m], rho_bulk[m], color=C["bulk"], lw=1.8, ls="--", label="bulk")
    ax.set_xlabel(r"$\omega\;[t_1]$"); ax.set_ylabel(r"LDOS $\rho(\omega)\;[t_1^{-1}]$")
    ax.set_title(fr"Gap region — $\mathcal{{E}}={E_int:.1f}\times$ (at $\eta={eta}\,t_1$)")
    ax.legend(**_LEG_BELOW)
    return fig, axes


def plot_eta_scaling(etas, enh, eta_op):
    apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    ax.loglog(etas, enh, "o-", color=C["edge"], ms=6, label=r"$\mathcal{E}(\eta)$")
    ax.axvline(eta_op, color="0.5", ls="--", lw=1.2, label=fr"operating $\eta={eta_op}$")
    ax.set_xlabel(r"broadening $\eta\;[t_1]$"); ax.set_ylabel(r"enhancement $\mathcal{E}$")
    ax.set_title(r"Enhancement ratio is $\eta$-dependent (bulk in-gap LDOS $\to0$)")
    ax.legend(**_LEG_RIGHT)
    return fig, ax


def plot_ldos_spatial(rho, N, edge_site, bulk_site):
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4.4), constrained_layout=True)
    sites = np.arange(2 * N)
    ax.bar(sites[0::2], rho[0::2], width=0.85, color=C["protected"], alpha=0.9,
           label="A sublattice (lossless)")
    ax.bar(sites[1::2], rho[1::2], width=0.85, color=C["lossy"], alpha=0.7,
           label="B sublattice (lossy)")
    ax.axvline(edge_site, color=C["edge"], ls="--", lw=1.4, label=f"edge emitter (site {edge_site})")
    ax.axvline(bulk_site, color="0.4", ls="--", lw=1.4, label=f"bulk emitter (site {bulk_site})")
    ax.set_xlabel("site index $i$"); ax.set_ylabel(r"$\rho_i(\omega=0)\;[t_1^{-1}]$")
    ax.set_title(r"Spatial LDOS map at $\omega=0$"); ax.legend(**_LEG_RIGHT)
    return fig, ax


# ── NB06 disorder ─────────────────────────────────────────────────────────────
def plot_disorder_protection(W_vals, study):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)
    W = np.array(W_vals)
    topo = study["topological"]; triv = study["trivial"]

    def series(d, key, stat):
        return np.array([d[w][key][stat] for w in W_vals])

    ax = axes[0]
    ax.errorbar(W, series(topo["bond"], "disp", "mean"),
                yerr=series(topo["bond"], "disp", "std"), fmt="o-", color=C["protected"],
                capsize=4, label="bond (chiral-preserving)")
    ax.errorbar(W, series(topo["on_site"], "disp", "mean"),
                yerr=series(topo["on_site"], "disp", "std"), fmt="s--", color=C["bulk"],
                capsize=4, label="on-site (chiral-breaking)")
    ax.set_xlabel(r"disorder $W\;[t_1]$")
    ax.set_ylabel(r"$\langle|\mathrm{Re}\,E_{\rm edge}|\rangle\;[t_1]$")
    ax.set_title("Zero-mode pinning (protection diagnostic)"); ax.legend(**_LEG_BELOW)

    ax = axes[1]
    ax.errorbar(W, series(topo["bond"], "enh", "mean"), yerr=series(topo["bond"], "enh", "std"),
                fmt="o-", color=C["protected"], capsize=4, label="bond, topological")
    ax.errorbar(W, series(topo["on_site"], "enh", "mean"), yerr=series(topo["on_site"], "enh", "std"),
                fmt="s--", color=C["bulk"], capsize=4, label="on-site, topological")
    ax.plot(W, series(triv["bond"], "enh", "mean"), "^:", color=C["trivial"],
            label="bond, trivial (control)")
    ax.plot(W, series(triv["on_site"], "enh", "mean"), "v:", color=C["neutral"],
            label="on-site, trivial (control)")
    ax.axhline(1.0, color="0.6", ls=":", lw=1.0)
    ax.set_xlabel(r"disorder $W\;[t_1]$"); ax.set_ylabel(r"enhancement $\mathcal{E}$")
    ax.set_title("Enhancement robustness with trivial controls"); ax.legend(**_LEG_BELOW)

    ax = axes[2]
    bp = ax.boxplot([topo["on_site"][w]["enh_all"] for w in W_vals],
                    positions=W, widths=0.06, patch_artist=True,
                    medianprops=dict(color="k", lw=1.6), manage_ticks=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(C["blue2"]); patch.set_alpha(0.7)
    ax.axhline(1.0, color="0.6", ls=":", lw=1.0)
    ax.set_xlabel(r"disorder $W\;[t_1]$"); ax.set_ylabel(r"enhancement $\mathcal{E}$")
    n = len(topo["on_site"][W_vals[0]]["enh_all"])
    ax.set_title(f"On-site $\\mathcal{{E}}$ distribution ({n} realisations)")
    return fig, axes


# ── NB07 sweep ────────────────────────────────────────────────────────────────
def plot_sweeps(r_vals, g_vals, enh, w0, g_slices):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    ext = [r_vals[0], r_vals[-1], g_vals[0], g_vals[-1]]
    ax = axes[0]
    vmax = float(np.percentile(enh, 97))
    im = ax.imshow(enh, origin="lower", aspect="auto", extent=ext, cmap="magma",
                   vmin=0, vmax=vmax)
    ax.axvline(1.0, color="w", ls="--", lw=1.6)
    ax.set_xlabel(r"$t_2/t_1$"); ax.set_ylabel(r"$\gamma/t_1$")
    ax.set_title(r"Enhancement $\mathcal{E}$ (97th-pct clip)")
    fig.colorbar(im, ax=ax, pad=0.02, aspect=30, label=r"$\mathcal{E}$")
    ax = axes[1]
    im2 = ax.imshow(w0, origin="lower", aspect="auto", extent=ext, cmap="viridis", vmin=0, vmax=1)
    ax.axvline(1.0, color="w", ls="--", lw=1.6)
    ax.set_xlabel(r"$t_2/t_1$"); ax.set_ylabel(r"$\gamma/t_1$")
    ax.set_title(r"Edge weight $W_0$ ($\eta$-independent)")
    fig.colorbar(im2, ax=ax, pad=0.02, aspect=30, label=r"$W_0$")
    ax = axes[2]
    for g, color in g_slices:
        j = int(np.argmin(np.abs(g_vals - g)))
        ax.plot(r_vals, enh[j, :], lw=1.8, color=color, label=fr"$\gamma/t_1={g:.1f}$")
    ax.axvline(1.0, color="k", ls="--", lw=1.2); ax.axhline(1.0, color="0.6", ls=":", lw=1.0)
    ax.set_yscale("log"); ax.set_xlabel(r"$t_2/t_1$"); ax.set_ylabel(r"$\mathcal{E}$")
    ax.set_title(r"Loss monotonically suppresses $\mathcal{E}$"); ax.legend(**_LEG_BELOW)
    return fig, axes


# ── NB08 validation ───────────────────────────────────────────────────────────
def plot_validation(omegas, rho_gf, rho_ed, rho_surf, evals_obc, evals_pbc,
                    n_cells, intensity_A, xi, gap_hw):
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(omegas, rho_gf, color=C["edge"], lw=2, label="Green's function (solve)")
    ax.plot(omegas, rho_ed, color=C["bulk"], lw=1.4, ls="--", label="biorthogonal eig-decomp")
    ax.plot(omegas, rho_surf, color=C["protected"], lw=1.0, ls=":", label="semi-infinite (Sancho–Rubio)")
    ax.set_xlabel(r"$\omega\;[t_1]$"); ax.set_ylabel(r"$\rho(\omega)\;[t_1^{-1}]$")
    ax.set_title("Three independent LDOS algorithms agree"); ax.legend(**_LEG_BELOW)
    ax = axes[0, 1]
    ax.scatter(np.arange(len(evals_obc)), evals_obc, s=16, color=C["edge"], label="OBC")
    ax.scatter(np.arange(len(evals_pbc)), evals_pbc, s=16, color=C["bulk"], marker="x", label="PBC")
    ax.axhspan(-gap_hw, gap_hw, color=C["accent"], alpha=0.2, label="gap window")
    ax.set_xlabel("eigenstate index"); ax.set_ylabel(r"$\mathrm{Re}(E)\;[t_1]$")
    ax.set_title("OBC has edge modes; PBC does not"); ax.legend(**_LEG_BELOW)
    ax = axes[1, 0]
    ax.semilogy(n_cells, intensity_A, "o", color=C["edge"], ms=6, label=r"numerical $|\psi_A(n)|^2$")
    ax.semilogy(n_cells, np.exp(-2 * n_cells / xi), "--", color=C["bulk"], lw=1.8,
                label=fr"$e^{{-2n/\xi}},\ \xi={xi:.2f}$")
    ax.set_xlabel("unit cell $n$"); ax.set_ylabel(r"$|\psi_A(n)|^2$ (norm.)")
    ax.set_title("Edge-mode exponential decay"); ax.legend(**_LEG_RIGHT)
    ax = axes[1, 1]
    ax.semilogy(omegas, np.abs(rho_gf - rho_ed) + 1e-20, color=C["neutral"], lw=1.4,
                label="|GF $-$ eig-decomp|")
    ax.semilogy(omegas, np.abs(rho_gf - rho_surf) + 1e-20, color=C["protected"], lw=1.0, ls=":",
                label="|GF $-$ semi-infinite|")
    ax.set_xlabel(r"$\omega\;[t_1]$"); ax.set_ylabel("absolute difference")
    ax.set_title("Pairwise agreement"); ax.legend(**_LEG_BELOW)
    return fig, axes
