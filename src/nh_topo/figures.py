"""
figures.py
==========
Publication-quality figure generation.

LAYOUT CONTRACT
---------------
Every figure produced here satisfies, and is automatically CHECKED against,
these rules:

  R1  One panel per figure. No multi-panel composites: each scientific
      statement gets a full-size canvas of its own.
  R2  The plotting box contains data only. Legends, colour bars, annotations
      and explanatory notes live in a reserved margin outside the axes.
  R3  Nothing is clipped. After drawing, the legend and every text artist are
      measured with the renderer; if the legend would overflow the canvas the
      axes are shrunk iteratively until it fits with a margin to spare.
  R4  Nothing overlaps. The measured bounding boxes of the legend, the axes and
      every margin note are tested pairwise for intersection.
  R5  Generous canvas (11 x 6.2 in) and large type, so figures stay legible at
      journal column width and on screen.

`save()` runs the audit and RAISES on violation, so a layout defect cannot
silently reach the manuscript. `scripts/check_figures.py` re-runs the same audit
across the whole figure set.

Output: vector PDF (fonts embedded, type 42) plus 400 dpi PNG, identical content.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# Wong colour-blind-safe palette (Nature Methods 8, 441 (2011))
C = {
    "edge":     "#0072B2",   # blue
    "bulk":     "#D55E00",   # vermillion
    "topo":     "#009E73",   # green
    "trivial":  "#E69F00",   # orange
    "surface":  "#CC79A7",   # purple
    "lossy":    "#56B4E9",   # sky blue
    "neutral":  "#4D4D4D",
    "grid":     "#DCDCDC",
    "analytic": "#000000",
}

FIG_W, FIG_H = 11.0, 6.2                      # generous single-panel canvas (in)
RECT_LEGEND = (0.105, 0.150, 0.515, 0.780)    # axes when a side legend is used
RECT_PLAIN = (0.105, 0.150, 0.860, 0.780)     # axes with no side legend
RECT_CBAR = (0.105, 0.150, 0.640, 0.780)      # axes when a colour bar is used

RIGHT_MARGIN = 0.020        # keep this fraction clear at the right edge
LEFT_MARGIN = 0.012         # keep this much clear at the left edge
BOTTOM_MARGIN = 0.012       # keep this much clear at the bottom
MIN_AXES_W = 0.28           # never shrink the plot box below this

_RC = {
    "figure.dpi": 110,
    "savefig.dpi": 400,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 15,
    "axes.labelsize": 17,
    "axes.titlesize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 1.1,
    "lines.linewidth": 2.4,
    "lines.markersize": 8.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 6.5,
    "ytick.major.size": 6.5,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "xtick.minor.size": 3.5,
    "ytick.minor.size": 3.5,
    "axes.grid": True,
    "grid.color": C["grid"],
    "grid.linewidth": 0.8,
    "grid.alpha": 0.9,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "#B0B0B0",
    "legend.borderpad": 0.6,
    "legend.labelspacing": 0.55,
    "legend.handlelength": 2.0,
    "savefig.bbox": "standard",     # fixed canvas; layout solved explicitly
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style():
    """Apply the project-wide Matplotlib style."""
    plt.rcParams.update(_RC)


# ---------------------------------------------------------------------------
# Layout engine
# ---------------------------------------------------------------------------

def _new(rect=RECT_LEGEND, w=FIG_W, h=FIG_H):
    """Figure with one axes at an explicit rectangle, leaving a right margin."""
    apply_style()
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes(rect)
    fig._margin_notes = []          # queued texts, stacked under the legend
    fig._cbar_ax = None
    return fig, ax


def _bbox(artist, fig):
    """Bounding box of an artist in figure coordinates."""
    return artist.get_window_extent().transformed(fig.transFigure.inverted())


def _legend_right(ax, title=None, ncol=1):
    """Place the legend in the reserved margin, outside and to the right."""
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.025, 1.0),
                    borderaxespad=0.0, ncol=ncol, title=title)
    if title:
        leg.get_title().set_fontsize(14)
    return leg


def _note(fig, text, color=None):
    """
    Queue an explanatory note for the right margin.

    Notes are stacked beneath the legend by `_solve_layout` using measured
    geometry, never placed by hand, so they cannot collide with the legend, with
    each other, or with the data.
    """
    fig._margin_notes.append((text, color or C["neutral"]))


def _solve_layout(fig, ax):
    """
    Make the legend fit, then stack the queued margin notes beneath it.

    Shrinks the axes until the legend's measured bounding box lies inside the
    canvas with RIGHT_MARGIN to spare (R3), then positions notes in the same
    column with measured spacing (R4).
    """
    leg = ax.get_legend()

    # --- left/bottom: make room for the axis labels and tick labels ---
    for _ in range(6):
        fig.canvas.draw()
        pos = ax.get_position()
        dx = dy = 0.0
        if ax.yaxis.label.get_text():
            b = _bbox(ax.yaxis.label, fig)
            dx = max(dx, LEFT_MARGIN - b.x0)
        for t in ax.get_yticklabels():
            if t.get_text():
                dx = max(dx, LEFT_MARGIN - _bbox(t, fig).x0)
        if ax.xaxis.label.get_text():
            b = _bbox(ax.xaxis.label, fig)
            dy = max(dy, BOTTOM_MARGIN - b.y0)
        if dx <= 1e-4 and dy <= 1e-4:
            break
        ax.set_position([pos.x0 + dx, pos.y0 + dy,
                         max(pos.width - dx, MIN_AXES_W),
                         max(pos.height - dy, 0.35)])
        if fig._cbar_ax is not None:
            p = ax.get_position()
            fig._cbar_ax.set_position([p.x1 + 0.022, p.y0, 0.026, p.height])

    # --- right: shrink until the legend fits inside the canvas ---
    for _ in range(10):
        fig.canvas.draw()
        if leg is None:
            break
        over = _bbox(leg, fig).x1 - (1.0 - RIGHT_MARGIN)
        if over <= 0:
            break
        pos = ax.get_position()
        new_w = max(pos.width - over - 0.008, MIN_AXES_W)
        if abs(new_w - pos.width) < 1e-4:
            break
        ax.set_position([pos.x0, pos.y0, new_w, pos.height])
        if fig._cbar_ax is not None:
            p = ax.get_position()
            fig._cbar_ax.set_position([p.x1 + 0.022, p.y0, 0.026, p.height])

    notes = getattr(fig, "_margin_notes", [])
    if not notes:
        return

    fig.canvas.draw()
    if leg is not None:
        lb = _bbox(leg, fig)
        x0, y = lb.x0, lb.y0 - 0.060
    else:
        pos = ax.get_position()
        x0, y = pos.x1 + 0.030, pos.y1 - 0.02

    for text, color in notes:
        t = fig.text(x0, y, text, fontsize=13, color=color,
                     va="top", ha="left", linespacing=1.45)
        fig.canvas.draw()
        y = _bbox(t, fig).y0 - 0.048


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------

def audit(fig, ax, name="figure"):
    """
    Verify the layout contract. Returns a list of problems (empty list = pass).

    Checks that every element lies inside the canvas, that the legend does not
    intrude on the plot box, and that margin notes clear both the legend and the
    plot box.
    """
    fig.canvas.draw()
    problems = []
    axb = _bbox(ax, fig)
    leg = ax.get_legend()

    def inside(b, what, pad=2e-3):
        if b.x0 < -pad or b.y0 < -pad or b.x1 > 1 + pad or b.y1 > 1 + pad:
            problems.append(
                f"{name}: {what} outside canvas "
                f"(x {b.x0:.3f}-{b.x1:.3f}, y {b.y0:.3f}-{b.y1:.3f})")

    def overlaps(a, b):
        return not (a.x1 <= b.x0 or b.x1 <= a.x0 or a.y1 <= b.y0 or b.y1 <= a.y0)

    boxes = []
    if leg is not None:
        lb = _bbox(leg, fig)
        inside(lb, "legend")
        if overlaps(lb, axb):
            problems.append(f"{name}: legend overlaps the plotting area")
        boxes.append(("legend", lb))

    for t in fig.texts:
        if not t.get_text().strip():
            continue
        tb = _bbox(t, fig)
        short = t.get_text().replace("\n", " ")[:24]
        inside(tb, f"note {short!r}")
        if overlaps(tb, axb):
            problems.append(f"{name}: note {short!r} overlaps the plotting area")
        for other, ob in boxes:
            if overlaps(tb, ob):
                problems.append(f"{name}: note {short!r} overlaps {other}")
        boxes.append((f"note {short!r}", tb))

    for lab, art in (("x-label", ax.xaxis.label), ("y-label", ax.yaxis.label)):
        if art.get_text():
            inside(_bbox(art, fig), lab)

    return problems


def save(fig, outdir, name, ax=None, strict=True):
    """Solve the layout, audit it, then write PDF + PNG. Raises on violation."""
    if ax is None:
        ax = fig.axes[0]
    _solve_layout(fig, ax)
    problems = audit(fig, ax, name)
    if problems:
        msg = f"layout audit failed for {name}:\n  " + "\n  ".join(problems)
        if strict:
            plt.close(fig)
            raise RuntimeError(msg)
        print("WARNING:", msg)

    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), format=ext)
    plt.close(fig)
    return os.path.join(outdir, f"{name}.pdf")


def _cbar(fig, ax, mappable, label):
    """Colour bar in the reserved right margin, clear of the plot box."""
    pos = ax.get_position()
    cax = fig.add_axes([pos.x1 + 0.022, pos.y0, 0.026, pos.height])
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label, fontsize=17, labelpad=12)
    cb.ax.tick_params(labelsize=13)
    fig._cbar_ax = cax
    return cb


# ---------------------------------------------------------------------------
# 1. Spectrum and eigenstates
# ---------------------------------------------------------------------------

def fig_complex_spectrum(evals, protected_idx, lossy_idx, gamma):
    """Complex eigenvalue plane: bulk cluster plus the split boundary doublet."""
    fig, ax = _new()
    mask = np.ones(evals.size, dtype=bool)
    mask[np.asarray(protected_idx, int)] = False
    mask[np.asarray(lossy_idx, int)] = False

    ax.scatter(evals[mask].real, evals[mask].imag, s=52, facecolor="white",
               edgecolor=C["neutral"], linewidth=1.2, label="bulk states", zorder=3)
    ax.scatter(evals[protected_idx].real, evals[protected_idx].imag, s=190,
               marker="o", color=C["edge"], edgecolor="black", linewidth=1.0,
               label=r"protected edge mode ($A$)", zorder=5)
    ax.scatter(evals[lossy_idx].real, evals[lossy_idx].imag, s=190,
               marker="s", color=C["bulk"], edgecolor="black", linewidth=1.0,
               label=r"lossy edge mode ($B$)", zorder=5)

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.55, zorder=1)
    ax.axhline(-gamma / 2, color=C["neutral"], linewidth=1.4, linestyle="--",
               alpha=0.85, zorder=1, label=r"$\mathrm{Im}\,E=-\gamma/2$ (bulk)")
    ax.axhline(-gamma, color=C["bulk"], linewidth=1.4, linestyle=":",
               alpha=0.9, zorder=1, label=r"$\mathrm{Im}\,E=-\gamma$")
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.35, zorder=1)

    ax.set_xlabel(r"$\mathrm{Re}\,E \;/\; t_1$")
    ax.set_ylabel(r"$\mathrm{Im}\,E \;/\; t_1$")
    ax.set_ylim(-1.20 * gamma, 0.26 * gamma)
    _legend_right(ax)
    return fig


def fig_edge_profile(psi_protected, psi_lossy, N):
    """Site-resolved intensity of the two boundary modes."""
    fig, ax = _new()
    sites = np.arange(2 * N)
    for psi, col, lab, mk in (
            (psi_protected, C["edge"], r"protected ($\mathrm{Im}\,E\simeq0$)", "o"),
            (psi_lossy, C["bulk"], r"lossy ($\mathrm{Im}\,E\simeq-\gamma$)", "s")):
        inten = np.abs(psi) ** 2
        inten = inten / inten.max()
        ax.semilogy(sites, np.maximum(inten, 1e-18), marker=mk, markersize=5.5,
                    color=col, linewidth=1.7, label=lab, alpha=0.92)
    ax.set_xlabel("site index")
    ax.set_ylabel(r"normalised intensity $|\psi|^2$")
    ax.set_ylim(1e-17, 4.0)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    ax.yaxis.set_minor_formatter(NullFormatter())
    _legend_right(ax)
    _note(fig, "even indices: $A$ sublattice\nodd indices: $B$ sublattice")
    return fig


def fig_localization_fit(intensity_A, xi_fit, xi_analytic):
    """Exponential decay of the protected mode on its host sublattice."""
    fig, ax = _new()
    n = np.arange(intensity_A.size)
    ax.semilogy(n, np.maximum(intensity_A, 1e-18), "o", color=C["edge"],
                markersize=9, label="numerical", zorder=4)
    nn = np.linspace(0, intensity_A.size - 1, 200)
    ax.semilogy(nn, np.exp(-2 * nn / xi_analytic), "-", color=C["analytic"],
                linewidth=2.0, label=rf"analytic, $\xi={xi_analytic:.3f}$", zorder=3)
    ax.semilogy(nn, np.exp(-2 * nn / xi_fit), "--", color=C["bulk"],
                linewidth=2.0, label=rf"fit, $\xi={xi_fit:.3f}$", zorder=3)
    ax.set_xlabel("unit cell $n$")
    ax.set_ylabel(r"$|\psi_A(n)|^2 / |\psi_A(0)|^2$")
    ax.set_ylim(1e-13, 4.0)
    _legend_right(ax)
    _note(fig, f"deviation: {abs(xi_fit - xi_analytic) / xi_analytic * 100:.2f}%")
    return fig


# ---------------------------------------------------------------------------
# 2. Topology
# ---------------------------------------------------------------------------

def fig_winding(r_scan, nu_scan):
    """Winding number across the topological transition."""
    fig, ax = _new()
    ax.plot(r_scan, np.abs(nu_scan), "-", color=C["topo"], linewidth=3.0,
            label=r"$|\nu|$")
    ax.axvline(1.0, color=C["neutral"], linestyle="--", linewidth=1.8,
               label=r"transition, $t_2=t_1$")
    ax.set_xlabel(r"$t_2/t_1$")
    ax.set_ylabel(r"winding number $|\nu|$")
    ax.set_ylim(-0.15, 1.25)
    ax.set_yticks([0, 1])
    _legend_right(ax)
    _note(fig, r"$t_2/t_1<1$: trivial", C["trivial"])
    _note(fig, r"$t_2/t_1>1$: topological", C["topo"])
    return fig


def fig_edge_weight(r_scan, w0_numeric, w0_analytic):
    """Broadening-independent boundary spectral weight across the transition."""
    fig, ax = _new()
    ax.plot(r_scan, w0_analytic, "-", color=C["analytic"], linewidth=2.4,
            label=r"analytic $1-(t_1/t_2)^2$")
    ax.plot(r_scan, w0_numeric, "o", color=C["edge"], markersize=7,
            markerfacecolor="white", markeredgewidth=1.6, label="numerical")
    ax.axvline(1.0, color=C["neutral"], linestyle="--", linewidth=1.8,
               label=r"transition, $t_2=t_1$")
    ax.set_xlabel(r"$t_2/t_1$")
    ax.set_ylabel(r"boundary spectral weight $W_0$")
    ax.set_ylim(-0.05, 0.85)
    _legend_right(ax)
    return fig


def fig_qk_trajectories(cases):
    """q(k) trajectories in the complex plane."""
    fig, ax = _new()
    for (label, q, col) in cases:
        ax.plot(q.real, q.imag, "-", color=col, linewidth=2.6, label=label)
    ax.plot(0, 0, "x", color="black", markersize=13, markeredgewidth=2.6,
            label="origin", zorder=5)
    ax.set_xlabel(r"$\mathrm{Re}\,q(k)$")
    ax.set_ylabel(r"$\mathrm{Im}\,q(k)$")
    ax.set_aspect("equal", adjustable="datalim")
    _legend_right(ax)
    _note(fig, "encircling the origin\n" r"$\Leftrightarrow$ topological")
    return fig


def fig_bloch_bands(k, bands, gamma):
    """Real part of the two Bloch bands over the Brillouin zone."""
    fig, ax = _new()
    ax.plot(k, bands[:, 0].real, "-", color=C["edge"], linewidth=2.6, label="lower band")
    ax.plot(k, bands[:, 1].real, "-", color=C["bulk"], linewidth=2.6, label="upper band")
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\mathrm{Re}\,E(k) \;/\; t_1$")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    _legend_right(ax)
    _note(fig, rf"$\gamma={gamma:g}\,t_1$" "\n"
               r"both bands share $\mathrm{Im}\,E=-\gamma/2$")
    return fig


def fig_obc_spectrum(evals, boundary_weight, gap_hw):
    """Open-boundary spectrum coloured by boundary localisation."""
    fig, ax = _new(RECT_CBAR)
    order = np.argsort(evals.real)
    sc = ax.scatter(np.arange(evals.size), evals[order].real,
                    c=boundary_weight[order], cmap="viridis", s=78,
                    edgecolor="black", linewidth=0.6, vmin=0, vmax=1, zorder=3)
    ax.axhspan(-gap_hw, gap_hw, color=C["grid"], alpha=0.7, zorder=1)
    ax.set_xlabel(r"eigenvalue index (sorted by $\mathrm{Re}\,E$)")
    ax.set_ylabel(r"$\mathrm{Re}\,E \;/\; t_1$")
    _cbar(fig, ax, sc, "boundary weight")
    return fig


# ---------------------------------------------------------------------------
# 3. LDOS
# ---------------------------------------------------------------------------

def fig_ldos_spectra(omegas, rho_edge, rho_bulk, rho_surface, gap_hw, eta, enh):
    """Edge, bulk and semi-infinite surface LDOS."""
    fig, ax = _new()
    ax.plot(omegas, rho_edge, "-", color=C["edge"], linewidth=2.8,
            label=r"edge site ($A$)")
    ax.plot(omegas, rho_bulk, "-", color=C["bulk"], linewidth=2.4,
            label=r"bulk site ($A$)")
    if rho_surface is not None:
        ax.plot(omegas, rho_surface, ":", color=C["surface"], linewidth=3.0,
                label="semi-infinite surface")
    ax.axvspan(-gap_hw, gap_hw, color=C["grid"], alpha=0.75, zorder=0,
               label="integration window")
    ax.set_xlabel(r"$\omega \;/\; t_1$")
    ax.set_ylabel(r"LDOS $\rho \;/\; t_1^{-1}$")
    _legend_right(ax)
    _note(fig, rf"$\eta={eta:g}\,t_1$" "\n" rf"$\mathcal{{E}}={enh:.2f}$")
    return fig


def fig_eta_scaling(etas, enh, eta_operating):
    """Broadening dependence of the integrated enhancement."""
    fig, ax = _new()
    ax.plot(etas, enh, "o-", color=C["edge"], linewidth=2.6, markersize=9,
            label=r"$\mathcal{E}(\eta)$")
    ax.axvline(eta_operating, color=C["neutral"], linestyle="--", linewidth=1.8,
               label=rf"operating $\eta={eta_operating:g}\,t_1$")
    ax.set_xscale("log")
    ax.set_xlabel(r"broadening $\eta \;/\; t_1$")
    ax.set_ylabel(r"enhancement $\mathcal{E}$")
    _legend_right(ax)
    _note(fig, "the ratio is not intrinsic:\nit must be quoted with $\\eta$")
    return fig


def fig_ldos_spatial(rho, N, edge_site, bulk_site):
    """LDOS across the chain at the edge-mode frequency."""
    fig, ax = _new()
    sites = np.arange(rho.size)
    colors = [C["edge"] if i % 2 == 0 else C["lossy"] for i in sites]
    ax.bar(sites, rho, color=colors, edgecolor="black", linewidth=0.5, width=0.84)
    ax.set_xlabel("site index")
    ax.set_ylabel(r"$\rho(\omega=0) \;/\; t_1^{-1}$")
    handles = [plt.Rectangle((0, 0), 1, 1, color=C["edge"]),
               plt.Rectangle((0, 0), 1, 1, color=C["lossy"])]
    ax.legend(handles, [r"$A$ sublattice (lossless)", r"$B$ sublattice (lossy)"],
              loc="upper left", bbox_to_anchor=(1.025, 1.0), borderaxespad=0.0)
    _note(fig, "weight concentrates on the\nlossless $A$ sublattice\nat the boundary")
    return fig


# ---------------------------------------------------------------------------
# 4. Finite size and parameter sweeps
# ---------------------------------------------------------------------------

def fig_finite_size_enhancement(N_vals, enh):
    """Enhancement versus chain length."""
    fig, ax = _new()
    ax.plot(N_vals, enh, "o-", color=C["edge"], linewidth=2.6, markersize=10,
            label=r"$\mathcal{E}(N)$")
    ax.set_xlabel("number of unit cells $N$")
    ax.set_ylabel(r"enhancement $\mathcal{E}$")
    _legend_right(ax)
    _note(fig, "converged once the chain\nexceeds a few $\\xi$")
    return fig


def fig_finite_size_weight(N_vals, w0, w0_analytic):
    """Boundary spectral weight versus chain length."""
    fig, ax = _new()
    ax.plot(N_vals, w0, "o-", color=C["edge"], linewidth=2.6, markersize=10,
            label="numerical")
    ax.axhline(w0_analytic, color=C["analytic"], linestyle="--", linewidth=2.2,
               label=rf"analytic $W_0={w0_analytic:.4f}$")
    ax.set_xlabel("number of unit cells $N$")
    ax.set_ylabel(r"boundary spectral weight $W_0$")
    _legend_right(ax)
    _note(fig, "parity-safe bulk reference\n(central $A$ site)")
    return fig


def fig_map(r_vals, g_vals, data, cbar_label, log_color=False):
    """2D parameter map with the colour bar outside the plotting region."""
    fig, ax = _new(RECT_CBAR)
    norm = matplotlib.colors.LogNorm(vmin=max(np.nanmin(data), 1e-3),
                                     vmax=np.nanmax(data)) if log_color else None
    im = ax.pcolormesh(r_vals, g_vals, data, cmap="viridis", shading="auto",
                       norm=norm, rasterized=True)
    ax.axvline(1.0, color="white", linestyle="--", linewidth=2.4)
    ax.set_xlabel(r"$t_2/t_1$")
    ax.set_ylabel(r"$\gamma \;/\; t_1$")
    _cbar(fig, ax, im, cbar_label)
    return fig


def fig_gamma_slices(g_vals, slices, r_labels):
    """Enhancement versus loss at several coupling ratios."""
    fig, ax = _new()
    palette = [C["edge"], C["topo"], C["bulk"], C["trivial"]]
    marks = ["o-", "s-", "^-", "d-"]
    for i, (curve, lab) in enumerate(zip(slices, r_labels)):
        ax.plot(g_vals, curve, marks[i % 4], color=palette[i % len(palette)],
                linewidth=2.4, markersize=7, label=rf"$t_2/t_1={lab}$")
    ax.set_xlabel(r"$\gamma \;/\; t_1$")
    ax.set_ylabel(r"enhancement $\mathcal{E}$")
    _legend_right(ax)
    _note(fig, "loss attenuates the\nresponse magnitude")
    return fig


# ---------------------------------------------------------------------------
# 5. Symmetry, disorder and hybridisation (central results)
# ---------------------------------------------------------------------------

def fig_symmetry_residual(W_vals, bond, on_site):
    """CS-dagger residual: bond disorder preserves it exactly, on-site breaks it."""
    fig, ax = _new()
    floor = 1e-18
    ax.semilogy(W_vals, np.maximum(on_site, floor), "s-", color=C["bulk"],
                linewidth=2.6, markersize=10,
                label=r"on-site (CS$^\dagger$ breaking)")
    ax.semilogy(W_vals, np.maximum(bond, floor), "o-", color=C["edge"],
                linewidth=2.6, markersize=10,
                label=r"bond (CS$^\dagger$ preserving)")
    ax.set_xlabel(r"disorder strength $W \;/\; t_1$")
    ax.set_ylabel(r"$\|\sigma_z\tilde H\sigma_z+\tilde H^\dagger\|_{\max}$")
    ax.set_ylim(floor / 10, 20)
    _legend_right(ax)
    _note(fig, "bond residual is exactly zero;\nplotted at the numerical floor",
          C["edge"])
    return fig


def fig_displacement(W_vals, bond_mean, bond_err, onsite_mean, onsite_err):
    """Boundary-mode displacement off the imaginary axis, by disorder class."""
    fig, ax = _new()
    floor = 1e-18
    ax.errorbar(W_vals, np.maximum(onsite_mean, floor), yerr=onsite_err,
                fmt="s-", color=C["bulk"], linewidth=2.6, markersize=10,
                capsize=5, capthick=1.6, label="on-site (symmetry breaking)")
    ax.errorbar(W_vals, np.maximum(bond_mean, floor), yerr=bond_err,
                fmt="o-", color=C["edge"], linewidth=2.6, markersize=10,
                capsize=5, capthick=1.6, label="bond (symmetry preserving)")
    ax.set_yscale("log")
    ax.set_xlabel(r"disorder strength $W \;/\; t_1$")
    ax.set_ylabel(r"$|\mathrm{Re}\,E|$ of boundary mode $\;/\; t_1$")
    ax.set_ylim(floor / 10, 2.0)
    _legend_right(ax)
    _note(fig, "error bars: 95% CI\n200 realisations per point")
    return fig


def fig_enhancement_disorder(W_vals, curves):
    """Enhancement under disorder with trivial-phase controls."""
    fig, ax = _new()
    style = {
        "bond (topological)":    (C["edge"], "o-"),
        "on-site (topological)": (C["bulk"], "s-"),
        "bond (trivial)":        (C["topo"], "o--"),
        "on-site (trivial)":     (C["trivial"], "s--"),
    }
    for label, (mean, err) in curves.items():
        col, fmt = style.get(label, (C["neutral"], "o-"))
        ax.errorbar(W_vals, mean, yerr=err, fmt=fmt, color=col, linewidth=2.4,
                    markersize=8, capsize=5, capthick=1.5, label=label)
    ax.set_xlabel(r"disorder strength $W \;/\; t_1$")
    ax.set_ylabel(r"enhancement $\mathcal{E}$")
    ax.set_yscale("log")
    _legend_right(ax)
    _note(fig, "trivial controls stay near unity:\nthe effect is topological,\nnot merely gapped")
    return fig


def fig_hybridisation_vs_disorder(W_vals, splitting, sem, gamma_vals):
    """Doublet splitting under identical bond-disorder ensembles."""
    fig, ax = _new()
    floor = 1e-18
    marks, cols = ["s-", "o-", "^-"], [C["bulk"], C["edge"], C["topo"]]
    for i, g in enumerate(gamma_vals):
        lab = (r"$\gamma=0$ (Hermitian)" if g == 0
               else rf"$\gamma={g:g}\,t_1$ (lossy)")
        yerr = None if sem is None else 1.96 * np.asarray(sem[i])
        ax.errorbar(W_vals, np.maximum(splitting[i], floor), yerr=yerr,
                    fmt=marks[i % 3], color=cols[i % 3], linewidth=2.6,
                    markersize=10, capsize=5, capthick=1.5, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel(r"bond disorder strength $W \;/\; t_1$")
    ax.set_ylabel(r"doublet splitting $\min|\mathrm{Re}\,E| \;/\; t_1$")
    ax.set_ylim(floor / 10, 1e-2)
    _legend_right(ax)
    _note(fig, "identical disorder ensembles\nin both curves")
    return fig


def fig_hybridisation_vs_size(N_vals, splitting, gamma_vals, xi=None):
    """Size scaling: Hermitian splitting decays exponentially, lossy stays pinned."""
    fig, ax = _new()
    floor = 1e-18
    marks, cols = ["s-", "o-"], [C["bulk"], C["edge"]]
    for i, g in enumerate(gamma_vals):
        lab = (r"$\gamma=0$ (Hermitian)" if g == 0
               else rf"$\gamma={g:g}\,t_1$ (lossy)")
        ax.semilogy(N_vals, np.maximum(splitting[i], floor), marks[i % 2],
                    color=cols[i % 2], linewidth=2.6, markersize=10, label=lab)
    if xi is not None:
        nn = np.linspace(min(N_vals), max(N_vals), 100)
        ref = splitting[0][0] * np.exp(-(nn - N_vals[0]) / xi)
        ax.semilogy(nn, ref, "--", color=C["analytic"], linewidth=1.8,
                    label=rf"$\propto e^{{-N/\xi}}$, $\xi={xi:.3f}$")
    ax.set_xlabel("number of unit cells $N$")
    ax.set_ylabel(r"doublet splitting $\min|\mathrm{Re}\,E| \;/\; t_1$")
    ax.set_ylim(floor / 10, 2.0)
    _legend_right(ax)
    return fig


def fig_edge_exceptional_point(gammas, observed, delta0, analytic=None):
    """The boundary doublet's exceptional point at gamma = 2*delta_0."""
    fig, ax = _new()
    floor = 1e-18
    ax.semilogy(gammas / delta0, np.maximum(observed, floor), "o",
                color=C["edge"], markersize=9, label="numerical", zorder=4)
    if analytic is not None:
        ax.semilogy(gammas / delta0, np.maximum(analytic, floor), "-",
                    color=C["analytic"], linewidth=2.2,
                    label=r"$\sqrt{\delta_0^2-\gamma^2/4}$", zorder=3)
    ax.axvline(2.0, color=C["bulk"], linestyle="--", linewidth=2.4,
               label=r"edge EP, $\gamma=2\delta_0$")
    ax.set_xlabel(r"$\gamma / \delta_0$")
    ax.set_ylabel(r"$\min|\mathrm{Re}\,E| \;/\; t_1$")
    ax.set_ylim(floor / 10, 20 * delta0)
    _legend_right(ax)
    _note(fig, r"$\gamma<2\delta_0$: hybridised")
    _note(fig, r"$\gamma>2\delta_0$: pinned", C["edge"])
    return fig


def fig_edge_weight_vs_gamma(gammas, w0, delta0, w0_analytic):
    """Boundary spectral weight doubling across the edge exceptional point."""
    fig, ax = _new()
    ax.semilogx(gammas / delta0, w0, "o-", color=C["edge"], linewidth=2.4,
                markersize=8, label="numerical")
    ax.axhline(w0_analytic, color=C["analytic"], linestyle="-", linewidth=2.0,
               label=rf"$W_0=1-(t_1/t_2)^2={w0_analytic:.4f}$")
    ax.axhline(w0_analytic / 2, color=C["neutral"], linestyle=":", linewidth=2.0,
               label=r"$W_0/2$ (hybridised)")
    ax.axvline(2.0, color=C["bulk"], linestyle="--", linewidth=2.4,
               label=r"edge EP, $\gamma=2\delta_0$")
    ax.set_xlabel(r"$\gamma / \delta_0$")
    ax.set_ylabel(r"boundary spectral weight $W_0$")
    ax.set_ylim(0, 1.28 * w0_analytic)
    _legend_right(ax)
    _note(fig, "the weight doubles\nacross the transition")
    return fig


def fig_ensemble_convergence(n_vals, running_mean, running_err, ylabel, n_used=200):
    """Convergence of an ensemble mean with the number of realisations."""
    fig, ax = _new()
    ax.errorbar(n_vals, running_mean, yerr=running_err, fmt="o-",
                color=C["edge"], linewidth=2.4, markersize=8, capsize=5,
                capthick=1.5, label="running mean")
    ax.axvline(n_used, color=C["neutral"], linestyle="--", linewidth=1.8,
               label=f"ensemble used ({n_used})")
    ax.set_xscale("log")
    ax.set_xlabel("number of realisations")
    ax.set_ylabel(ylabel)
    _legend_right(ax)
    _note(fig, "error bars: 95% CI")
    return fig


def fig_delta0_scaling(N_vals, delta0, xi_fit, xi_analytic):
    """Bare hybridisation splitting versus chain length, with the fitted decay."""
    fig, ax = _new()
    ax.semilogy(N_vals, delta0, "o", color=C["edge"], markersize=10,
                label=r"$\delta_0(N)$ numerical")
    nn = np.linspace(min(N_vals), max(N_vals), 200)
    amp = delta0[0] * np.exp(N_vals[0] / xi_fit)
    ax.semilogy(nn, amp * np.exp(-nn / xi_fit), "-", color=C["analytic"],
                linewidth=2.2, label=rf"fit, $\xi_{{\rm fit}}={xi_fit:.3f}$")
    ax.set_xlabel("number of unit cells $N$")
    ax.set_ylabel(r"$\delta_0 \;/\; t_1$")
    _legend_right(ax)
    _note(fig, rf"analytic $\xi={xi_analytic:.3f}$" "\n"
               rf"deviation {abs(xi_fit - xi_analytic) / xi_analytic * 100:.1f}%")
    return fig


def fig_convergence(x_vals, y_vals, xlabel, ylabel, note=None, logx=True,
                    reference=None, ref_label=None):
    """Convergence of a reported quantity against a numerical control parameter."""
    fig, ax = _new()
    ax.plot(x_vals, y_vals, "o-", color=C["edge"], linewidth=2.4, markersize=9,
            label="computed")
    if reference is not None:
        ax.axhline(reference, color=C["analytic"], linestyle="--", linewidth=2.0,
                   label=ref_label or "reference")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _legend_right(ax)
    if note:
        _note(fig, note)
    return fig


# ---------------------------------------------------------------------------
# 6. Cross-validation
# ---------------------------------------------------------------------------

def fig_cross_validation(omegas, rho_gf, rho_ed, rho_surf):
    """Three independent LDOS algorithms on the same axes."""
    fig, ax = _new()
    ax.plot(omegas, rho_gf, "-", color=C["edge"], linewidth=3.4,
            label="Green's function (linear solve)")
    ax.plot(omegas, rho_ed, "--", color=C["bulk"], linewidth=2.4,
            label="biorthogonal eigendecomposition")
    ax.plot(omegas, rho_surf, ":", color=C["surface"], linewidth=2.8,
            label="semi-infinite decimation")
    ax.set_xlabel(r"$\omega \;/\; t_1$")
    ax.set_ylabel(r"LDOS $\rho \;/\; t_1^{-1}$")
    _legend_right(ax)
    return fig


def fig_algorithm_difference(omegas, diff_ed, diff_surf):
    """Pointwise disagreement between the LDOS algorithms."""
    fig, ax = _new()
    ax.semilogy(omegas, np.maximum(np.abs(diff_ed), 1e-20), "-",
                color=C["bulk"], linewidth=2.4,
                label=r"GF $-$ eigendecomposition")
    ax.semilogy(omegas, np.maximum(np.abs(diff_surf), 1e-20), "-",
                color=C["surface"], linewidth=2.4,
                label=r"GF $-$ semi-infinite")
    ax.set_xlabel(r"$\omega \;/\; t_1$")
    ax.set_ylabel("absolute difference")
    _legend_right(ax)
    _note(fig, "machine precision vs\nfinite-size correction")
    return fig


def fig_obc_vs_pbc(evals_obc, evals_pbc, gap_hw):
    """Open versus periodic boundaries."""
    fig, ax = _new()
    ax.scatter(np.sort(evals_obc.real), np.ones(evals_obc.size), s=80,
               color=C["edge"], edgecolor="black", linewidth=0.6,
               label="open boundaries (OBC)")
    ax.scatter(np.sort(evals_pbc.real), np.zeros(evals_pbc.size), s=80,
               marker="s", color=C["bulk"], edgecolor="black", linewidth=0.6,
               label="periodic boundaries (PBC)")
    ax.axvspan(-gap_hw, gap_hw, color=C["grid"], alpha=0.8, zorder=0,
               label="topological gap")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["PBC", "OBC"])
    ax.set_ylim(-0.7, 1.7)
    ax.set_xlabel(r"$\mathrm{Re}\,E \;/\; t_1$")
    _legend_right(ax)
    _note(fig, "in-gap states appear\nonly for open boundaries")
    return fig


def fig_cmt_mapping(d_vals, t_of_d, d_intra, d_inter, t1, t2):
    """Coupled-mode-theory calibration."""
    fig, ax = _new()
    ax.semilogy(d_vals, t_of_d, "-", color=C["neutral"], linewidth=2.6,
                label=r"$t(d)=t_0\,e^{-d/L}$")
    ax.plot([d_intra], [t1], "o", color=C["edge"], markersize=13,
            markeredgecolor="black", label=rf"$t_1={t1:g}$ at $d={d_intra:g}$")
    ax.plot([d_inter], [t2], "s", color=C["bulk"], markersize=13,
            markeredgecolor="black", label=rf"$t_2={t2:g}$ at $d={d_inter:g}$")
    ax.set_xlabel(r"resonator separation $d/a$")
    ax.set_ylabel(r"coupling $t \;/\; t_1$")
    _legend_right(ax)
    _note(fig, "schematic mapping only:\nnot an electromagnetic validation")
    return fig


def audit_directory(figdir: str) -> list:
    """
    Independent check of a written figure set.

    `save` enforces the layout contract at generation time; this re-examines the
    files afterwards, so a defect cannot survive even if a figure were produced
    by some other route. Verifies that each figure exists in both vector and
    raster form, meets the resolution threshold, is neither blank nor cut off at
    the canvas edge, and shares a common canvas size with the rest of the set.
    """
    import os
    from collections import Counter
    from PIL import Image

    min_width, min_ink, edge_band = 3000, 0.004, 6
    problems, sizes = [], Counter()

    for pdf in sorted(f for f in os.listdir(figdir) if f.endswith(".pdf")):
        stem = pdf[:-4]
        png = os.path.join(figdir, stem + ".png")
        if not os.path.exists(png):
            problems.append(f"{stem}: raster companion missing")
            continue

        img = np.asarray(Image.open(png).convert("L"))
        sizes[img.shape] += 1
        if img.shape[1] < min_width:
            problems.append(f"{stem}: width {img.shape[1]} px below threshold")

        ink = img < 250
        if ink.mean() < min_ink:
            problems.append(f"{stem}: blank or nearly blank")
        if (ink[:edge_band, :].any() or ink[-edge_band:, :].any()
                or ink[:, :edge_band].any() or ink[:, -edge_band:].any()):
            problems.append(f"{stem}: content reaches the canvas edge")

    if len(sizes) > 1:
        problems.append("canvas size is not uniform across the set")
    return problems
