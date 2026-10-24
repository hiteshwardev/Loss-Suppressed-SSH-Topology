"""
extension_figures.py
====================
Figures for the extension "observability of the edge exceptional point" and for
the six-page conference paper built on it.

The figures in figures.py are single-panel 11 x 6.2 in canvases for the full
manuscript. A six-page paper needs compact multi-panel figures at final print
size, so the functions here draw at 6.5 in width (the text width of a US-letter
page with 1 in margins) with 8-9 pt type, using the same colour-blind-safe
palette (`figures.C`). Fonts are the STIX fonts bundled with Matplotlib, so the
output does not depend on fonts installed on the host. Every figure is written
as vector PDF and 600 dpi PNG.

Legends and panel labels are placed outside the plot areas, and
`layout_overlaps` checks that no legend or label covers a plot area, a tick
label, an axis label or another annotation.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .figures import C

WIDTH = 6.5

_RC = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.3,
    "lines.markersize": 4.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.6,
    "ytick.minor.size": 1.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.grid": False,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def apply_style():
    """Matplotlib style of the extension figures. Matplotlib defaults are
    restored first, so settings left by `figures.apply_style` (larger legend
    padding, frames, grids) cannot change the layout."""
    matplotlib.rcdefaults()
    plt.rcParams.update(_RC)


def _panel_label(ax, label):
    """Panel label outside the plot area, above the top-left corner of the axes
    and to the left of the y tick labels, so it cannot cover data."""
    ax.annotate(f"({label})", xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(-26, 4), textcoords="offset points", ha="left", va="bottom",
                fontsize=9, fontweight="bold", annotation_clip=False)


def _legend_above(ax, ncol, handles=None, row_major=True, **kw):
    """Legend outside the plot area, in the band above the axes. With
    row_major the entries read left to right, then top to bottom."""
    if handles is None:
        handles, _ = ax.get_legend_handles_labels()
    if row_major and ncol > 1:
        n = len(handles)
        handles = [handles[i] for j in range(ncol) for i in range(j, n, ncol)]
    opts = dict(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=ncol,
                borderaxespad=0.25, handlelength=1.5, columnspacing=0.9,
                labelspacing=0.25, handletextpad=0.4)
    opts.update(kw)
    return ax.legend(handles=handles, **opts)


def save(fig, outdir, name):
    """Write PDF and 600 dpi PNG; returns the PDF path."""
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), format=ext,
                    metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)
    return os.path.join(outdir, f"{name}.pdf")


# ---------------------------------------------------------------------------
# Layout check: annotations must not cover the plot areas or each other
# ---------------------------------------------------------------------------

def _visible_ticklabel_boxes(ax, renderer):
    """Window extents of the tick labels that are actually drawn on `ax`."""
    boxes = []
    for axis in (ax.xaxis, ax.yaxis):
        lo, hi = sorted(axis.get_view_interval())
        tol = 1e-9 * (abs(lo) + abs(hi))
        for tick in axis.get_major_ticks() + axis.get_minor_ticks():
            loc = tick.get_loc()
            if loc is None or not (lo - tol <= loc <= hi + tol):
                continue
            for lab in (tick.label1, tick.label2):
                if lab.get_visible() and lab.get_text():
                    boxes.append(lab.get_window_extent(renderer))
    return boxes


def layout_overlaps(fig, pad=0.5):
    """
    List of layout conflicts in `fig`; an empty list means the layout is clean.

    Annotations are the legends and the free text (panel labels) of every axes
    and of the figure. A conflict is any annotation that
      - intersects the plot area of any axes,
      - intersects a drawn tick label or an axis label,
      - intersects another annotation, or
      - extends beyond the figure.
    Boxes are compared in display pixels with a tolerance of `pad` pixels.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    def shrink(b):
        return b.from_extents(b.x0 + pad, b.y0 + pad, b.x1 - pad, b.y1 - pad)

    notes = []
    for i, ax in enumerate(fig.axes):
        leg = ax.get_legend()
        if leg is not None:
            notes.append((f"axes {i} legend", leg.get_window_extent(r)))
        for t in ax.texts:
            if t.get_visible() and t.get_text():
                notes.append((f"axes {i} text {t.get_text()!r}", t.get_window_extent(r)))
    for k, leg in enumerate(fig.legends):
        notes.append((f"figure legend {k}", leg.get_window_extent(r)))
    for t in fig.texts:
        if t.get_visible() and t.get_text():
            notes.append((f"figure text {t.get_text()!r}", t.get_window_extent(r)))

    obstacles = []
    for i, ax in enumerate(fig.axes):
        obstacles.append((f"plot area of axes {i}", ax.get_window_extent(r)))
        for b in _visible_ticklabel_boxes(ax, r):
            obstacles.append((f"tick label of axes {i}", b))
        for lab in (ax.xaxis.label, ax.yaxis.label):
            if lab.get_visible() and lab.get_text():
                obstacles.append((f"axis label of axes {i}", lab.get_window_extent(r)))

    W, H = fig.canvas.get_width_height()
    conflicts = []
    for name, b in notes:
        sb = shrink(b)
        if b.x0 < -pad or b.y0 < -pad or b.x1 > W + pad or b.y1 > H + pad:
            conflicts.append(f"{name} extends beyond the figure")
        for oname, ob_ in obstacles:
            if sb.overlaps(ob_):
                conflicts.append(f"{name} overlaps {oname}")
    for a in range(len(notes)):
        for b in range(a + 1, len(notes)):
            if shrink(notes[a][1]).overlaps(notes[b][1]):
                conflicts.append(f"{notes[a][0]} overlaps {notes[b][0]}")
    return conflicts


def _top_margin(fig, band_in):
    """Fraction of the figure height left for a band of `band_in` inches on top."""
    return 1.0 - band_in / fig.get_figheight()


# ---------------------------------------------------------------------------
# Figure 1: pinning under bond and on-site disorder
# ---------------------------------------------------------------------------

def fig_pinning(W_vals, bond_disp, onsite_disp, onsite_sem,
                N_vals, split_herm, split_lossy, delta0_N, delta0_vals):
    """(a) real-energy displacement of the protected mode versus disorder;
    (b) doublet splitting min|Re E| versus chain length, with and without loss.
    Legends and panel labels sit above the plot areas."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.45),
                             gridspec_kw=dict(wspace=0.32))
    ax = axes[0]
    floor = 1e-17
    ax.semilogy(W_vals, np.maximum(bond_disp, floor), "o-", color=C["edge"],
                label="bond disorder")
    ax.errorbar(W_vals[1:], onsite_disp[1:], yerr=1.96 * np.asarray(onsite_sem[1:]),
                fmt="s-", color=C["bulk"], capsize=2, label="on-site disorder")
    ax.set_xlabel(r"disorder width $W/\bar t$")
    ax.set_ylabel(r"mean $|\mathrm{Re}\,E_{\rm p}|/\bar t$")
    ax.set_ylim(1e-17, 1)
    _legend_above(ax, ncol=2)
    _panel_label(ax, "a")

    ax = axes[1]
    ax.semilogy(N_vals, split_herm, "o-", color=C["neutral"], label=r"$\gamma=0$")
    ax.semilogy(N_vals, np.maximum(split_lossy, floor), "s-", color=C["topo"],
                label=r"$\gamma=0.3\,\bar t$")
    ax.semilogy(delta0_N, delta0_vals, ":", color=C["analytic"], lw=1.0,
                label=r"clean $\delta_0(N)$")
    ax.set_xlabel(r"unit cells $N$")
    ax.set_ylabel(r"mean $\min|\mathrm{Re}\,E|/\bar t$")
    ax.set_ylim(1e-17, 1)
    _legend_above(ax, ncol=3)
    _panel_label(ax, "b")
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.17,
                        top=_top_margin(fig, 0.22))
    return fig


# ---------------------------------------------------------------------------
# Figure 2: resolvability of the boundary doublet
# ---------------------------------------------------------------------------

def fig_observability(x_omega, spectra, spectra_labels,
                      g_over_d, e_over_d, cls, curve_g, curve_e,
                      eta_vals, nmax_curves, nmax_labels):
    """
    (a) normalised edge LDOS versus omega/delta0 for several gamma/delta0;
    (b) classification of all (N, gamma, eta) points in reduced units with the
        analytic boundary; (c) longest chain whose Hermitian boundary doublet is
        resolved, versus broadening. Legends and panel labels sit above the
        plot areas; the classes of (b) are named in its legend, not in the map.
    """
    from matplotlib.lines import Line2D

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.95),
                             gridspec_kw=dict(wspace=0.45))
    small = dict(fontsize=7, handlelength=1.2, columnspacing=0.7, handletextpad=0.3,
                 labelspacing=0.2, borderaxespad=0.25)

    ax = axes[0]
    cols = [C["neutral"], C["edge"], C["topo"], C["bulk"], C["surface"]]
    for k, (rho, lab) in enumerate(zip(spectra, spectra_labels)):
        ax.plot(x_omega, rho / rho.max(), color=cols[k % len(cols)], label=lab)
    ax.set_xlim(x_omega.min(), x_omega.max())
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$\omega/\delta_0$")
    ax.set_ylabel(r"$\rho_1(\omega)/\max\rho_1$")
    _legend_above(ax, ncol=3, title=r"$\gamma/\delta_0$", title_fontsize=7, **small)
    _panel_label(ax, "a")

    ax = axes[1]
    style = {1: (C["edge"], "two peaks"),
             3: (C["trivial"], r"one peak, $\gamma\leq2\delta_0$"),
             2: (C["topo"], r"one peak, $\gamma>2\delta_0$")}
    handles = []
    for k, (col, lab) in style.items():
        sel = cls == k
        ax.scatter(g_over_d[sel], e_over_d[sel], s=4.5, color=col, marker="s",
                   linewidths=0, rasterized=True)
        handles.append(Line2D([], [], ls="none", marker="s", ms=4, color=col, label=lab))
    ax.plot(curve_g, curve_e, "-", color=C["analytic"], lw=1.2)
    ax.axvline(2.0, color=C["analytic"], ls="--", lw=0.9)
    handles += [Line2D([], [], color=C["analytic"], lw=1.2, label="Eq. (5)"),
                Line2D([], [], color=C["analytic"], lw=0.9, ls="--", label="EP")]
    ax.set_xscale("log")
    ax.set_yscale("log")
    # limits from the data, so that every grid point is shown
    ax.set_xlim(10 ** np.floor(np.log10(np.min(g_over_d))), 10 ** np.ceil(np.log10(np.max(g_over_d))))
    ax.set_ylim(10 ** np.floor(np.log10(np.min(e_over_d))), 10 ** np.ceil(np.log10(np.max(e_over_d))))
    from matplotlib.ticker import LogLocator
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.set_xlabel(r"$\gamma/\delta_0$")
    ax.set_ylabel(r"$\eta/\delta_0$")
    _legend_above(ax, ncol=2, handles=handles, row_major=False, **small)
    _panel_label(ax, "b")

    ax = axes[2]
    mk = ["o", "s", "^"]
    cl = [C["edge"], C["bulk"], C["topo"]]
    for k, (n, lab) in enumerate(zip(nmax_curves, nmax_labels)):
        ax.semilogx(eta_vals, n, mk[k % 3] + "-", color=cl[k % 3], ms=3, label=lab,
                    drawstyle="steps-post")
    ax.set_xlabel(r"broadening $\eta/\bar t$")
    ax.set_ylabel(r"largest resolvable $N$")
    _legend_above(ax, ncol=2, title=r"$t_2/t_1$", title_fontsize=7, **small)
    _panel_label(ax, "c")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.15,
                        top=_top_margin(fig, 0.56))
    return fig


# ---------------------------------------------------------------------------
# Figure 3: pole weights and the enhancement decomposition
# ---------------------------------------------------------------------------

def fig_weights(x, c_prot, c_loss, c_sum, w_window, w0,
                gammas, I_edge, I_bulk, enh):
    """(a) residues of the two doublet poles at the edge site and the
    gap-integrated edge weight; (b) numerator and denominator of the
    gap-integrated enhancement versus loss. Legends and panel labels sit above
    the plot areas."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.75),
                             gridspec_kw=dict(wspace=0.32))
    small = dict(fontsize=7, handlelength=1.6, columnspacing=0.8, handletextpad=0.35,
                 labelspacing=0.2, borderaxespad=0.25)
    ax = axes[0]
    # Below the EP Re c_p = Re c_l = W_N/2 and the window weight is close to
    # c_p + c_l, so coinciding curves are drawn wide-under-narrow to stay visible.
    ax.axvline(2.0, color=C["grid"], lw=3.0, zorder=0, label="EP")
    ax.axhline(w0 / 2, color=C["neutral"], lw=0.6, ls="-.", zorder=1, label=r"$W_N/2$")
    ax.semilogx(x, c_prot.real, "-", color=C["edge"], lw=3.2, alpha=0.55, zorder=2,
                label=r"Re $c_{\rm p}$")
    ax.semilogx(x, c_loss.real, "-", color=C["bulk"], lw=1.1, zorder=3, label=r"Re $c_{\rm l}$")
    ax.semilogx(x, np.abs(c_prot.imag), ":", color=C["edge"], zorder=3, label=r"$|$Im $c_{\rm p}|$")
    ax.semilogx(x, (c_prot + c_loss).real, "-", color=C["analytic"], lw=2.8, alpha=0.35, zorder=2,
                label=r"$c_{\rm p}+c_{\rm l}$")
    ax.semilogx(x, w_window, "--", color=C["topo"], lw=1.1, zorder=3, label="window weight")
    ax.set_ylim(-0.6, 1.4)
    ax.set_xlabel(r"$\gamma/\delta_0$")
    ax.set_ylabel(r"weight at edge site")
    _legend_above(ax, ncol=4, **small)
    _panel_label(ax, "a")

    ax = axes[1]
    ax.plot(gammas, I_edge / I_edge[0], "-", color=C["edge"],
            label=r"edge $I_{\rm e}(\gamma)/I_{\rm e}(0)$")
    ax.plot(gammas, I_bulk / I_bulk[0], "-", color=C["bulk"],
            label=r"bulk $I_{\rm b}(\gamma)/I_{\rm b}(0)$")
    ax.plot(gammas, enh[0] / enh, "--", color=C["analytic"],
            label=r"$\mathcal{E}(0)/\mathcal{E}(\gamma)$")
    ax.set_xlabel(r"loss $\gamma/\bar t$")
    ax.set_ylabel("ratio to lossless value")
    ax.set_ylim(0, 1.1 * max(np.max(I_bulk / I_bulk[0]), np.max(enh[0] / enh)))
    _legend_above(ax, ncol=2, **small)
    _panel_label(ax, "b")
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.16,
                        top=_top_margin(fig, 0.42))
    return fig


# ---------------------------------------------------------------------------
# Figure 4: full-wave rod chain
# ---------------------------------------------------------------------------

def fig_fullwave(freqs, topo_edge, topo_bulk, triv_edge, gap_window,
                 loss_vals, loss_enh):
    """(a) Purcell spectra of the rod chain; (b) gap-averaged enhancement
    versus material loss on the B rods. Legend and panel labels sit above the
    plot areas."""
    from matplotlib.patches import Patch

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.4),
                             gridspec_kw=dict(wspace=0.32, width_ratios=[1.45, 1]))
    ax = axes[0]
    ax.axvspan(*gap_window, color=C["grid"], alpha=0.8, lw=0, zorder=0)
    ax.semilogy(freqs, topo_edge, "-", color=C["edge"], label="topological, edge")
    ax.semilogy(freqs, topo_bulk, "-", color=C["bulk"], label="topological, bulk")
    ax.semilogy(freqs, triv_edge, "--", color=C["topo"], label="trivial, edge")
    ax.set_xlim(freqs.min(), freqs.max())
    ax.set_xlabel(r"frequency $a/\lambda$")
    ax.set_ylabel("Purcell factor")
    h, _ = ax.get_legend_handles_labels()
    h.append(Patch(color=C["grid"], alpha=0.8, lw=0, label="gap window"))
    _legend_above(ax, ncol=2, handles=h, fontsize=7, labelspacing=0.2)
    _panel_label(ax, "a")

    ax = axes[1]
    ax.plot(loss_vals, loss_enh, "o-", color=C["edge"])
    ax.set_xlabel(r"Im $\varepsilon$ on $B$ rods")
    ax.set_ylabel("gap-averaged edge/bulk")
    ax.set_ylim(0, 1.1 * max(loss_enh))
    _panel_label(ax, "b")
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.18,
                        top=_top_margin(fig, 0.36))
    return fig
