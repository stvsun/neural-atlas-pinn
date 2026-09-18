#!/usr/bin/env python3
"""Figure for Example 2: solution and errors on the Stanford bunny, FEM convergence, PINN training.

Reads the outputs of ``run_fem.py`` and ``run_pinn.py``:

    python plot_results.py --fem-dir out_ex2_fem --pinn-dir out_ex2_pinn --out figures/bunny_example2.png
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import BunnyGeometry, fetch_bunny  # noqa: E402


def render(geom, pts_mm, values, cmap, clim, size=800):
    import pyvista as pv

    pv.OFF_SCREEN = True
    surf = pv.PolyData(geom.verts * geom.scale_mm, np.hstack([np.full((len(geom.tris), 1), 3), geom.tris]).ravel())
    pl = pv.Plotter(off_screen=True, window_size=(size, size))
    pl.set_background("white")
    pl.add_mesh(surf, color="#d9c9a3", opacity=0.12, smooth_shading=True)
    cloud = pv.PolyData(pts_mm)
    cloud["v"] = values
    pl.add_mesh(cloud, scalars="v", cmap=cmap, clim=clim, point_size=3.5, render_points_as_spheres=True,
                show_scalar_bar=False)
    pl.view_vector(np.array([0.0, 0.0, 1.0]), viewup=(0.0, 1.0, 0.0))
    pl.reset_camera(bounds=surf.bounds)
    pl.camera.zoom(1.45)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def crop(imgs, pad=10):
    ink = np.zeros(imgs[0].shape[:2], dtype=bool)
    for im in imgs:
        ink |= np.any(im[..., :3] < 245, axis=-1)
    r, c = np.where(ink)
    return [im[max(r.min() - pad, 0):r.max() + pad, max(c.min() - pad, 0):c.max() + pad] for im in imgs]


def colorbar(fig, ax, cmap, clim, label):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    cax = ax.inset_axes([0.12, -0.04, 0.76, 0.04])
    cb = fig.colorbar(ScalarMappable(Normalize(*clim), cmap), cax=cax, orientation="horizontal")
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-2, 3))
    cb.locator, cb.formatter = MaxNLocator(5), fmt
    cb.update_ticks()
    cb.ax.tick_params(labelsize=8)
    cb.ax.xaxis.get_offset_text().set_fontsize(8)
    cb.set_label(label, fontsize=8, labelpad=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fem-dir", default="out_ex2_fem", help="One or more run_fem.py output dirs (comma-separated).")
    ap.add_argument("--pinn-dir", default="out_ex2_pinn", help="One or more run_pinn.py output dirs (comma-separated).")
    ap.add_argument("--fem-atlas", type=int, default=8, help="Atlas whose finest FEM solution is rendered.")
    ap.add_argument("--pinn-atlas", type=int, default=12, help="Atlas whose PINN solution is rendered.")
    ap.add_argument("--ply", default=None)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures",
                                                  "bunny_example2.png"))
    args = ap.parse_args()

    geom = BunnyGeometry(fetch_bunny(args.ply))
    fem_dirs, pinn_dirs = args.fem_dir.split(","), args.pinn_dir.split(",")
    sweeps = {}
    for d in fem_dirs:
        for f in glob.glob(os.path.join(d, "fem_*chart_sweep.json")):
            sweeps[int(os.path.basename(f).split("_")[1].replace("chart", ""))] = (d, json.load(open(f)))
    pinns = {}
    for d in pinn_dirs:
        for f in glob.glob(os.path.join(d, "pinn_*chart_metrics.json")):
            k = int(os.path.basename(f).split("_")[1].replace("chart", ""))
            pinns[k] = (d, json.load(open(f)), json.load(open(f.replace("_metrics", "_history"))))

    d_fem, rows = sweeps[args.fem_atlas]
    n_max = max(r["n_cells"] for r in rows)
    fem = np.load(os.path.join(d_fem, f"fem_{args.fem_atlas}chart_n{n_max}_solution.npz"))
    d_pinn, m_pinn, _ = pinns[args.pinn_atlas]
    pinn = np.load(os.path.join(d_pinn, f"pinn_{args.pinn_atlas}chart_solution.npz"))
    pts_mm = fem["points"] * geom.scale_mm
    u = fem["u_true"]
    e_fem, e_pinn = np.abs(fem["u_pred"] - u), np.abs(pinn["u_pred"] - pinn["u_true"])
    ulim = float(np.percentile(np.abs(u), 97))
    elim = float(np.percentile(np.concatenate([e_fem, e_pinn]), 99))
    imgs = crop([render(geom, pts_mm, u, "coolwarm", (-ulim, ulim)),
                 render(geom, pts_mm, e_fem, "inferno", (0.0, elim)),
                 render(geom, pts_mm, e_pinn, "inferno", (0.0, elim))])

    rel_fem = 100 * next(r["rel_l2"] for r in rows if r["n_cells"] == n_max)
    fig = plt.figure(figsize=(15.5, 10.4))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.8], hspace=0.28, wspace=0.6,
                          left=0.05, right=0.985, top=0.90, bottom=0.07)
    fig.suptitle("Example 2: Poisson on the Stanford bunny, Dirichlet data on the surface only\n"
                 f"FEM ({args.fem_atlas}-chart atlas, n = {n_max}): rel $L^2$ = {rel_fem:.2f}%     "
                 f"PINN ({args.pinn_atlas}-chart atlas): rel $L^2$ = {100 * m_pinn['rel_l2']:.2f}%     "
                 "(50,000 points uniform in the interior)", fontsize=13)
    for k, (img, title, cmap, clim, lab) in enumerate([
            (imgs[0], "exact solution $u$", "coolwarm", (-ulim, ulim), "$u$"),
            (imgs[1], f"FEM error $|u_h - u|$, n = {n_max}", "inferno", (0.0, elim), "$|u_h - u|$"),
            (imgs[2], "PINN error $|u_\\theta - u|$", "inferno", (0.0, elim), "$|u_\\theta - u|$")]):
        ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(title, fontsize=11)
        colorbar(fig, ax, cmap, clim, lab)

    ax = fig.add_subplot(gs[1, 0:3])
    for k, col in ((8, "#2166ac"), (12, "#1a9850")):
        if k not in sweeps:
            continue
        rr = sweeps[k][1]
        h = np.array([r["element_edge_mm"] for r in rr])
        e = 100 * np.array([r["rel_l2"] for r in rr])
        p = np.polyfit(np.log(h), np.log(e), 1)[0]
        ax.plot(h, e, "o-", color=col, lw=2, label=f"{k}-chart atlas (slope {p:.2f})")
    hs = np.array([4.0, 8.0])
    ax.plot(hs, 0.25 * (hs / hs[0]) ** 2, "k-", lw=1)
    ax.text(hs[1] * 1.05, 0.25 * (hs[1] / hs[0]) ** 2, "$O(h^2)$", va="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    ax.set_xticks([3, 5, 10, 20])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("element edge (mm)")
    ax.set_ylabel("relative $L^2$ error (%)")
    ax.set_title("FEM: h-convergence (boundary-fitted P1, Schwarz)", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    ax = fig.add_subplot(gs[1, 3:6])
    for k, col in ((8, "#2166ac"), (12, "#1a9850")):
        if k not in pinns:
            continue
        hist = pinns[k][2]
        ax.plot([h["sweep"] for h in hist], [100 * h["rel_l2_monitor"] for h in hist], color=col, lw=1.2,
                label=f"{k}-chart atlas: final {100 * pinns[k][1]['rel_l2']:.2f}%")
    ax.set_yscale("log")
    ax.set_xlabel("Schwarz sweep")
    ax.set_ylabel("relative $L^2$ error (%)")
    ax.set_title("PINN: training (error logged only; the kept state is chosen by the training loss)",
                 fontsize=10.5)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
