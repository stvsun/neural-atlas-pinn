#!/usr/bin/env python3
"""Example 2 (FEM): Poisson on the Stanford bunny, P1 FEM on the neural atlas.

    -Laplace(u) = f in the bunny,   u = g on its surface,
    u = sin(pi x) sin(pi y) sin(pi z),  f = 3 pi^2 u  (normalised coordinates).

Each chart carries a boundary-fitted P1 mesh (see ``chart_fem.py``); charts are
coupled by multiplicative overlapping Schwarz. The exact solution is used only as
the Dirichlet data g on the surface and to score the result:

* every node starts at zero, and the iteration stops on its own relative nodal
  update, never on the error;
* an interface node receives the partition-of-unity blend of every other chart
  whose mesh covers it; a node no chart covers is left natural;
* surface nodes take g at their closest point on the surface.

The error is measured on 50,000 points uniform in the true interior.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atlas import load_atlas  # noqa: E402
from chart_fem import ChartFEM  # noqa: E402
from geometry import BunnyGeometry, fetch_bunny, u_exact  # noqa: E402


def invert(atlas, x: torch.Tensor, i: int, iters: int = 10) -> torch.Tensor:
    """Newton-invert chart i's decoder at physical points x."""
    xi = atlas.local_coords(x, i)
    for _ in range(iters):
        xr = xi.clone().requires_grad_(True)
        y = atlas.decode(xr, i)
        jac = torch.stack([torch.autograd.grad(y[:, k].sum(), xr, retain_graph=True)[0] for k in range(3)], 1)
        res = (y - x).detach()
        xi = (xr - torch.linalg.solve(jac, res.unsqueeze(-1)).squeeze(-1)).detach()
        if float(res.norm(dim=1).max()) < 1e-12:
            break
    return xi


def covering_charts(atlas, charts, x: torch.Tensor, exclude: int = -1):
    """For each chart covering x (its mesh box holds the pulled-back point): the
    reference coordinates, the mask logits, and which points it covers."""
    out = []
    for j, cj in enumerate(charts):
        if j == exclude:
            continue
        lin = atlas.local_coords(x, j)
        near = torch.all(lin.abs() <= cj.r, dim=1)
        if not bool(near.any()):
            continue
        with torch.no_grad():
            logit = atlas.mask_logit(x, j).numpy()
        xi = lin.clone()
        xi[near] = invert(atlas, x[near], j)
        xi = xi.numpy()
        cover = near.numpy() & np.all(np.abs(xi) <= cj.r, axis=1)
        out.append((j, xi, logit, cover))
    return out


def blend(entries, charts, n: int):
    """Partition-of-unity blend over the covering charts; NaN where none covers."""
    if not entries:
        return np.full(n, np.nan)
    logits = np.stack([np.where(e[3], e[2], -np.inf) for e in entries])
    m = logits.max(axis=0)
    w = np.where(np.isfinite(m), np.exp(logits - np.where(np.isfinite(m), m, 0.0)), 0.0)
    vals = np.zeros_like(logits)
    for k, (j, xi, _, cover) in enumerate(entries):
        if cover.any():
            vals[k, cover] = charts[j].evaluate(xi[cover])
    s = w.sum(axis=0)
    return np.where(s > 0, (w * vals).sum(axis=0) / np.where(s > 0, s, 1.0), np.nan)


def solve_level(atlas, geom, n_cells: int, eval_x: np.ndarray, max_sweeps: int, rtol: float):
    t0 = time.time()
    charts = [ChartFEM(atlas, i, geom, n_cells) for i in range(atlas.n_charts)]
    for c in charts:
        c.assemble()
    # Dirichlet data on the surface: g at the closest surface point (the node itself once snapped).
    g_data = [dict(zip(c.phys_nodes.tolist(),
                       u_exact(geom.closest_point(c.decode(c.nodes[c.phys_nodes]))).tolist())) for c in charts]
    transfer = []
    n_art = n_orphan = 0
    for i, c in enumerate(charts):
        x_art = torch.as_tensor(c.decode(c.nodes[c.art_nodes]))
        entries = covering_charts(atlas, charts, x_art, exclude=i) if len(c.art_nodes) else []
        covered = np.zeros(len(c.art_nodes), dtype=bool)
        for e in entries:
            covered |= e[3]
        transfer.append((entries, covered))
        n_art += len(c.art_nodes)
        n_orphan += int((~covered).sum())
    for c in charts:
        c.u = np.zeros(c.n_nodes)

    history = []
    for sweep in range(1, max_sweeps + 1):
        before = [c.u.copy() for c in charts]
        for group in atlas.color_groups:
            for i in group:
                c = charts[i]
                entries, covered = transfer[i]
                data = dict(g_data[i])
                if covered.any():
                    vals = blend(entries, charts, len(c.art_nodes))
                    data.update(zip(c.art_nodes[covered].tolist(), vals[covered].tolist()))
                c.solve(data)
        num = sum(float(np.sum((c.u - b) ** 2)) for c, b in zip(charts, before))
        den = sum(float(np.sum(c.u ** 2)) for c in charts)
        update = (num / max(den, 1e-300)) ** 0.5
        history.append(update)
        print(f"  n={n_cells:2d} sweep {sweep:2d}: relative nodal update {update:.2e}", flush=True)
        if update < rtol:
            break

    x = torch.as_tensor(eval_x)
    u_h = blend(covering_charts(atlas, charts, x), charts, len(eval_x))
    u_h = np.where(np.isnan(u_h), 0.0, u_h)
    u = u_exact(eval_x)
    err = u_h - u
    snapped = np.concatenate([c.snap_fraction for c in charts if hasattr(c, "snap_fraction")])
    return {
        "n_cells": n_cells,
        "element_edge_mm": float(np.mean([c.h for c in charts]) * geom.scale_mm),
        "dofs": int(sum(c.n_nodes for c in charts)),
        "elements": int(sum(len(c.elements) for c in charts)),
        "rel_l2": float(np.linalg.norm(err) / np.linalg.norm(u)),
        "max_error": float(np.abs(err).max()),
        "schwarz_sweeps": len(history),
        "final_update": history[-1],
        "surface_nodes_fully_snapped": float(np.mean(snapped >= 1.0)),
        "interface_nodes": n_art,
        "interface_nodes_uncovered": n_orphan,
        "seconds": time.time() - t0,
    }, u_h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atlas", type=int, choices=[8, 12], default=8, help="Atlas: 8 or 12 charts.")
    ap.add_argument("--n-cells", type=int, nargs="+", default=[8, 16, 24, 32],
                    help="Cells per chart axis; the paper's sweep adds 48 56.")
    ap.add_argument("--max-sweeps", type=int, default=50)
    ap.add_argument("--rtol", type=float, default=1e-6, help="Stop when the relative nodal update is below this.")
    ap.add_argument("--eval-points", type=int, default=50000)
    ap.add_argument("--ply", default=None, help="Local bun_zipper.ply (downloaded if omitted).")
    ap.add_argument("--output-dir", default="out_ex2_fem")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    geom = BunnyGeometry(fetch_bunny(args.ply))
    atlas = load_atlas(args.atlas)
    eval_x = geom.sample_interior(args.eval_points, seed=2026)
    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for n in sorted(args.n_cells):
        row, u_h = solve_level(atlas, geom, n, eval_x, args.max_sweeps, args.rtol)
        rows.append(row)
        print(f"n={n:2d}: rel L2 {100 * row['rel_l2']:.3f}%  max error {row['max_error']:.2e}  "
              f"({row['schwarz_sweeps']} sweeps, {row['dofs']} dofs, {row['seconds']:.0f} s)", flush=True)
        np.savez(os.path.join(args.output_dir, f"fem_{args.atlas}chart_n{n}_solution.npz"),
                 points=eval_x, u_pred=u_h, u_true=u_exact(eval_x))
    with open(os.path.join(args.output_dir, f"fem_{args.atlas}chart_sweep.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    print("\n  n   edge(mm)   rel L2     order")
    for k, r in enumerate(rows):
        order = "" if k == 0 else f"{np.log(rows[k - 1]['rel_l2'] / r['rel_l2']) / np.log(rows[k - 1]['element_edge_mm'] / r['element_edge_mm']):.2f}"
        print(f" {r['n_cells']:2d}   {r['element_edge_mm']:6.2f}   {100 * r['rel_l2']:7.3f}%   {order}")


if __name__ == "__main__":
    main()
