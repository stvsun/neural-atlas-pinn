#!/usr/bin/env python3
"""Example 2 (PINN): Poisson on the Stanford bunny, per-chart PINNs coupled by multiplicative Schwarz.

    -Laplace(u) = f in the bunny,   u = g on its surface,
    u = sin(pi x) sin(pi y) sin(pi z),  f = 3 pi^2 u  (normalised coordinates).

Each chart has its own network u_i(xi) in the chart's frame coordinates. Charts are
visited colour group by colour group (multiplicative Schwarz); each visit takes a
few Adam steps on

    w_pde |r|^2 / <f^2>  +  w_bc |u_i - g|^2 / <g^2>                 (true surface only)
      +  w_iv |u_i - u_j|^2 / <g^2>  +  w_if |(grad u_i - grad u_j).n|^2 r^2 / <g^2>

with the neighbours frozen, where r = -Laplace(u_i) - f. The mean squares <f^2> and
<g^2> come from the data, so no term dominates through its units. The global
solution is the partition-of-unity blend of the charts.

The exact solution enters only as the Dirichlet data g on the surface and to score
the result: the state kept is the one with the best training score (PDE residual,
boundary and interface misfit on fixed samples), and the error on 50,000 points
uniform in the true interior is only ever logged.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atlas import MLP, load_atlas  # noqa: E402
from geometry import BunnyGeometry, fetch_bunny, forcing, u_exact  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atlas", type=int, choices=[8, 12], default=12)
    ap.add_argument("--iters", type=int, default=800, help="Schwarz sweeps.")
    ap.add_argument("--local-steps", type=int, default=15, help="Adam steps per chart per sweep.")
    ap.add_argument("--pretrain-epochs", type=int, default=300, help="Boundary-data warm start.")
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--lr-min-frac", type=float, default=0.02, help="Cosine schedule floor.")
    ap.add_argument("--pde-warmup", type=int, default=50, help="Sweeps over which w_pde ramps up.")
    ap.add_argument("--w-pde", type=float, default=1.0)
    ap.add_argument("--w-bc", type=float, default=10.0)
    ap.add_argument("--w-iv", type=float, default=10.0, help="Interface value weight.")
    ap.add_argument("--w-if", type=float, default=1.0, help="Interface flux weight.")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--pde-batch", type=int, default=192)
    ap.add_argument("--bc-batch", type=int, default=192)
    ap.add_argument("--if-batch", type=int, default=128)
    ap.add_argument("--noise", type=float, default=0.30, help="Jitter of the chart region, in chart radii.")
    ap.add_argument("--pool", type=int, default=20000, help="Collocation points per chart.")
    ap.add_argument("--surface-pool", type=int, default=120000)
    ap.add_argument("--pou-threshold", type=float, default=0.02)
    ap.add_argument("--plateau", type=int, default=150, help="Stop after this many sweeps without a better score.")
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-points", type=int, default=50000)
    ap.add_argument("--ply", default=None, help="Local bun_zipper.ply (downloaded if omitted).")
    ap.add_argument("--output-dir", default="out_ex2_pinn")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    T = lambda a: torch.as_tensor(a, dtype=torch.float64)  # noqa: E731
    t_start = time.time()

    geom = BunnyGeometry(fetch_bunny(args.ply))
    atlas = load_atlas(args.atlas)
    K = atlas.n_charts
    nbr, shared = atlas.neighbours()
    eval_x = T(geom.sample_interior(args.eval_points, seed=2026))

    # ---- training regions -------------------------------------------------------
    # Chart i trains on (a) its atlas members jittered by --noise * r and clamped to the
    # 1.25 r box -- a region that holds every interface point it is coupled on -- and
    # (b) wherever its partition-of-unity weight is at least --pou-threshold. Both are
    # restricted to the true body; boundary points lie on the true surface.
    rng = np.random.default_rng(args.seed)
    x_surf = T(geom.sample_surface(args.surface_pool, seed=args.seed))
    x_int = T(geom.sample_interior(max(20000, args.pool * K // 2), seed=args.seed))
    w_int, w_surf = atlas.pou_weights(x_int), atlas.pou_weights(x_surf)
    pool, surf_xi, surf_x = [], [], []
    from scipy.spatial import cKDTree
    for i in range(K):
        r = float(atlas.radii[i])
        mem = atlas.points[atlas.member_index(i)]
        xi_mem = atlas.local_coords(mem, i).numpy()
        parts, kept = [], 0
        for _ in range(40):
            if kept >= args.pool:
                break
            xi = np.clip(xi_mem[rng.integers(0, len(xi_mem), 2 * args.pool)]
                         + args.noise * r * rng.standard_normal((2 * args.pool, 3)), -1.25 * r, 1.25 * r)
            xi = xi[geom.inside(atlas.from_local(T(xi), i).numpy())]
            parts.append(xi)
            kept += len(xi)
        pou_xi = atlas.local_coords(x_int[w_int[:, i] >= args.pou_threshold], i).numpy()
        pool.append(T(np.concatenate([np.concatenate(parts)[:args.pool], pou_xi])))
        d, _ = cKDTree(mem.numpy()).query(x_surf.numpy())
        xi_s = atlas.local_coords(x_surf, i)
        sel = ((T(d) <= 1.5 * args.noise * r) & torch.all(xi_s.abs() <= 1.25 * r, dim=1)) \
            | (w_surf[:, i] >= args.pou_threshold)
        surf_xi.append(xi_s[sel])
        surf_x.append(x_surf[sel])
    f2 = float(torch.mean(forcing(x_int) ** 2))
    g2 = float(torch.mean(u_exact(x_surf) ** 2))
    rbar = float(atlas.radii.mean())
    s_pde, s_u, s_flux = 1.0 / math.sqrt(f2), 1.0 / g2, rbar ** 2 / g2
    print(f"{K} charts; collocation per chart {min(len(p) for p in pool)}-{max(len(p) for p in pool)}, "
          f"surface points per chart {min(len(s) for s in surf_xi)}-{max(len(s) for s in surf_xi)}; "
          f"<f^2> = {f2:.3f}, <g^2> = {g2:.4f}", flush=True)

    nets = [MLP(3, 1, args.width, args.depth) for _ in range(K)]
    opts = [torch.optim.Adam(n.parameters(), lr=args.lr) for n in nets]
    frames = [torch.stack([atlas.t1[i], atlas.t2[i], atlas.nvec[i]]) for i in range(K)]

    def grad_x(i, xi):
        """u_i and its physical gradient (the frame is orthonormal: grad_x = F^T grad_xi)."""
        xi = xi.clone().requires_grad_(True)
        u = nets[i](xi)
        g = torch.autograd.grad(u.sum(), xi, create_graph=True)[0]
        return u, g @ frames[i]

    def pde_residual(i, xi):
        x = atlas.from_local(xi, i).detach().requires_grad_(True)
        u = nets[i](atlas.local_coords(x, i))
        g = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        lap = sum(torch.autograd.grad(g[:, k].sum(), x, create_graph=True)[0][:, k:k + 1] for k in range(3))
        return s_pde * (-lap - forcing(x))

    def interface(i, j, idx, freeze_j=True):
        x = atlas.points[idx]
        n = atlas.seeds[j] - atlas.seeds[i]
        n = n / n.norm()
        ui, gi = grad_x(i, atlas.local_coords(x, i))
        uj, gj = grad_x(j, atlas.local_coords(x, j))
        if freeze_j:
            uj, gj = uj.detach(), gj.detach()
        return torch.mean((ui - uj) ** 2), torch.mean(((gi - gj) @ n) ** 2)

    def pick(n_total, n, gen=None):
        return torch.randint(0, n_total, (n,), generator=gen)

    # Fixed samples for the training score.
    gen = torch.Generator().manual_seed(1234)
    cache_pde = [p[torch.randperm(len(p), generator=gen)[:128]] for p in pool]
    cache_bc = [torch.randperm(len(s), generator=gen)[:128] for s in surf_xi]
    cache_if = {k: torch.as_tensor(np.random.default_rng(1234).choice(v, min(96, len(v)), replace=False))
                for k, v in shared.items()}

    def score_terms():
        pde, bc, iv, fl = [], [], [], []
        for i in range(K):
            pde.append(torch.mean(pde_residual(i, cache_pde[i]) ** 2))
            if len(cache_bc[i]):
                k = cache_bc[i]
                bc.append(torch.mean((nets[i](surf_xi[i][k]) - u_exact(surf_x[i][k])) ** 2))
            for j in nbr[i]:
                if j > i:
                    a, b = interface(i, j, cache_if[(i, j)], freeze_j=False)
                    iv.append(a)
                    fl.append(b)
        m = lambda v: float(torch.mean(torch.stack(v)).detach()) if v else 0.0  # noqa: E731
        return m(pde), s_u * m(bc), s_u * m(iv), s_flux * m(fl)

    @torch.no_grad()
    def blended(x):
        w = atlas.pou_weights(x)
        return torch.sum(w * torch.cat([nets[i](atlas.local_coords(x, i)) for i in range(K)], dim=1), dim=1)

    def rel_l2(x):
        ut = u_exact(x).reshape(-1)
        return float(torch.linalg.norm(blended(x) - ut) / torch.linalg.norm(ut))

    def step(i, loss):
        opts[i].zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nets[i].parameters(), args.grad_clip)
        opts[i].step()

    # ---- warm start on the boundary data ------------------------------------------
    for ep in range(args.pretrain_epochs):
        for i in range(K):
            loss = torch.zeros(())
            if len(surf_xi[i]):
                k = pick(len(surf_xi[i]), 256)
                loss = s_u * torch.mean((nets[i](surf_xi[i][k]) - u_exact(surf_x[i][k])) ** 2)
            iv = [interface(i, j, torch.as_tensor(shared[(min(i, j), max(i, j))])[
                pick(len(shared[(min(i, j), max(i, j))]), args.if_batch // 2)])[0] for j in nbr[i]]
            if iv:
                loss = loss + 0.2 * s_u * torch.mean(torch.stack(iv))
            if loss.requires_grad:
                step(i, loss)

    # ---- multiplicative Schwarz ------------------------------------------------------
    best, best_state, stale, history = float("inf"), None, 0, []
    for it in range(1, args.iters + 1):
        lr = args.lr * (args.lr_min_frac + (1 - args.lr_min_frac) * 0.5
                        * (1 + math.cos(math.pi * (it - 1) / max(1, args.iters - 1))))
        for o in opts:
            for g in o.param_groups:
                g["lr"] = lr
        w_pde = args.w_pde * min(1.0, it / args.pde_warmup)
        for group in atlas.color_groups:
            for i in group:
                for _ in range(args.local_steps):
                    loss = w_pde * torch.mean(pde_residual(i, pool[i][pick(len(pool[i]), args.pde_batch)]) ** 2)
                    if len(surf_xi[i]):
                        k = pick(len(surf_xi[i]), args.bc_batch)
                        loss = loss + args.w_bc * s_u * torch.mean(
                            (nets[i](surf_xi[i][k]) - u_exact(surf_x[i][k])) ** 2)
                    terms = []
                    for j in nbr[i]:
                        idx = torch.as_tensor(shared[(min(i, j), max(i, j))])
                        terms.append(interface(i, j, idx[pick(len(idx), min(args.if_batch, len(idx)))]))
                    if terms:
                        loss = loss + args.w_iv * s_u * torch.mean(torch.stack([a for a, _ in terms])) \
                            + args.w_if * s_flux * torch.mean(torch.stack([b for _, b in terms]))
                    step(i, loss)
        pde, bc, iv, fl = score_terms()
        score = args.w_pde * pde + args.w_bc * bc + args.w_iv * iv + args.w_if * fl
        monitor = rel_l2(eval_x[:4096])  # logged only, never used for a decision
        history.append({"sweep": it, "pde": pde, "bc": bc, "iv": iv, "if": fl, "score": score,
                        "lr": lr, "rel_l2_monitor": monitor})
        if score + 5e-5 < best:
            best, best_state, stale = score, [copy.deepcopy(n.state_dict()) for n in nets], 0
        else:
            stale += 1
        if it % 25 == 0 or it == 1:
            print(f"sweep {it:4d}  score {score:.3e} (pde {pde:.2e} bc {bc:.2e} if {iv:.2e}/{fl:.2e})  "
                  f"lr {lr:.1e}  [monitor rel L2 {100 * monitor:.2f}%]  {time.time() - t_start:.0f}s", flush=True)
        if stale >= args.plateau:
            print(f"no better score for {args.plateau} sweeps; stopping at sweep {it}")
            break

    for n, s in zip(nets, best_state):
        n.load_state_dict(s)
    u_h = blended(eval_x).numpy()
    u = u_exact(eval_x).reshape(-1).numpy()
    rel, mx = float(np.linalg.norm(u_h - u) / np.linalg.norm(u)), float(np.abs(u_h - u).max())
    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.join(args.output_dir, f"pinn_{args.atlas}chart")
    np.savez(stem + "_solution.npz", points=eval_x.numpy(), u_pred=u_h, u_true=u)
    with open(stem + "_metrics.json", "w") as fh:
        json.dump({"atlas_charts": K, "rel_l2": rel, "max_error": mx, "best_score": best,
                   "sweeps": len(history), "seconds": time.time() - t_start, "args": vars(args)}, fh, indent=2)
    with open(stem + "_history.json", "w") as fh:
        json.dump(history, fh)
    print(f"\n{K}-chart PINN: rel L2 {100 * rel:.3f}%  max error {mx:.2e}  "
          f"({len(history)} sweeps, {(time.time() - t_start) / 60:.0f} min)")


if __name__ == "__main__":
    main()
