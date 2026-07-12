#!/usr/bin/env python3
"""Forward elastoplastic BVP on a full torus with 8-chart multiplicative Schwarz.

Loading: cyclic opposite radial displacements on two cross-sections (φ=0 and φ=π)
to change the torus hole aspect ratio.

Material: J₂ elastoplasticity with kinematic hardening, smooth return mapping.

Usage:
    python examples/example3_torus_forward_elastoplastic/run_forward_bvp.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# Repo root
_REPO = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _REPO)

from mapped_sphere.elastoplastic.chart_vector_fem import ChartVectorFEMSolver
from mapped_sphere.elastoplastic.return_mapping import (
    ReturnMappingState,
    smooth_return_map,
    sym_logm,
)
from mapped_sphere.elastoplastic.schwarz_vector import (
    SchwarzVectorSolver,
    precompute_interface_geometry,
    choose_color_groups,
)

# ═══════════════════════════════════════════════════════════════════════════
#  Torus geometry
# ═══════════════════════════════════════════════════════════════════════════
R_MAJOR = 1.0
R_MINOR = 0.35
N_CHARTS = 8
PHI_HALFWIDTH = math.pi / 4.0


class TorusSDF:
    """Analytic signed distance function for a torus."""
    def __init__(self, R=R_MAJOR, r=R_MINOR):
        self.R, self.r = R, r

    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        xy = torch.sqrt(x[:, 0]**2 + x[:, 1]**2 + 1e-30)
        return torch.sqrt((xy - self.R)**2 + x[:, 2]**2) - self.r


class TorusChartDecoder(torch.nn.Module):
    """Analytic map from reference cube [-1,1]^3 to a torus sector."""
    def __init__(self, R=R_MAJOR, r=R_MINOR, phi_center=0.0,
                 phi_halfwidth=PHI_HALFWIDTH):
        super().__init__()
        self.R, self.r = R, r
        self.phi_center = phi_center
        self.phi_halfwidth = phi_halfwidth

    def forward(self, xi: torch.Tensor, **kw) -> torch.Tensor:
        phi = self.phi_center + xi[:, 0] * self.phi_halfwidth
        theta = math.pi * xi[:, 1]
        rho = 0.5 * self.r * (1.0 + xi[:, 2])
        rr = self.R + rho * torch.cos(theta)
        return torch.stack([rr * torch.cos(phi), rr * torch.sin(phi),
                            rho * torch.sin(theta)], dim=1)

    def jacobian(self, xi: torch.Tensor, **kw) -> torch.Tensor:
        """Exact Jacobian dx/dxi of the analytic torus chart map."""
        phi = self.phi_center + xi[:, 0] * self.phi_halfwidth
        theta = math.pi * xi[:, 1]
        rho = 0.5 * self.r * (1.0 + xi[:, 2])

        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        rr = self.R + rho * cos_theta

        dphi_dxi0 = self.phi_halfwidth
        dtheta_dxi1 = math.pi
        drho_dxi2 = 0.5 * self.r

        J = torch.zeros(xi.shape[0], 3, 3, device=xi.device, dtype=xi.dtype)
        J[:, 0, 0] = -rr * sin_phi * dphi_dxi0
        J[:, 1, 0] = rr * cos_phi * dphi_dxi0

        J[:, 0, 1] = -rho * sin_theta * dtheta_dxi1 * cos_phi
        J[:, 1, 1] = -rho * sin_theta * dtheta_dxi1 * sin_phi
        J[:, 2, 1] = rho * cos_theta * dtheta_dxi1

        J[:, 0, 2] = cos_theta * drho_dxi2 * cos_phi
        J[:, 1, 2] = cos_theta * drho_dxi2 * sin_phi
        J[:, 2, 2] = sin_theta * drho_dxi2
        return J


# ═══════════════════════════════════════════════════════════════════════════
#  Elastoplastic stress / tangent closures (per-chart, capturing state)
# ═══════════════════════════════════════════════════════════════════════════

def make_ep_stress_fn(
    state: ReturnMappingState,
    F_old: torch.Tensor,
    mu: torch.Tensor,
    K: torch.Tensor,
    tau_y: torch.Tensor,
    H_kin: torch.Tensor,
    eps: float,
) -> Callable:
    """Return a stress closure P(F) that uses the current plastic state."""
    def stress_fn(F: torch.Tensor) -> torch.Tensor:
        F_old_inv = torch.linalg.inv(F_old)
        F_delta = torch.einsum("eij,ejk->eik", F, F_old_inv)
        tau, _ = smooth_return_map(F_delta, state, mu, K, tau_y, H_kin, eps)
        F_inv_T = torch.linalg.inv(F).transpose(-2, -1)
        P = torch.einsum("eij,ejk->eik", tau, F_inv_T)
        return P
    return stress_fn


def make_ep_tangent_fn(
    state: ReturnMappingState,
    F_old: torch.Tensor,
    mu: torch.Tensor,
    K: torch.Tensor,
    tau_y: torch.Tensor,
    H_kin: torch.Tensor,
    eps: float,
) -> Callable:
    """Return a tangent closure dP/dF(F) via autograd."""
    def tangent_fn(F: torch.Tensor) -> torch.Tensor:
        n_elem = F.shape[0]
        F_var = F.detach().clone().requires_grad_(True)
        F_old_inv = torch.linalg.inv(F_old.detach())
        F_delta = torch.einsum("eij,ejk->eik", F_var, F_old_inv)
        tau, _ = smooth_return_map(F_delta, state, mu, K, tau_y, H_kin, eps)
        F_inv_T = torch.linalg.inv(F_var).transpose(-2, -1)
        P = torch.einsum("eij,ejk->eik", tau, F_inv_T)

        # dP/dF via 9 backward passes (row-major: P_iJ -> index 3i+J)
        C = torch.zeros(n_elem, 9, 9, device=F.device, dtype=F.dtype)
        P_flat = P.reshape(n_elem, 9)
        for ab in range(9):
            g = torch.zeros_like(P_flat)
            g[:, ab] = 1.0
            F_var.grad = None
            P_flat.backward(g, retain_graph=(ab < 8))
            C[:, ab, :] = F_var.grad.reshape(n_elem, 9)
        return C
    return tangent_fn


# ═══════════════════════════════════════════════════════════════════════════
#  Boundary condition setup
# ═══════════════════════════════════════════════════════════════════════════

def build_bc(
    solvers: List[ChartVectorFEMSolver],
    decoders: List[TorusChartDecoder],
    delta: float,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Build per-chart bc_masks and u_bc for the squeeze loading.

    Two cross-sections (φ≈0 and φ≈π) are loaded with opposite x-displacements:
    - Nodes near φ=0: u_x = +delta (push inward, reducing major radius on that side)
    - Nodes near φ=π: u_x = -delta (push inward from opposite side)
    """
    bc_masks = []
    u_bcs = []
    phi_tol = 0.15  # angular tolerance to identify cross-section nodes

    for ci, (sol, dec) in enumerate(zip(solvers, decoders)):
        n = sol.n_nodes
        device, dtype = sol.device, sol.dtype
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        u_bc = torch.zeros(n, 3, device=device, dtype=dtype)

        if n == 0:
            bc_masks.append(mask)
            u_bcs.append(u_bc)
            continue

        # Map to physical coords to get phi angle
        with torch.no_grad():
            x_phys = dec(sol.nodes)
        phi_node = torch.atan2(x_phys[:, 1], x_phys[:, 0])

        # Cross-section near φ=0: prescribe u_x = +delta
        near_0 = torch.abs(phi_node) < phi_tol
        if near_0.any():
            mask |= near_0
            u_bc[near_0, 0] = delta  # push in +x

        # Cross-section near φ=π: prescribe u_x = -delta
        near_pi = torch.abs(torch.abs(phi_node) - math.pi) < phi_tol
        if near_pi.any():
            mask |= near_pi
            u_bc[near_pi, 0] = -delta  # push in -x

        bc_masks.append(mask)
        u_bcs.append(u_bc)

    return bc_masks, u_bcs


# ═══════════════════════════════════════════════════════════════════════════
#  Main driver
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cells", type=int, default=6)
    parser.add_argument("--delta-max", type=float, default=0.02,
                        help="Max radial squeeze (fraction of R_MINOR)")
    parser.add_argument("--steps-per-half", type=int, default=30)
    parser.add_argument("--n-cycles", type=int, default=1)
    parser.add_argument("--n-schwarz", type=int, default=8)
    parser.add_argument("--max-newton", type=int, default=25)
    parser.add_argument("--newton-tol", type=float, default=1e-7)
    parser.add_argument("--eps", type=float, default=0.01,
                        help="Softplus sharpness for return mapping")
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=10,
                        help="Save a checkpoint every N load steps (and always at the final step).")
    parser.add_argument("--out-dir", type=str, default="runs/torus_forward_bvp")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set, look for the latest checkpoint_step*.pt in --out-dir "
        "and continue the load-step loop from step+1 instead of starting "
        "from scratch. Geometry/material/loading config (n_cells, n_charts, "
        "delta_max, steps_per_half, n_cycles) must match the checkpoint or "
        "the resume aborts. Run history.json from the original run is "
        "preserved; this run appends to it.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a specific checkpoint .pt to resume from. "
        "Overrides --resume's auto-discovery of the latest checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device. Default is cpu for bit-exact reproducibility "
        "against the manuscript Table 5 numbers. Use 'auto' for CUDA/MPS, "
        "or 'cuda' on euler's A100s for 10-50x speedup. NOTE: torch.det is "
        "required by the smooth return mapping (line 131, neo_hookean_p) and "
        "is NOT implemented on MPS in torch 2.2.2; --device mps will crash. "
        "Set PYTORCH_ENABLE_MPS_FALLBACK=1 as a workaround.",
    )
    parser.add_argument(
        "--dtype",
        default="float64",
        choices=["auto", "float32", "float64"],
        help="Tensor dtype. Default float64 matches the manuscript and the "
        "smooth return mapping numerical conditioning. 'auto' picks float32 "
        "on cuda/mps and float64 on cpu (per core.device.resolve_dtype).",
    )
    args = parser.parse_args()

    # Resolve device + dtype using the shared core helpers (added in
    # commit 357cf89). Inside main() — NOT at module load — so that the
    # default-dtype mutation only affects this run.
    from mapped_sphere.device import resolve_device, resolve_dtype  # noqa: WPS433
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    torch.set_default_dtype(dtype)
    torch.set_num_threads(args.n_threads)
    print(f"Device={device.type} dtype={dtype} threads={args.n_threads}")
    print(f"PyTorch threads: {torch.get_num_threads()}")

    out_dir = Path(_REPO) / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Material parameters ──────────────────────────────────────────────
    E, nu = 200.0, 0.3
    mu = torch.tensor(E / (2 * (1 + nu)), device=device)
    K = torch.tensor(E / (3 * (1 - 2 * nu)), device=device)
    tau_y = torch.tensor(0.5, device=device)
    H_kin = torch.tensor(20.0, device=device)
    eps_rm = args.eps

    print(f"Material: E={E}, nu={nu}, mu={mu.item():.2f}, K={K.item():.2f}")
    print(f"Plasticity: tau_y={tau_y.item()}, H_kin={H_kin.item()}, eps={eps_rm}")

    # ── Build 8-chart torus atlas ────────────────────────────────────────
    sdf = TorusSDF()
    chart_phi_centers = [i * 2 * math.pi / N_CHARTS for i in range(N_CHARTS)]
    decoders = []
    solvers = []

    print(f"\nBuilding {N_CHARTS}-chart torus atlas (n_cells={args.n_cells})...")
    for ci in range(N_CHARTS):
        phi_c = chart_phi_centers[ci]
        dec = TorusChartDecoder(phi_center=phi_c, phi_halfwidth=PHI_HALFWIDTH)
        decoders.append(dec)
        sol = ChartVectorFEMSolver(
            n_cells=args.n_cells,
            support_r=1.0,
            chart_decoder=dec,
            sdf_oracle=sdf,
            sdf_threshold=-0.005,
            device=device,
            dtype=dtype,
        )
        solvers.append(sol)
        print(f"  Chart {ci}: phi={phi_c:.3f} rad, "
              f"{sol.n_nodes} nodes, {sol.n_elements} elements")

    # ── Build Schwarz solver ─────────────────────────────────────────────
    print("\nPrecomputing interface geometry...")
    t0 = time.time()
    schwarz = SchwarzVectorSolver(
        solvers=solvers,
        chart_decoders=decoders,
    )
    print(f"  Interface precomputation: {time.time()-t0:.1f}s")
    print(f"  Color groups: {schwarz.color_groups}")
    print(f"  Interface pairs: {list(schwarz.iface_cache.keys())}")

    # ── Initialize per-chart plastic state ───────────────────────────────
    states = []
    F_olds = []
    for ci, sol in enumerate(solvers):
        ne = sol.n_elements
        states.append(ReturnMappingState.zeros(
            (ne,), device=device, dtype=dtype))
        F_olds.append(torch.eye(3, device=device, dtype=dtype
                                ).unsqueeze(0).expand(ne, 3, 3).clone())

    # ── Build cyclic loading schedule ────────────────────────────────────
    delta_max = args.delta_max * R_MINOR
    sph = args.steps_per_half
    deltas = []
    for _ in range(args.n_cycles):
        # 0 -> +delta_max
        for s in range(sph):
            deltas.append(delta_max * (s + 1) / sph)
        # +delta_max -> -delta_max
        for s in range(2 * sph):
            deltas.append(delta_max * (1.0 - (s + 1) / sph))
        # -delta_max -> 0
        for s in range(sph):
            deltas.append(delta_max * (-1.0 + (s + 1) / sph))

    n_steps = len(deltas)
    print(f"\nLoading: {args.n_cycles} cycle(s), {sph} steps/half, "
          f"delta_max={delta_max:.4f}, total steps={n_steps}")

    # ── Resume from checkpoint (optional) ────────────────────────────────
    start_step = 0  # first load step to compute (0 = fresh run)
    history = {
        "step": [], "delta": [], "max_u": [], "max_ep_bar": [],
        "interface_jump": [], "wall_time": [],
    }
    if args.resume or args.resume_from is not None:
        if args.resume_from is not None:
            ckpt_path = Path(args.resume_from)
        else:
            existing = sorted(out_dir.glob("checkpoint_step*.pt"))
            ckpt_path = existing[-1] if existing else None

        if ckpt_path is None or not ckpt_path.exists():
            print(f"\n[resume] no checkpoint found in {out_dir}; starting fresh.")
        else:
            print(f"\n[resume] loading {ckpt_path.name}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            # Sanity-check geometry to make sure the resume target matches
            # the current --n-cells / chart config; otherwise per-chart
            # tensor shapes won't line up.
            if ckpt["n_cells"] != args.n_cells:
                raise RuntimeError(
                    f"checkpoint n_cells={ckpt['n_cells']} != args.n_cells={args.n_cells}; "
                    "cannot resume into a different mesh refinement."
                )
            if ckpt["n_charts"] != N_CHARTS:
                raise RuntimeError(
                    f"checkpoint n_charts={ckpt['n_charts']} != N_CHARTS={N_CHARTS}; "
                    "cannot resume into a different atlas."
                )
            for ci in range(N_CHARTS):
                schwarz.u_charts[ci] = ckpt["u_charts"][ci].to(
                    device=device, dtype=dtype
                )
                states[ci] = ReturnMappingState(
                    Be=ckpt["states"][ci]["Be"].to(device=device, dtype=dtype),
                    ep_bar=ckpt["states"][ci]["ep_bar"].to(
                        device=device, dtype=dtype
                    ),
                    beta=ckpt["states"][ci]["beta"].to(
                        device=device, dtype=dtype
                    ),
                )
                F_olds[ci] = ckpt["F_olds"][ci].to(
                    device=device, dtype=dtype
                )
            start_step = ckpt["step"] + 1
            # If a partial history.json from the previous run is on disk,
            # carry it through so the post-run summary covers the full run.
            history_path = out_dir / "history.json"
            if history_path.exists():
                try:
                    with open(history_path) as f:
                        prior = json.load(f)
                    for k in history:
                        if k in prior:
                            history[k].extend(prior[k][:start_step])
                except Exception as e:
                    print(f"[resume] could not read prior history.json: {e}")
            print(
                f"[resume] continuing from load step {start_step+1}/{n_steps}; "
                f"checkpoint delta={ckpt['delta']:.5f}, "
                f"will compute deltas[{start_step}:{n_steps}]."
            )

    # ── Incremental Schwarz solve ────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"{'Step':>5} {'delta':>9} {'max|u|':>10} {'max_ep':>10} "
          f"{'iface_jump':>12} {'time(s)':>8}")
    print(f"{'='*70}")

    for step_idx in range(start_step, n_steps):
        delta = deltas[step_idx]
        t_step = time.time()

        # Build BCs for this load level
        bc_masks, u_bcs = build_bc(solvers, decoders, delta)

        # Build per-chart elastoplastic closures
        stress_fns = []
        tangent_fns = []
        for ci in range(N_CHARTS):
            sfn = make_ep_stress_fn(
                states[ci], F_olds[ci], mu, K, tau_y, H_kin, eps_rm)
            tfn = make_ep_tangent_fn(
                states[ci], F_olds[ci], mu, K, tau_y, H_kin, eps_rm)
            stress_fns.append(sfn)
            tangent_fns.append(tfn)

        # Schwarz iteration (manually, to use per-chart closures)
        for sweep in range(args.n_schwarz):
            for group in schwarz.color_groups:
                for chart_i in group:
                    sol_i = solvers[chart_i]
                    if sol_i.n_elements == 0:
                        continue

                    # Combine physical + interface BCs
                    iface_mask, iface_vals = schwarz._interpolate_interface_bc(
                        chart_i)
                    bc_mask = bc_masks[chart_i].clone()
                    bc_vals = u_bcs[chart_i].clone()
                    iface_only = iface_mask & ~bc_mask
                    bc_mask |= iface_only
                    bc_vals[iface_only] = iface_vals[iface_only]

                    f_ext = torch.zeros(
                        sol_i.n_nodes, 3, device=device, dtype=dtype)

                    try:
                        u_new = sol_i.solve_nonlinear(
                            stress_fn=stress_fns[chart_i],
                            tangent_fn=tangent_fns[chart_i],
                            f_ext=f_ext,
                            u_bc=bc_vals,
                            bc_mask=bc_mask,
                            u_init=schwarz.u_charts[chart_i],
                            max_iter=args.max_newton,
                            tol=args.newton_tol,
                        )
                        schwarz.u_charts[chart_i] = u_new
                    except Exception as e:
                        print(f"  [WARN] Chart {chart_i} Newton failed: {e}")

        # Update plastic state after Schwarz converges
        with torch.no_grad():
            for ci in range(N_CHARTS):
                sol_i = solvers[ci]
                if sol_i.n_elements == 0:
                    continue
                F_cur = sol_i.compute_F(schwarz.u_charts[ci])
                F_old_inv = torch.linalg.inv(F_olds[ci])
                F_delta = torch.einsum("eij,ejk->eik", F_cur, F_old_inv)
                _, new_state = smooth_return_map(
                    F_delta, states[ci], mu, K, tau_y, H_kin, eps_rm)
                states[ci] = new_state
                F_olds[ci] = F_cur.detach().clone()

        # Diagnostics
        max_u = max(
            torch.norm(schwarz.u_charts[ci], dim=1).max().item()
            for ci in range(N_CHARTS) if solvers[ci].n_elements > 0
        )
        max_ep = max(
            states[ci].ep_bar.max().item()
            for ci in range(N_CHARTS) if solvers[ci].n_elements > 0
        )
        jump = schwarz.interface_jump()
        dt = time.time() - t_step

        history["step"].append(step_idx)
        history["delta"].append(delta)
        history["max_u"].append(max_u)
        history["max_ep_bar"].append(max_ep)
        history["interface_jump"].append(jump)
        history["wall_time"].append(dt)

        print(f"{step_idx+1:5d} {delta:9.5f} {max_u:10.4e} {max_ep:10.4e} "
              f"{jump:12.4e} {dt:8.1f}")

        # Checkpoint every 10 steps
        if (step_idx + 1) % args.checkpoint_every == 0 or step_idx == n_steps - 1:
            ckpt = {
                "step": step_idx,
                "delta": delta,
                "n_cells": args.n_cells,
                "n_charts": N_CHARTS,
                "support_r": 1.0,
                "chart_phi_centers": chart_phi_centers,
                "phi_halfwidth": PHI_HALFWIDTH,
                "u_charts": [u.cpu() for u in schwarz.u_charts],
                "states": [{
                    "Be": states[ci].Be.cpu(),
                    "ep_bar": states[ci].ep_bar.cpu(),
                    "beta": states[ci].beta.cpu(),
                } for ci in range(N_CHARTS)],
                "F_olds": [fo.cpu() for fo in F_olds],
            }
            torch.save(ckpt, out_dir / f"checkpoint_step{step_idx+1:04d}.pt")

    # ── Save history ─────────────────────────────────────────────────────
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    total_time = sum(history["wall_time"])
    print(f"\n{'='*70}")
    print(f"Done. {n_steps} steps in {total_time:.1f}s "
          f"({total_time/n_steps:.1f}s/step)")
    print(f"Final max|u| = {history['max_u'][-1]:.4e}")
    print(f"Final max ep_bar = {history['max_ep_bar'][-1]:.4e}")
    print(f"Final interface jump = {history['interface_jump'][-1]:.4e}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
