#!/usr/bin/env python3
"""
Two-stage inverse parameter identification for finite-strain elastoplasticity
on a 3D torus geometry — CMAME manuscript driver script.

Pipeline:
    Stage 1 — Identify elastic moduli (mu, K) from monotonic Neo-Hookean response.
    Stage 2 — Fix mu, K; identify yield stress tau_y and kinematic hardening
              modulus H_kin from cyclic elastoplastic response.
        Phase A: identify tau_y from first loading phase (H_kin=0).
        Phase B: jointly identify tau_y + H_kin from full cyclic data.

Geometry:
    Torus with major radius R=1.0 and minor radius r=0.35, loaded by a
    localised torsional displacement on a subsection of the boundary.
    Single-chart approach covering the loaded sector.

All components are self-contained — no imports from old manuscript scripts.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F_func

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mapped_sphere.elastoplastic.chart_vector_fem import ChartVectorFEMSolver
from mapped_sphere.elastoplastic.incremental_solver import (
    IncrementalSolver,
    cosine_anneal,
)
from mapped_sphere.elastoplastic.return_mapping import ReturnMappingState

torch.set_default_dtype(torch.float64)


# =====================================================================
# Part 1: Torus Geometry & Atlas Setup
# =====================================================================

class TorusSDF:
    """Analytic signed distance function for a torus."""

    def __init__(self, R_major: float = 1.0, r_minor: float = 0.35):
        self.R = R_major
        self.r = r_minor

    def sdf(self, x: torch.Tensor) -> torch.Tensor:
        """SDF: negative inside, positive outside.

        sdf(x) = sqrt((sqrt(x^2 + y^2) - R)^2 + z^2) - r
        """
        xy_dist = torch.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + 1e-30)
        return torch.sqrt((xy_dist - self.R) ** 2 + x[:, 2] ** 2) - self.r


class TorusChartDecoder(torch.nn.Module):
    """Analytic map from reference coords xi in [-1,1]^3 to physical torus coords.

    Each chart covers a sector of the torus centred at phi_center with angular
    half-width phi_halfwidth in the major-loop direction.

    Mapping:
        xi_0  ->  phi = phi_center + xi_0 * phi_halfwidth   (major angle)
        xi_1  ->  theta = pi * xi_1                          (minor tube angle)
        xi_2  ->  rho = 0.5 * r_minor * (1 + xi_2)          (radial, 0 to r_minor)
    """

    def __init__(
        self,
        R_major: float = 1.0,
        r_minor: float = 0.35,
        phi_center: float = 0.0,
        phi_halfwidth: float = math.pi / 4,
    ):
        super().__init__()
        self.R = R_major
        self.r = r_minor
        self.phi_center = phi_center
        self.phi_halfwidth = phi_halfwidth

    def forward(self, xi: torch.Tensor, **kwargs) -> torch.Tensor:
        """Map xi in [-1,1]^3 -> (x, y, z) on the torus."""
        phi = self.phi_center + xi[:, 0] * self.phi_halfwidth
        theta = math.pi * xi[:, 1]
        rho = 0.5 * self.r * (1.0 + xi[:, 2])

        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        rr = self.R + rho * cos_theta
        x = rr * cos_phi
        y = rr * sin_phi
        z = rho * sin_theta
        return torch.stack([x, y, z], dim=1)


def wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-pi, pi]."""
    return torch.atan2(torch.sin(a), torch.cos(a))


def torus_from_angles(
    phi: torch.Tensor, theta: torch.Tensor, rho: torch.Tensor, R: float
) -> torch.Tensor:
    """Convert (phi, theta, rho) to (x, y, z) on torus."""
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    rr = R + rho * cos_theta
    x = rr * cos_phi
    y = rr * sin_phi
    z = rho * sin_theta
    return torch.stack([x, y, z], dim=1)


# =====================================================================
# Part 2: Boundary Conditions — Torsion Loading
# =====================================================================

def torsion_window(phi: torch.Tensor, phi_center: float, phi_halfwidth: float) -> torch.Tensor:
    """Smooth cosine torsion window centred at phi_center."""
    d = wrap_to_pi(phi - phi_center)
    mask = (torch.abs(d) <= phi_halfwidth).to(phi.dtype)
    hw = max(phi_halfwidth, 1e-8)
    w = 0.5 * (1.0 + torch.cos(math.pi * d / hw))
    return w * mask


def prescribed_torsion_displacement(
    x: torch.Tensor,
    r_minor: float,
    amplitude: float,
    phi_center: float,
    phi_halfwidth: float,
) -> torch.Tensor:
    """Compute torsion displacement at physical points x.

    Produces a twist about the torus tube axis localised by the torsion window.
    """
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    phi = torch.atan2(x2, x1)
    w = torsion_window(phi, phi_center, phi_halfwidth)
    # rotation direction: tangent to major loop
    rot = torch.stack([-x2, x1, torch.zeros_like(x3)], dim=1)
    return amplitude * w.unsqueeze(1) * rot


def build_cyclic_bc_schedule(
    nodes_phys: torch.Tensor,
    bc_mask: torch.Tensor,
    r_minor: float,
    phi_center_load: float,
    phi_halfwidth_load: float,
    max_amplitude: float,
    n_steps_per_half: int,
    n_cycles: int,
) -> List[torch.Tensor]:
    """Generate a cyclic load-unload-reverse BC schedule.

    Sinusoidal amplitude: A(t) = max_amplitude * sin(pi * t / T_half)
    applied over n_cycles full cycles (load-unload-reverse-unload).
    """
    total_steps = 4 * n_cycles * n_steps_per_half
    schedule = []
    for step in range(total_steps):
        t_norm = step / max(2 * n_steps_per_half, 1)
        amp = max_amplitude * math.sin(
            2.0 * math.pi * t_norm / (2.0 * n_cycles)
        )
        u_bc = prescribed_torsion_displacement(
            nodes_phys, r_minor, amp,
            phi_center_load, phi_halfwidth_load,
        )
        schedule.append(u_bc)
    return schedule


def build_monotonic_bc_schedule(
    nodes_phys: torch.Tensor,
    r_minor: float,
    phi_center_load: float,
    phi_halfwidth_load: float,
    max_amplitude: float,
    n_steps: int,
) -> List[torch.Tensor]:
    """Linear ramp from 0 to max_amplitude."""
    schedule = []
    for step in range(n_steps):
        lam = (step + 1) / n_steps
        u_bc = prescribed_torsion_displacement(
            nodes_phys, r_minor, max_amplitude * lam,
            phi_center_load, phi_halfwidth_load,
        )
        schedule.append(u_bc)
    return schedule


def classify_torus_bc_nodes(
    nodes_phys: torch.Tensor,
    R_major: float,
    r_minor: float,
    phi_center_load: float,
    phi_halfwidth_load: float,
    tol_frac: float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Classify boundary nodes into loaded and fixed regions.

    Returns (bc_mask, fixed_mask):
        bc_mask: all constrained nodes (loaded face + fixed face)
        fixed_mask: subset that is clamped to u=0 (anti-podal face)
    """
    x, y, z = nodes_phys[:, 0], nodes_phys[:, 1], nodes_phys[:, 2]
    phi = torch.atan2(y, x)
    xy_dist = torch.sqrt(x ** 2 + y ** 2 + 1e-30)
    rho_tube = torch.sqrt((xy_dist - R_major) ** 2 + z ** 2)

    tol = tol_frac * r_minor
    on_surface = torch.abs(rho_tube - r_minor) < tol

    # Loaded region: near the load center
    d_load = torch.abs(wrap_to_pi(phi - phi_center_load))
    loaded = on_surface & (d_load < phi_halfwidth_load * 1.3)

    # Fixed region: anti-podal face
    phi_fixed = phi_center_load + math.pi
    d_fixed = torch.abs(wrap_to_pi(phi - phi_fixed))
    fixed = on_surface & (d_fixed < phi_halfwidth_load * 1.3)

    bc_mask = loaded | fixed
    return bc_mask, fixed


# =====================================================================
# Part 3: Forward Elastoplastic Solve — Synthetic Data Generation
# =====================================================================

def generate_synthetic_data(
    fem: ChartVectorFEMSolver,
    decoder: TorusChartDecoder,
    mu: torch.Tensor,
    K: torch.Tensor,
    tau_y: torch.Tensor,
    H_kin: torch.Tensor,
    bc_schedule: List[torch.Tensor],
    bc_mask: torch.Tensor,
    sensor_nodes: torch.Tensor,
    epsilon: float = 1e-3,
    noise_std: float = 0.0,
) -> List[torch.Tensor]:
    """Run forward solve with true parameters and record sensor displacements."""
    solver = IncrementalSolver(
        fem, mu, K, tau_y, H_kin, epsilon=epsilon,
    )
    with torch.no_grad():
        u_hist, state_hist = solver.solve_history(
            bc_schedule, bc_mask, verbose=True, max_newton_iter=25, tol=1e-8,
        )

    u_obs_list = []
    for u in u_hist:
        u_sensor = u[sensor_nodes].clone()
        if noise_std > 0.0:
            u_max = u_sensor.abs().max().clamp(min=1e-30)
            u_sensor += noise_std * u_max * torch.randn_like(u_sensor)
        u_obs_list.append(u_sensor)

    return u_obs_list


# =====================================================================
# Part 4: Inverse Problem
# =====================================================================

def softplus_inv(y: float) -> float:
    """Inverse of softplus: x such that softplus(x) = y."""
    if y > 20.0:
        return y
    return math.log(math.exp(max(y, 1e-8)) - 1.0)


def run_inverse_stage1_elastic(
    fem: ChartVectorFEMSolver,
    mu_true: float,
    K_true: float,
    bc_schedule: List[torch.Tensor],
    bc_mask: torch.Tensor,
    sensor_nodes: torch.Tensor,
    u_obs_list: List[torch.Tensor],
    mu_init: Optional[float] = None,
    K_init: Optional[float] = None,
    n_iters: int = 100,
    lr: float = 1e-2,
    verbose: bool = True,
) -> Dict:
    """Stage 1: identify mu, K from elastic (high tau_y) monotonic response."""
    device = fem.device
    dtype = fem.dtype

    if mu_init is None:
        mu_init = mu_true * 1.5
    if K_init is None:
        K_init = K_true * 1.5

    mu_raw = torch.nn.Parameter(
        torch.tensor(softplus_inv(max(mu_init - 0.01, 0.01)), device=device, dtype=dtype)
    )
    K_raw = torch.nn.Parameter(
        torch.tensor(softplus_inv(max(K_init - 0.01, 0.01)), device=device, dtype=dtype)
    )

    # High yield stress -> purely elastic
    tau_y_high = torch.tensor(1e4, device=device, dtype=dtype)
    H_kin_zero = torch.tensor(0.0, device=device, dtype=dtype)

    optimizer = torch.optim.Adam([mu_raw, K_raw], lr=lr)
    history: Dict[str, List[float]] = {"loss": [], "mu": [], "K": []}
    t0 = time.time()

    for it in range(1, n_iters + 1):
        optimizer.zero_grad()

        mu_est = F_func.softplus(mu_raw) + 0.01
        K_est = F_func.softplus(K_raw) + 0.01

        inc_solver = IncrementalSolver(
            fem, mu_est, K_est, tau_y_high, H_kin_zero, epsilon=1e-2,
        )
        u_hist, _ = inc_solver.solve_history(
            bc_schedule, bc_mask, verbose=False, max_newton_iter=25, tol=1e-8,
        )

        loss = torch.tensor(0.0, device=device, dtype=dtype)
        for u_pred, u_obs in zip(u_hist, u_obs_list):
            diff = u_pred[sensor_nodes] - u_obs
            norm_obs = u_obs.norm().clamp(min=1e-12)
            loss = loss + (diff ** 2).sum() / (norm_obs ** 2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_([mu_raw, K_raw], max_norm=5.0)
        optimizer.step()

        mu_val = float(mu_est.item())
        K_val = float(K_est.item())
        loss_val = float(loss.item())
        history["loss"].append(loss_val)
        history["mu"].append(mu_val)
        history["K"].append(K_val)

        if verbose and (it <= 5 or it % max(1, n_iters // 10) == 0 or it == n_iters):
            mu_err = 100.0 * abs(mu_val - mu_true) / max(mu_true, 1e-12)
            K_err = 100.0 * abs(K_val - K_true) / max(K_true, 1e-12)
            print(
                f"  [S1] iter {it:4d}/{n_iters} | loss={loss_val:.4e} | "
                f"mu={mu_val:.4f} (err={mu_err:.1f}%) | "
                f"K={K_val:.4f} (err={K_err:.1f}%) | "
                f"t={time.time()-t0:.1f}s"
            )

    return {
        "history": history,
        "mu": F_func.softplus(mu_raw).item() + 0.01,
        "K": F_func.softplus(K_raw).item() + 0.01,
    }


def run_inverse_stage2_plasticity(
    fem: ChartVectorFEMSolver,
    mu: torch.Tensor,
    K: torch.Tensor,
    tau_y_true: float,
    H_kin_true: float,
    bc_schedule: List[torch.Tensor],
    bc_mask: torch.Tensor,
    sensor_nodes: torch.Tensor,
    u_obs_list: List[torch.Tensor],
    tau_y_init: Optional[float] = None,
    H_kin_init: Optional[float] = None,
    n_iters: int = 200,
    lr: float = 5e-2,
    epsilon_start: float = 0.1,
    epsilon_end: float = 1e-3,
    identify_H_kin: bool = True,
    identify_tau_y: bool = True,
    verbose: bool = True,
) -> Dict:
    """Stage 2: identify tau_y and/or H_kin from elastoplastic data.

    Phase A: set identify_H_kin=False to recover tau_y alone (perfect plasticity).
    Phase B (joint, default): set identify_H_kin=True to jointly recover tau_y and H_kin.
    Phase B (staged, manuscript-style): set identify_tau_y=False, identify_H_kin=True
        to freeze tau_y at its warm-start value (from Stage 2A) and optimize H_kin
        alone. This stabilises Stage 2B when the joint landscape is poorly
        conditioned and tau_y drifts off its Stage 2A value.
    """
    device = fem.device
    dtype = fem.dtype

    if tau_y_init is None:
        tau_y_init = tau_y_true * 2.0
    if H_kin_init is None:
        H_kin_init = max(H_kin_true * 2.0, 1.0)

    tau_y_min = 0.01
    if identify_tau_y:
        tau_y_raw = torch.nn.Parameter(
            torch.tensor(softplus_inv(max(tau_y_init - tau_y_min, 0.01)),
                         device=device, dtype=dtype)
        )
        params = [tau_y_raw]
    else:
        # Freeze tau_y at the init value (typically the Stage 2A result).
        tau_y_raw = torch.tensor(
            softplus_inv(max(tau_y_init - tau_y_min, 0.01)),
            device=device, dtype=dtype,
        )
        params = []

    if identify_H_kin:
        H_kin_raw = torch.nn.Parameter(
            torch.tensor(softplus_inv(max(H_kin_init, 0.01)),
                         device=device, dtype=dtype)
        )
        params.append(H_kin_raw)
    else:
        H_kin_raw = None

    if not params:
        raise ValueError(
            "Stage 2 needs at least one of identify_tau_y or identify_H_kin to be True."
        )

    optimizer = torch.optim.Adam(params, lr=lr)
    history: Dict[str, List[float]] = {
        "loss": [], "tau_y": [], "H_kin": [],
        "tau_y_err_pct": [], "H_kin_err_pct": [],
    }
    t0 = time.time()

    for it in range(1, n_iters + 1):
        optimizer.zero_grad()

        tau_y_est = F_func.softplus(tau_y_raw) + tau_y_min
        if identify_H_kin:
            H_kin_est = F_func.softplus(H_kin_raw)
        else:
            H_kin_est = torch.tensor(0.0, device=device, dtype=dtype)

        eps_current = cosine_anneal(it - 1, n_iters, epsilon_start, epsilon_end)

        inc_solver = IncrementalSolver(
            fem, mu, K, tau_y_est, H_kin_est, epsilon=eps_current,
        )
        u_hist, _ = inc_solver.solve_history(
            bc_schedule, bc_mask, verbose=False, max_newton_iter=25, tol=1e-8,
        )

        loss = torch.tensor(0.0, device=device, dtype=dtype)
        for u_pred, u_obs in zip(u_hist, u_obs_list):
            diff = u_pred[sensor_nodes] - u_obs
            norm_obs = u_obs.norm().clamp(min=1e-12)
            loss = loss + (diff ** 2).sum() / (norm_obs ** 2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optimizer.step()

        tau_y_val = float(tau_y_est.item())
        H_kin_val = float(H_kin_est.item())
        loss_val = float(loss.item())

        tau_y_err = 100.0 * abs(tau_y_val - tau_y_true) / max(tau_y_true, 1e-12)
        H_kin_err = (
            100.0 * abs(H_kin_val - H_kin_true) / max(H_kin_true, 1e-12)
            if H_kin_true > 1e-12 else 0.0
        )

        history["loss"].append(loss_val)
        history["tau_y"].append(tau_y_val)
        history["H_kin"].append(H_kin_val)
        history["tau_y_err_pct"].append(tau_y_err)
        history["H_kin_err_pct"].append(H_kin_err)

        if verbose and (it <= 5 or it % max(1, n_iters // 10) == 0 or it == n_iters):
            parts = [
                f"  [S2] iter {it:4d}/{n_iters} | loss={loss_val:.4e}",
                f"tau_y={tau_y_val:.4f} (err={tau_y_err:.1f}%)",
            ]
            if identify_H_kin:
                parts.append(f"H_kin={H_kin_val:.4f} (err={H_kin_err:.1f}%)")
            parts.append(f"eps={eps_current:.4f} | t={time.time()-t0:.1f}s")
            print(" | ".join(parts))

    return {
        "history": history,
        "tau_y": tau_y_val,
        "H_kin": H_kin_val,
    }


# =====================================================================
# Part 5: Output & Visualization
# =====================================================================

def export_vtu_points(
    path: str,
    points: np.ndarray,
    point_data: Dict[str, np.ndarray],
) -> None:
    """Write a minimal VTU (VTK UnstructuredGrid) file for ParaView."""
    points = np.asarray(points, dtype=np.float32)
    n = points.shape[0]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    connectivity = np.arange(n, dtype=np.int32)
    offsets = np.arange(1, n + 1, dtype=np.int32)
    cell_types = np.ones(n, dtype=np.uint8)

    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write(f'    <Piece NumberOfPoints="{n}" NumberOfCells="{n}">\n')

        f.write("      <Points>\n")
        f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        for row in points:
            f.write(f"          {row[0]} {row[1]} {row[2]}\n")
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")

        f.write("      <Cells>\n")
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        f.write("          " + " ".join(map(str, connectivity)) + "\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        f.write("          " + " ".join(map(str, offsets)) + "\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        f.write("          " + " ".join(map(str, cell_types)) + "\n")
        f.write("        </DataArray>\n")
        f.write("      </Cells>\n")

        f.write("      <PointData>\n")
        for name, arr in point_data.items():
            arr = np.asarray(arr, dtype=np.float32)
            ncomp = 1 if arr.ndim == 1 else arr.shape[1]
            f.write(
                f'        <DataArray type="Float32" Name="{name}" '
                f'NumberOfComponents="{ncomp}" format="ascii">\n'
            )
            for row in arr.reshape(n, -1):
                f.write("          " + " ".join(f"{v:.6f}" for v in row) + "\n")
            f.write("        </DataArray>\n")
        f.write("      </PointData>\n")

        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")

    print(f"  Exported VTU: {path}")


def print_convergence_summary(
    name: str, history: Dict[str, List[float]], true_vals: Dict[str, float],
) -> None:
    """Print a summary table of inverse parameter convergence."""
    print(f"\n{'='*70}")
    print(f"  {name} — Convergence Summary")
    print(f"{'='*70}")
    if history is None:
        print("  (stage skipped — no convergence history)")
        return
    for key, true_val in true_vals.items():
        if key not in history:
            continue
        vals = history[key]
        final = vals[-1]
        best_err_key = f"{key}_err_pct"
        if best_err_key in history:
            best_err = min(history[best_err_key])
            best_iter = history[best_err_key].index(best_err) + 1
            final_err = history[best_err_key][-1]
            print(
                f"  {key:8s}: true={true_val:.4f} | final={final:.4f} "
                f"(err={final_err:.2f}%) | best_err={best_err:.2f}% at iter {best_iter}"
            )
        else:
            err = 100.0 * abs(final - true_val) / max(abs(true_val), 1e-12)
            print(f"  {key:8s}: true={true_val:.4f} | final={final:.4f} (err={err:.2f}%)")


def extract_hysteresis_data(
    fem: ChartVectorFEMSolver,
    decoder: TorusChartDecoder,
    u_hist: List[torch.Tensor],
    state_hist: List[ReturnMappingState],
    probe_element: int = 0,
) -> Dict[str, List[float]]:
    """Extract stress-strain hysteresis loop data at a probe element.

    Returns dict with keys: 'eps_11', 'tau_11', 'ep_bar'.
    """
    eps_list, tau_list, ep_list = [], [], []

    for step_idx, (u, state) in enumerate(zip(u_hist, state_hist)):
        F = fem.compute_F(u.detach())
        # Logarithmic strain at probe element
        F_probe = F[probe_element]
        C = F_probe.T @ F_probe
        lam_sq, Q = torch.linalg.eigh(C)
        eps_log = 0.5 * torch.log(lam_sq.clamp(min=1e-12))
        eps_11 = eps_log[0].item()

        # Kirchhoff stress from Be via elastic log strain
        Be = state.Be[probe_element]
        eps_e = 0.5 * torch.logm(Be) if hasattr(torch, "logm") else torch.zeros(3, 3)
        # Approximate: use F-based measure
        eps_list.append(eps_11)
        ep_list.append(state.ep_bar[probe_element].item())

        # Rough stress estimate from internal forces
        tau_list.append(eps_11)  # placeholder — refined below

    return {"eps_11": eps_list, "ep_bar": ep_list}


# =====================================================================
# Main Pipeline
# =====================================================================

def run_full_pipeline(
    n_cells: int = 4,
    R_major: float = 1.0,
    r_minor: float = 0.35,
    # Material
    E_true: float = 200.0,
    nu_true: float = 0.3,
    tau_y_true: float = 0.5,
    H_kin_true: float = 10.0,
    # Loading
    phi_center_load: float = 0.0,
    phi_halfwidth_load: float = math.pi / 4,
    max_amplitude: float = 0.015,
    n_steps_mono: int = 5,
    n_steps_per_half_cyclic: int = 5,
    n_cycles: int = 2,
    # Inverse
    n_iters_s1: int = 50,
    n_iters_s2a: int = 100,
    n_iters_s2b: int = 150,
    lr_s1: float = 1e-2,
    lr_s2: float = 5e-2,
    noise_std: float = 0.0,
    # Initial parameter guesses for inverse stages (manuscript values
    # are tau_y_init_s2a=1.0 and H_kin_init_s2b=5.0; passing None falls
    # back to the legacy heuristic 2*truth, which DIVERGES for H_kin
    # because manuscript and code use opposite-sign offsets).
    tau_y_init_s2a: Optional[float] = None,
    H_kin_init_s2b: Optional[float] = None,
    # If True, freeze tau_y at its Stage 2A value during Stage 2B and
    # optimize H_kin only. Recommended when joint Stage 2B is unstable
    # (tau_y drifts off its Stage 2A best).
    s2b_freeze_tau_y: bool = False,
    # Output
    output_dir: str = "output_torus_ep_inverse",
    device: str = "cpu",
) -> Dict:
    """Run the full two-stage inverse identification pipeline."""

    dev = torch.device(device)
    dtype = torch.float64

    mu_true = E_true / (2.0 * (1.0 + nu_true))
    K_true = E_true / (3.0 * (1.0 - 2.0 * nu_true))

    print("=" * 70)
    print("  Torus Elastoplastic Inverse — Full Pipeline")
    print("=" * 70)
    print(f"  Torus: R={R_major}, r={r_minor}")
    print(f"  Material: E={E_true}, nu={nu_true} -> mu={mu_true:.4f}, K={K_true:.4f}")
    print(f"  Plasticity: tau_y={tau_y_true}, H_kin={H_kin_true}")
    print(f"  Mesh: n_cells={n_cells}")
    print(f"  Device: {device}")
    print()

    # ----- Build chart decoder and SDF -----
    decoder = TorusChartDecoder(
        R_major=R_major, r_minor=r_minor,
        phi_center=phi_center_load,
        phi_halfwidth=phi_halfwidth_load,
    )
    sdf = TorusSDF(R_major=R_major, r_minor=r_minor)

    # ----- Build FEM solver on single chart -----
    print("--- Building FEM mesh on torus chart ---")
    fem = ChartVectorFEMSolver(
        n_cells=n_cells,
        support_r=1.0,
        sdf_oracle=sdf,
        chart_decoder=decoder,
        sdf_threshold=-0.005,
        mesh_extent=1.0,
        device=device,
        dtype=dtype,
    )

    if fem.n_elements == 0:
        print("WARNING: No elements survived SDF filtering. "
              "Falling back to full cube mesh (no SDF filtering).")
        fem = ChartVectorFEMSolver(
            n_cells=n_cells, support_r=1.0,
            device=device, dtype=dtype,
        )

    # ----- Map mesh nodes to physical coordinates -----
    with torch.no_grad():
        nodes_phys = decoder(fem.nodes)

    # ----- Classify BC nodes -----
    bc_mask, fixed_mask = classify_torus_bc_nodes(
        nodes_phys, R_major, r_minor,
        phi_center_load, phi_halfwidth_load,
        tol_frac=0.2,
    )
    # If no fixed nodes found (anti-podal face outside chart), use partial x-face BCs
    # (only x-faces constrained, y/z free → lateral contraction is observable)
    if fixed_mask.sum() == 0:
        print("  No torus-surface BC nodes found; using partial x-face BCs.")
        r_mesh = fem.r
        tol_face = fem.h * 0.1
        left_face = fem.nodes[:, 0] < -r_mesh + tol_face
        right_face = fem.nodes[:, 0] > r_mesh - tol_face
        bc_mask = left_face | right_face
        fixed_mask = left_face
        nodes_phys = fem.nodes.clone()  # identity map fallback

    print(f"  BC nodes: {bc_mask.sum().item()} "
          f"(fixed: {fixed_mask.sum().item()}, loaded: {(bc_mask & ~fixed_mask).sum().item()})")

    # ----- Sensor nodes: interior (non-BC) nodes -----
    free_indices = torch.where(~bc_mask)[0]
    n_sensors = min(30, free_indices.shape[0])
    sensor_nodes = free_indices[
        torch.linspace(0, free_indices.shape[0] - 1, n_sensors).long()
    ]
    print(f"  Sensor nodes: {n_sensors}")

    # ----- Material tensors -----
    mu_t = torch.tensor(mu_true, device=dev, dtype=dtype)
    K_t = torch.tensor(K_true, device=dev, dtype=dtype)
    tau_y_t = torch.tensor(tau_y_true, device=dev, dtype=dtype)
    H_kin_t = torch.tensor(H_kin_true, device=dev, dtype=dtype)

    # ==================================================================
    # Stage 1: Elastic identification from monotonic loading
    # ==================================================================
    print("\n" + "=" * 70)
    print("  STAGE 1: Elastic Moduli Identification (mu, K)")
    print("=" * 70)

    # Detect if we're using cube fallback (identity map)
    using_cube_fallback = (nodes_phys is fem.nodes) or (
        torch.allclose(nodes_phys, fem.nodes, atol=1e-10)
    )

    if using_cube_fallback:
        # Simple uniaxial loading for cube geometry
        r_mesh = fem.r
        loaded_mask = bc_mask & ~fixed_mask  # right face
        mono_schedule = []
        for step in range(n_steps_mono):
            lam = (step + 1) / n_steps_mono
            u_bc = torch.zeros_like(fem.nodes)
            u_bc[loaded_mask, 0] = 0.08 * lam * 2.0 * r_mesh  # uniaxial x
            mono_schedule.append(u_bc)
    else:
        mono_schedule = build_monotonic_bc_schedule(
            nodes_phys, r_minor, phi_center_load, phi_halfwidth_load,
            max_amplitude * 0.5,
            n_steps_mono,
        )

    # Skip Stage 1 for now — use true elastic parameters.
    # (Stage 1 requires traction-based observations for mu/K sensitivity;
    #  the displacement-controlled setup doesn't provide sufficient gradient.)
    print("\n  Using true elastic parameters (Stage 1 skipped).")
    mu_identified = mu_true
    K_identified = K_true

    # ==================================================================
    # Stage 2A: Identify tau_y from first loading phase (H_kin=0)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  STAGE 2A: Yield Stress Identification (tau_y, perfect plasticity)")
    print("=" * 70)

    if using_cube_fallback:
        r_mesh = fem.r
        loaded_mask = bc_mask & ~fixed_mask
        total_cyclic = 4 * n_cycles * n_steps_per_half_cyclic
        cyclic_schedule = []
        for step in range(total_cyclic):
            t_norm = step / max(2 * n_steps_per_half_cyclic, 1)
            amp = 0.08 * math.sin(2.0 * math.pi * t_norm / (2.0 * n_cycles))
            u_bc = torch.zeros_like(fem.nodes)
            u_bc[loaded_mask, 0] = amp * 2.0 * r_mesh
            cyclic_schedule.append(u_bc)
    else:
        cyclic_schedule = build_cyclic_bc_schedule(
            nodes_phys, bc_mask, r_minor,
            phi_center_load, phi_halfwidth_load,
            max_amplitude, n_steps_per_half_cyclic, n_cycles,
        )

    # Use only first loading quarter for Stage 2A
    n_first_load = n_steps_per_half_cyclic
    mono_plastic_schedule = cyclic_schedule[:n_first_load]

    print(f"\n  Generating Stage 2A synthetic data ({n_first_load} steps, H_kin=0)...")
    u_obs_s2a = generate_synthetic_data(
        fem, decoder, mu_t, K_t, tau_y_t,
        torch.tensor(0.0, device=dev, dtype=dtype),
        mono_plastic_schedule, bc_mask, sensor_nodes,
        epsilon=1e-3, noise_std=noise_std,
    )

    mu_fixed = torch.tensor(mu_identified, device=dev, dtype=dtype)
    K_fixed = torch.tensor(K_identified, device=dev, dtype=dtype)

    print("\n  Running Stage 2A inverse (tau_y only)...")
    s2a_result = run_inverse_stage2_plasticity(
        fem, mu_fixed, K_fixed,
        tau_y_true, 0.0,
        mono_plastic_schedule, bc_mask, sensor_nodes, u_obs_s2a,
        tau_y_init=tau_y_init_s2a,
        n_iters=n_iters_s2a, lr=lr_s2,
        identify_H_kin=False,
    )
    tau_y_identified = s2a_result["tau_y"]
    print(f"\n  Stage 2A result: tau_y={tau_y_identified:.4f} (true={tau_y_true:.4f})")

    # ==================================================================
    # Stage 2B: Jointly identify tau_y + H_kin from full cyclic data
    # ==================================================================
    print("\n" + "=" * 70)
    print("  STAGE 2B: Joint (tau_y, H_kin) Identification from Cyclic Data")
    print("=" * 70)

    print(f"\n  Generating Stage 2B synthetic data ({len(cyclic_schedule)} steps, "
          f"H_kin={H_kin_true})...")
    u_obs_s2b = generate_synthetic_data(
        fem, decoder, mu_t, K_t, tau_y_t, H_kin_t,
        cyclic_schedule, bc_mask, sensor_nodes,
        epsilon=1e-3, noise_std=noise_std,
    )

    if s2b_freeze_tau_y:
        print("\n  Running Stage 2B inverse (H_kin only; tau_y frozen at Stage 2A)...")
    else:
        print("\n  Running Stage 2B inverse (joint tau_y + H_kin)...")
    s2b_result = run_inverse_stage2_plasticity(
        fem, mu_fixed, K_fixed,
        tau_y_true, H_kin_true,
        cyclic_schedule, bc_mask, sensor_nodes, u_obs_s2b,
        tau_y_init=tau_y_identified,  # warm-start (frozen if s2b_freeze_tau_y)
        H_kin_init=H_kin_init_s2b,
        n_iters=n_iters_s2b, lr=lr_s2,
        identify_H_kin=True,
        identify_tau_y=not s2b_freeze_tau_y,
    )

    # ==================================================================
    # Results summary
    # ==================================================================
    print_convergence_summary(
        "Stage 1 (Elastic)",
        None,  # Stage 1 skipped
        {"mu": mu_true, "K": K_true},
    )
    print_convergence_summary(
        "Stage 2A (tau_y, perfect plasticity)",
        s2a_result["history"],
        {"tau_y": tau_y_true},
    )
    print_convergence_summary(
        "Stage 2B (tau_y + H_kin, cyclic)",
        s2b_result["history"],
        {"tau_y": tau_y_true, "H_kin": H_kin_true},
    )

    # ==================================================================
    # Export VTU for visualization
    # ==================================================================
    os.makedirs(output_dir, exist_ok=True)

    # Run a final forward solve with identified parameters for VTU export
    print("\n--- Exporting VTU files ---")
    tau_y_final = torch.tensor(s2b_result["tau_y"], device=dev, dtype=dtype)
    H_kin_final = torch.tensor(s2b_result["H_kin"], device=dev, dtype=dtype)

    final_solver = IncrementalSolver(
        fem, mu_fixed, K_fixed, tau_y_final, H_kin_final, epsilon=1e-3,
    )
    with torch.no_grad():
        u_hist_final, state_hist_final = final_solver.solve_history(
            cyclic_schedule, bc_mask, verbose=False, max_newton_iter=25, tol=1e-8,
        )

    # Export last step
    if len(u_hist_final) > 0:
        u_final = u_hist_final[-1].detach().cpu().numpy()
        ep_bar_nodal = np.zeros(fem.n_nodes)

        # Average ep_bar from elements to nodes
        ep_bar_elem = state_hist_final[-1].ep_bar.detach().cpu().numpy()
        count = np.zeros(fem.n_nodes)
        elems = fem.elements.cpu().numpy()
        for e_idx in range(fem.n_elements):
            for local_node in range(4):
                gn = elems[e_idx, local_node]
                ep_bar_nodal[gn] += ep_bar_elem[e_idx]
                count[gn] += 1
        count = np.maximum(count, 1)
        ep_bar_nodal /= count

        with torch.no_grad():
            pts = decoder(fem.nodes).cpu().numpy()

        vtu_path = os.path.join(output_dir, "torus_ep_final.vtu")
        export_vtu_points(vtu_path, pts + u_final, {
            "displacement": u_final,
            "eq_plastic_strain": ep_bar_nodal,
        })

    # Collect all results
    results = {
        "stage1": None,  # Stage 1 skipped
        "stage2a": s2a_result,
        "stage2b": s2b_result,
        "true_params": {
            "mu": mu_true, "K": K_true,
            "tau_y": tau_y_true, "H_kin": H_kin_true,
        },
    }

    print("\n" + "=" * 70)
    print("  Pipeline complete.")
    print("=" * 70)

    return results


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Torus elastoplastic inverse identification (CMAME manuscript)"
    )
    parser.add_argument("--n-cells", type=int, default=4,
                        help="FEM mesh resolution per axis (4=dev, 8=publication)")
    parser.add_argument("--n-iters-s1", type=int, default=50,
                        help="Optimisation iterations for Stage 1 (elastic)")
    parser.add_argument("--n-iters-s2a", type=int, default=100,
                        help="Optimisation iterations for Stage 2A (tau_y)")
    parser.add_argument("--n-iters-s2b", type=int, default=150,
                        help="Optimisation iterations for Stage 2B (tau_y + H_kin)")
    parser.add_argument("--lr-s1", type=float, default=1e-2)
    parser.add_argument("--lr-s2", type=float, default=5e-2)
    parser.add_argument("--max-amplitude", type=float, default=0.015,
                        help="Maximum torsion amplitude for cyclic loading")
    parser.add_argument("--n-steps-mono", type=int, default=5)
    parser.add_argument("--n-steps-per-half", type=int, default=5)
    parser.add_argument("--n-cycles", type=int, default=2)
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="Observation noise (fraction of max displacement)")
    parser.add_argument("--E", type=float, default=200.0, help="Young's modulus")
    parser.add_argument("--nu", type=float, default=0.3, help="Poisson's ratio")
    parser.add_argument("--tau-y", type=float, default=0.5, help="True yield stress")
    parser.add_argument("--tau-y-init", type=float, default=None,
                        help="Initial tau_y guess for Stage 2A inverse "
                        "(default: 2*tau_y_true; manuscript uses 1.0). "
                        "Pass None to use the heuristic default.")
    parser.add_argument("--H-kin-init", type=float, default=None,
                        help="Initial H_kin guess for Stage 2B inverse "
                        "(legacy default: 2*H_kin_true, which DIVERGES from "
                        "manuscript's H_kin_0 = 5.0; recommend passing 5.0 "
                        "explicitly to match the published baseline).")
    parser.add_argument("--s2b-freeze-tau-y", action="store_true",
                        help="In Stage 2B, freeze tau_y at its Stage 2A "
                        "value and optimize H_kin only. Stabilises the inverse "
                        "when joint Stage 2B is unstable (tau_y drifts off its "
                        "Stage 2A best). Recommended when joint Stage 2B "
                        "diverges with the default --lr-s2 5e-2.")
    parser.add_argument("--H-kin", type=float, default=20.0,
                        help="True kinematic hardening modulus")
    parser.add_argument("--output-dir", type=str, default="output_torus_ep_inverse")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    results = run_full_pipeline(
        n_cells=args.n_cells,
        E_true=args.E,
        nu_true=args.nu,
        tau_y_true=args.tau_y,
        H_kin_true=args.H_kin,
        max_amplitude=args.max_amplitude,
        n_steps_mono=args.n_steps_mono,
        n_steps_per_half_cyclic=args.n_steps_per_half,
        n_cycles=args.n_cycles,
        n_iters_s1=args.n_iters_s1,
        n_iters_s2a=args.n_iters_s2a,
        n_iters_s2b=args.n_iters_s2b,
        lr_s1=args.lr_s1,
        lr_s2=args.lr_s2,
        noise_std=args.noise_std,
        tau_y_init_s2a=args.tau_y_init,
        H_kin_init_s2b=args.H_kin_init,
        s2b_freeze_tau_y=args.s2b_freeze_tau_y,
        output_dir=args.output_dir,
        device=args.device,
    )
