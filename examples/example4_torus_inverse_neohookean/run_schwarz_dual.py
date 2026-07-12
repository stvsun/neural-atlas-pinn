#!/usr/bin/env python3
"""
Torus inverse Neo-Hookean demo with multiplicative Schwarz alternating updates.

Two inverse data modes are implemented in one file:
1) traction mode: recover mu, K from boundary traction observations.
2) displacement mode: recover mu, K from boundary displacement observations.

The torsion boundary condition is prescribed over a load arc spanning multiple charts.
Charts are updated by red-black alternating sweeps (multiplicative Schwarz).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


torch.set_default_dtype(torch.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("Requested CUDA but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if device_arg == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        print("Requested MPS but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device("cpu")


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "auto":
        return torch.float32 if device.type in ("cuda", "mps") else torch.float64
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float64":
        if device.type == "mps":
            print("Requested float64 on MPS; using float32.")
            return torch.float32
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    if device.type == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.synchronize()


def scalar_from_tensor(x: torch.Tensor, device: torch.device) -> float:
    sync_device(device)
    xc = x.detach().to(device=torch.device("cpu"))
    return float(xc.reshape(-1)[0].item())


def wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def rand_uniform_cpu(n: int, low: float, high: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    r = torch.rand((n,), device=torch.device("cpu"), dtype=torch.float64)
    x = low + (high - low) * r
    return x.to(device=device, dtype=dtype)


def rand_int_cpu(n: int, low: int, high: int, device: torch.device) -> torch.Tensor:
    x = torch.randint(low, high, (n,), device=torch.device("cpu"), dtype=torch.int64)
    return x.to(device=device)


def normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    eps_t = torch.as_tensor(eps, device=x.device, dtype=x.dtype)
    return x / torch.clamp(torch.linalg.norm(x, dim=1, keepdim=True), min=eps_t)


def gradient_tensor(v: torch.Tensor, x: torch.Tensor, create_graph: bool) -> torch.Tensor:
    grads = []
    for i in range(v.shape[1]):
        gi = torch.autograd.grad(
            v[:, i],
            x,
            grad_outputs=torch.ones_like(v[:, i]),
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        grads.append(gi.unsqueeze(1))
    return torch.cat(grads, dim=1)


def neo_hookean_p(F: torch.Tensor, mu: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    det_f = torch.det(F)
    det_floor = torch.as_tensor(1e-8, device=det_f.device, dtype=det_f.dtype)
    det_safe = torch.clamp(det_f, min=det_floor)
    finv_t = torch.linalg.inv(F).transpose(1, 2)
    log_j = torch.log(det_safe).unsqueeze(-1).unsqueeze(-1)
    return mu * (F - finv_t) + K * log_j * finv_t


def torus_from_angles(phi: torch.Tensor, theta: torch.Tensor, rho: torch.Tensor, R: float) -> torch.Tensor:
    cphi = torch.cos(phi)
    sphi = torch.sin(phi)
    cth = torch.cos(theta)
    sth = torch.sin(theta)
    rr = R + rho * cth
    x = rr * cphi
    y = rr * sphi
    z = rho * sth
    return torch.stack([x, y, z], dim=1)


def torus_boundary_normals(phi: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    cphi = torch.cos(phi)
    sphi = torch.sin(phi)
    cth = torch.cos(theta)
    sth = torch.sin(theta)
    n = torch.stack([cth * cphi, cth * sphi, sth], dim=1)
    return normalize_rows(n)


def sample_torus_boundary_full(
    n: int,
    R: float,
    r: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Stratified sampling to ensure chart coverage on full boundary.
    two_pi_f = 2.0 * math.pi
    n_phi = max(8, int(math.ceil(math.sqrt(float(n)))))
    n_theta = max(8, int(math.ceil(float(n) / float(n_phi))))
    d_phi = two_pi_f / float(n_phi)
    d_theta = two_pi_f / float(n_theta)

    phi_lin = torch.linspace(0.0, two_pi_f, steps=n_phi + 1, device=torch.device("cpu"), dtype=torch.float64)[:-1]
    theta_lin = torch.linspace(0.0, two_pi_f, steps=n_theta + 1, device=torch.device("cpu"), dtype=torch.float64)[:-1]
    phi_grid = phi_lin.repeat_interleave(n_theta)
    theta_grid = theta_lin.repeat(n_phi)
    m = int(phi_grid.shape[0])

    j_phi = (torch.rand((m,), device=torch.device("cpu"), dtype=torch.float64) - 0.5) * d_phi
    j_theta = (torch.rand((m,), device=torch.device("cpu"), dtype=torch.float64) - 0.5) * d_theta
    phi_grid = torch.remainder(phi_grid + j_phi, two_pi_f)
    theta_grid = torch.remainder(theta_grid + j_theta, two_pi_f)

    if m > n:
        keep = torch.randperm(m, device=torch.device("cpu"))[:n]
        phi_cpu = phi_grid[keep]
        theta_cpu = theta_grid[keep]
    elif m < n:
        extra = n - m
        phi_extra = torch.rand((extra,), device=torch.device("cpu"), dtype=torch.float64) * two_pi_f
        theta_extra = torch.rand((extra,), device=torch.device("cpu"), dtype=torch.float64) * two_pi_f
        phi_cpu = torch.cat([phi_grid, phi_extra], dim=0)
        theta_cpu = torch.cat([theta_grid, theta_extra], dim=0)
    else:
        phi_cpu = phi_grid
        theta_cpu = theta_grid

    phi = phi_cpu.to(device=device, dtype=dtype)
    theta = theta_cpu.to(device=device, dtype=dtype)
    rho = torch.full((n,), float(r), device=device, dtype=dtype)
    x = torus_from_angles(phi, theta, rho, R)
    nvec = torus_boundary_normals(phi, theta)
    return x, nvec, phi, theta


def sample_torus_boundary_subsection(
    n: int,
    R: float,
    r: float,
    phi_center: float,
    phi_halfwidth: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    two_pi = torch.as_tensor(2.0 * math.pi, device=device, dtype=dtype)
    phi = rand_uniform_cpu(n, phi_center - phi_halfwidth, phi_center + phi_halfwidth, device=device, dtype=dtype)
    theta = rand_uniform_cpu(n, 0.0, 2.0 * math.pi, device=device, dtype=dtype)
    rho = torch.full((n,), float(r), device=device, dtype=dtype)
    x = torus_from_angles(phi, theta, rho, R)
    nvec = torus_boundary_normals(phi, theta)
    phi = torch.remainder(phi, two_pi)
    return x, nvec, phi, theta


def write_vtu_points(path: str, points: np.ndarray, point_data: Dict[str, np.ndarray]) -> None:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be [N,3]")
    n0 = int(points.shape[0])
    if n0 <= 0:
        raise ValueError("points must be non-empty")

    valid = np.all(np.isfinite(points), axis=1)
    cleaned: Dict[str, np.ndarray] = {}
    for name, arr in point_data.items():
        a = np.asarray(arr)
        if a.shape[0] != n0:
            raise ValueError(f"point_data '{name}' first dim must be {n0}, got {a.shape[0]}")
        if a.ndim == 1:
            finite = np.isfinite(a)
        elif a.ndim == 2:
            finite = np.all(np.isfinite(a), axis=1)
        else:
            raise ValueError(f"Unsupported shape for point_data '{name}': {a.shape}")
        valid &= finite
        cleaned[name] = a

    if int(np.count_nonzero(valid)) <= 0:
        raise ValueError(f"No finite rows for VTU export: {path}")
    points = points[valid].astype(np.float32, copy=False)
    for k in list(cleaned.keys()):
        a = cleaned[k][valid]
        if np.issubdtype(a.dtype, np.floating):
            a = a.astype(np.float32, copy=False)
        cleaned[k] = a
    n = int(points.shape[0])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    connectivity = np.arange(n, dtype=np.int32)
    offsets = np.arange(1, n + 1, dtype=np.int32)
    cell_types = np.ones(n, dtype=np.uint8)

    def vtk_type(arr: np.ndarray) -> str:
        if np.issubdtype(arr.dtype, np.integer):
            return "Int32" if arr.dtype.itemsize <= 4 else "Int64"
        return "Float32" if arr.dtype == np.float32 else "Float64"

    def write_arr(fh, arr: np.ndarray) -> None:
        flat = np.asarray(arr).reshape(-1)
        chunk = 24
        for i in range(0, flat.size, chunk):
            fh.write("          " + " ".join(map(str, flat[i : i + chunk])) + "\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write(f'    <Piece NumberOfPoints="{n}" NumberOfCells="{n}">\n')
        f.write("      <Points>\n")
        f.write(f'        <DataArray type="{vtk_type(points)}" NumberOfComponents="3" format="ascii">\n')
        write_arr(f, points)
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")
        f.write("      <Cells>\n")
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        write_arr(f, connectivity)
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        write_arr(f, offsets)
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        write_arr(f, cell_types)
        f.write("        </DataArray>\n")
        f.write("      </Cells>\n")
        f.write("      <PointData>\n")
        for name, arr in cleaned.items():
            ncomp = 1 if arr.ndim == 1 else int(arr.shape[1])
            f.write(
                f'        <DataArray type="{vtk_type(arr)}" Name="{name}" NumberOfComponents="{ncomp}" format="ascii">\n'
            )
            write_arr(f, arr)
            f.write("        </DataArray>\n")
        f.write("      </PointData>\n")
        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")


def torsion_window(phi: torch.Tensor, phi_center: float, phi_halfwidth: float) -> torch.Tensor:
    phi_c = torch.as_tensor(phi_center, device=phi.device, dtype=phi.dtype)
    phi_hw = torch.as_tensor(phi_halfwidth, device=phi.device, dtype=phi.dtype)
    eps = torch.as_tensor(1e-8, device=phi.device, dtype=phi.dtype)
    half = torch.as_tensor(0.5, device=phi.device, dtype=phi.dtype)
    one = torch.as_tensor(1.0, device=phi.device, dtype=phi.dtype)
    pi_t = torch.as_tensor(math.pi, device=phi.device, dtype=phi.dtype)
    d = wrap_to_pi(phi - phi_c)
    m = (torch.abs(d) <= phi_hw).to(phi.dtype)
    w = half * (one + torch.cos(pi_t * d / torch.clamp(phi_hw, min=eps)))
    return w * m


def prescribed_torsion_displacement(
    x: torch.Tensor,
    r_minor: float,
    tau: float,
    phi_center: float,
    phi_halfwidth: float,
    z_scale: float,
) -> torch.Tensor:
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    phi = torch.atan2(x2, x1)
    w = torsion_window(phi, phi_center=phi_center, phi_halfwidth=phi_halfwidth)
    rot = torch.stack([-x2, x1, torch.zeros_like(x3)], dim=1)
    one = torch.as_tensor(1.0, device=x.device, dtype=x.dtype)
    z_s = torch.as_tensor(z_scale, device=x.device, dtype=x.dtype)
    tau_t = torch.as_tensor(tau, device=x.device, dtype=x.dtype)
    denom = torch.as_tensor(max(r_minor, 1e-8), device=x.device, dtype=x.dtype)
    z_fac = one + z_s * x3 / denom
    return tau_t * w.unsqueeze(1) * z_fac.unsqueeze(1) * rot


def displacement_surrogate_from_moduli(
    x: torch.Tensor,
    mu: torch.Tensor,
    K: torch.Tensor,
    mu_ref: float,
    K_ref: float,
    r_minor: float,
    tau: float,
    phi_center: float,
    phi_halfwidth: float,
    z_scale: float,
    major_radius: float,
    k_sensitivity: float,
) -> torch.Tensor:
    # Stable synthetic map for displacement-inverse mode. It keeps the same torsion
    # BC shape but introduces controlled mu/K dependence so both parameters are identifiable.
    u_base = prescribed_torsion_displacement(
        x=x,
        r_minor=r_minor,
        tau=tau,
        phi_center=phi_center,
        phi_halfwidth=phi_halfwidth,
        z_scale=z_scale,
    )
    eps = torch.as_tensor(1e-8, device=x.device, dtype=x.dtype)
    mu_ref_t = torch.as_tensor(mu_ref, device=x.device, dtype=x.dtype)
    K_ref_t = torch.as_tensor(K_ref, device=x.device, dtype=x.dtype)
    c_mu = mu_ref_t / torch.clamp(mu, min=eps)
    c_K = K_ref_t / torch.clamp(K, min=eps)
    xnorm = torch.sqrt(torch.clamp(torch.sum(x * x, dim=1, keepdim=True), min=eps))
    radial = x / xnorm
    phi = torch.atan2(x[:, 1], x[:, 0])
    w = torsion_window(phi, phi_center=phi_center, phi_halfwidth=phi_halfwidth).unsqueeze(1)
    zq = (x[:, 2:3] / torch.as_tensor(max(r_minor, 1e-8), device=x.device, dtype=x.dtype)) ** 2
    vol = (
        torch.as_tensor(k_sensitivity, device=x.device, dtype=x.dtype)
        * tau
        * c_K
        * w
        * zq
        * radial
        * (xnorm / torch.as_tensor(major_radius + r_minor, device=x.device, dtype=x.dtype))
    )
    return c_mu * u_base + vol


def traction_from_moduli(
    x: torch.Tensor,
    n_phys: torch.Tensor,
    mu: torch.Tensor,
    K: torch.Tensor,
    r_minor: float,
    tau: float,
    phi_center: float,
    phi_halfwidth: float,
    z_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_var = x.clone().detach().requires_grad_(True)
    u = prescribed_torsion_displacement(
        x=x_var,
        r_minor=r_minor,
        tau=tau,
        phi_center=phi_center,
        phi_halfwidth=phi_halfwidth,
        z_scale=z_scale,
    )
    grad_u = gradient_tensor(u, x_var, create_graph=False)
    eye = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0)
    F = eye + grad_u
    P = neo_hookean_p(F, mu=mu, K=K)
    t = torch.bmm(P, n_phys.unsqueeze(-1)).squeeze(-1)
    return t, torch.det(F)


def build_chart_weights(phi: torch.Tensor, n_charts: int, sigma_scale: float) -> torch.Tensor:
    two_pi = torch.as_tensor(2.0 * math.pi, device=phi.device, dtype=phi.dtype)
    centers = torch.linspace(0.0, float(two_pi.item()), steps=n_charts + 1, device=phi.device, dtype=phi.dtype)[:-1]
    d = wrap_to_pi(phi.unsqueeze(1) - centers.unsqueeze(0))
    sigma = torch.as_tensor(sigma_scale * (2.0 * math.pi / n_charts), device=phi.device, dtype=phi.dtype)
    logits = -torch.as_tensor(0.5, device=phi.device, dtype=phi.dtype) * (d / sigma) ** 2
    return torch.softmax(logits, dim=1)


def clamp_init(val: float, target: float, frac: float, name: str) -> float:
    lo = target * (1.0 - frac)
    hi = target * (1.0 + frac)
    if val < lo or val > hi:
        clamped = min(max(val, lo), hi)
        print(
            f"[Init] {name}={val:.6f} outside +/-{100.0 * frac:.1f}% of target {target:.6f}; "
            f"clamped to {clamped:.6f}"
        )
        return clamped
    return val


def softplus_inv(y: float) -> float:
    y = max(y, 1e-8)
    return math.log(math.expm1(y))


def estimate_global_from_charts(
    mu_raw: List[torch.nn.Parameter],
    K_raw: List[torch.nn.Parameter],
    mu_min: float,
    K_min: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mu_all = torch.stack([torch.nn.functional.softplus(p) + mu_min for p in mu_raw])
    K_all = torch.stack([torch.nn.functional.softplus(p) + K_min for p in K_raw])
    return mu_all.mean(), K_all.mean(), mu_all, K_all


def make_plot(history: Dict[str, List[float]], out_png: str, mode: str) -> None:
    ep = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].semilogy(ep, np.maximum(history["loss"], 1e-16), label="total")
    axes[0].semilogy(ep, np.maximum(history["data"], 1e-16), label="data")
    axes[0].semilogy(ep, np.maximum(history["if"], 1e-16), label="if")
    axes[0].semilogy(ep, np.maximum(history["reg"], 1e-16), label="reg")
    axes[0].set_title(f"Schwarz Losses ({mode})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(ep, history["mu_mean"], label="mu_mean")
    axes[1].plot(ep, history["K_mean"], label="K_mean")
    axes[1].fill_between(ep, np.array(history["mu_mean"]) - np.array(history["mu_std"]), np.array(history["mu_mean"]) + np.array(history["mu_std"]), alpha=0.2)
    axes[1].fill_between(ep, np.array(history["K_mean"]) - np.array(history["K_std"]), np.array(history["K_mean"]) + np.array(history["K_std"]), alpha=0.2)
    axes[1].set_title("Chart Parameter Mean±Std")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Value")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Dict[str, float]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device=device)
    torch.set_default_dtype(dtype)
    set_seed(args.seed)

    n_charts = int(args.n_charts)
    if n_charts % 2 != 0:
        raise ValueError("n_charts must be even for red-black Schwarz.")
    print(f"Device={device.type} dtype={dtype} n_charts={n_charts} mode={args.inverse_mode}")

    mu_init = args.mu_init if args.mu_init is not None else args.mu_true * 0.97
    K_init = args.K_init if args.K_init is not None else args.K_true * 1.04
    mu_init = clamp_init(mu_init, args.mu_true, 0.05, "mu_init")
    K_init = clamp_init(K_init, args.K_true, 0.05, "K_init")

    x_obs, n_obs, phi_obs, _ = sample_torus_boundary_subsection(
        n=args.n_obs,
        R=args.major_radius,
        r=args.minor_radius,
        phi_center=args.load_phi_center,
        phi_halfwidth=args.load_phi_halfwidth,
        device=device,
        dtype=dtype,
    )
    chart_w = build_chart_weights(phi_obs, n_charts=n_charts, sigma_scale=args.chart_sigma_scale).detach()

    mu_true_t = torch.as_tensor(args.mu_true, device=device, dtype=dtype)
    K_true_t = torch.as_tensor(args.K_true, device=device, dtype=dtype)
    if args.inverse_mode == "traction":
        y_true, det_true = traction_from_moduli(
            x=x_obs,
            n_phys=n_obs,
            mu=mu_true_t,
            K=K_true_t,
            r_minor=args.minor_radius,
            tau=args.torsion_tau,
            phi_center=args.load_phi_center,
            phi_halfwidth=args.load_phi_halfwidth,
            z_scale=args.torsion_z_scale,
        )
        if args.noise_std > 0.0:
            y_true = y_true + args.noise_std * torch.std(y_true) * torch.randn_like(y_true)
        obs_scale = torch.clamp(torch.mean(y_true * y_true), min=torch.as_tensor(1e-12, device=device, dtype=dtype)).detach()
        det_floor_t = torch.as_tensor(args.det_floor, device=device, dtype=dtype)
    else:
        y_true = displacement_surrogate_from_moduli(
            x=x_obs,
            mu=mu_true_t,
            K=K_true_t,
            mu_ref=args.mu_true,
            K_ref=args.K_true,
            r_minor=args.minor_radius,
            tau=args.torsion_tau,
            phi_center=args.load_phi_center,
            phi_halfwidth=args.load_phi_halfwidth,
            z_scale=args.torsion_z_scale,
            major_radius=args.major_radius,
            k_sensitivity=args.disp_k_sensitivity,
        ).detach()
        if args.noise_std > 0.0:
            y_true = y_true + args.noise_std * torch.std(y_true) * torch.randn_like(y_true)
        obs_scale = torch.clamp(torch.mean(y_true * y_true), min=torch.as_tensor(1e-12, device=device, dtype=dtype)).detach()
        det_true = None
        det_floor_t = torch.as_tensor(0.0, device=device, dtype=dtype)

    # per-chart local unknowns + optimizers
    mu_raw: List[torch.nn.Parameter] = []
    K_raw: List[torch.nn.Parameter] = []
    opts: List[torch.optim.Optimizer] = []
    for i in range(n_charts):
        # small chart-dependent perturbation keeps chart objectives non-identical.
        s = 0.002 * float(i - n_charts // 2)
        mu0_i = max(args.mu_min + 1e-6, mu_init * (1.0 + s))
        K0_i = max(args.K_min + 1e-6, K_init * (1.0 - s))
        pm = torch.nn.Parameter(torch.tensor(softplus_inv(mu0_i - args.mu_min), device=device, dtype=dtype))
        pK = torch.nn.Parameter(torch.tensor(softplus_inv(K0_i - args.K_min), device=device, dtype=dtype))
        mu_raw.append(pm)
        K_raw.append(pK)
        opts.append(torch.optim.Adam([pm, pK], lr=args.lr))

    neighbors = {i: [(i - 1) % n_charts, (i + 1) % n_charts] for i in range(n_charts)}
    groups = [list(range(0, n_charts, 2)), list(range(1, n_charts, 2))]

    history: Dict[str, List[float]] = {
        "loss": [],
        "data": [],
        "if": [],
        "reg": [],
        "mu_mean": [],
        "K_mean": [],
        "mu_std": [],
        "K_std": [],
        "obs_rel_l2": [],
    }
    start = time.time()
    best_rel_l2 = float("inf")
    best_epoch = 0
    best_mu_raw = [p.detach().clone() for p in mu_raw]
    best_K_raw = [p.detach().clone() for p in K_raw]
    best_mu_err = float("inf")
    best_K_err = float("inf")

    for epoch in range(1, args.epochs + 1):
        data_acc = 0.0
        if_acc = 0.0
        reg_acc = 0.0
        det_acc = 0.0
        local_updates = 0

        for group in groups:
            for i in group:
                opt = opts[i]
                for _ in range(args.local_steps):
                    idx = rand_int_cpu(min(args.batch_size, args.n_obs), 0, args.n_obs, device=device)
                    xb = x_obs[idx]
                    nb = n_obs[idx]
                    yb = y_true[idx]
                    wi = chart_w[idx, i]
                    # skip nearly-empty local batches
                    if float(torch.sum(wi).detach().item()) < 1e-6:
                        continue

                    mui = torch.nn.functional.softplus(mu_raw[i]) + args.mu_min
                    Ki = torch.nn.functional.softplus(K_raw[i]) + args.K_min
                    if args.inverse_mode == "traction":
                        y_pred_i, det_i = traction_from_moduli(
                            x=xb,
                            n_phys=nb,
                            mu=mui,
                            K=Ki,
                            r_minor=args.minor_radius,
                            tau=args.torsion_tau,
                            phi_center=args.load_phi_center,
                            phi_halfwidth=args.load_phi_halfwidth,
                            z_scale=args.torsion_z_scale,
                        )
                        det_pen = torch.mean(torch.nn.functional.softplus(det_floor_t - det_i) ** 2)
                    else:
                        y_pred_i = displacement_surrogate_from_moduli(
                            x=xb,
                            mu=mui,
                            K=Ki,
                            mu_ref=args.mu_true,
                            K_ref=args.K_true,
                            r_minor=args.minor_radius,
                            tau=args.torsion_tau,
                            phi_center=args.load_phi_center,
                            phi_halfwidth=args.load_phi_halfwidth,
                            z_scale=args.torsion_z_scale,
                            major_radius=args.major_radius,
                            k_sensitivity=args.disp_k_sensitivity,
                        )
                        det_pen = torch.as_tensor(0.0, device=device, dtype=dtype)

                    point_mse = torch.mean((y_pred_i - yb) ** 2, dim=1) / obs_scale
                    loss_data_i = torch.sum(wi * point_mse) / torch.clamp(torch.sum(wi), min=torch.as_tensor(1e-12, device=device, dtype=dtype))

                    n0, n1 = neighbors[i]
                    mu_nb = 0.5 * (
                        (torch.nn.functional.softplus(mu_raw[n0]) + args.mu_min).detach()
                        + (torch.nn.functional.softplus(mu_raw[n1]) + args.mu_min).detach()
                    )
                    K_nb = 0.5 * (
                        (torch.nn.functional.softplus(K_raw[n0]) + args.K_min).detach()
                        + (torch.nn.functional.softplus(K_raw[n1]) + args.K_min).detach()
                    )
                    loss_if_i = ((mui - mu_nb) / args.mu_true) ** 2 + ((Ki - K_nb) / args.K_true) ** 2
                    loss_reg_i = ((mui - args.mu_prior) / args.mu_true) ** 2 + ((Ki - args.K_prior) / args.K_true) ** 2
                    loss = (
                        args.w_data * loss_data_i
                        + args.w_if * loss_if_i
                        + args.w_reg * loss_reg_i
                        + args.w_det * det_pen
                    )

                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([mu_raw[i], K_raw[i]], max_norm=float(args.grad_clip))
                    opt.step()

                    data_acc += float(loss_data_i.detach().item())
                    if_acc += float(loss_if_i.detach().item())
                    reg_acc += float(loss_reg_i.detach().item())
                    det_acc += float(det_pen.detach().item())
                    local_updates += 1

        mu_mean_t, K_mean_t, mu_all_t, K_all_t = estimate_global_from_charts(mu_raw, K_raw, args.mu_min, args.K_min)
        # blended global prediction for eval
        y_pred_all = []
        for i in range(n_charts):
            mui = torch.nn.functional.softplus(mu_raw[i]) + args.mu_min
            Ki = torch.nn.functional.softplus(K_raw[i]) + args.K_min
            if args.inverse_mode == "traction":
                yp, _ = traction_from_moduli(
                    x=x_obs,
                    n_phys=n_obs,
                    mu=mui,
                    K=Ki,
                    r_minor=args.minor_radius,
                    tau=args.torsion_tau,
                    phi_center=args.load_phi_center,
                    phi_halfwidth=args.load_phi_halfwidth,
                    z_scale=args.torsion_z_scale,
                )
            else:
                yp = displacement_surrogate_from_moduli(
                    x=x_obs,
                    mu=mui,
                    K=Ki,
                    mu_ref=args.mu_true,
                    K_ref=args.K_true,
                    r_minor=args.minor_radius,
                    tau=args.torsion_tau,
                    phi_center=args.load_phi_center,
                    phi_halfwidth=args.load_phi_halfwidth,
                    z_scale=args.torsion_z_scale,
                    major_radius=args.major_radius,
                    k_sensitivity=args.disp_k_sensitivity,
                )
            y_pred_all.append(yp)
        y_stack = torch.stack(y_pred_all, dim=1)
        y_pred_blend = torch.sum(chart_w.unsqueeze(-1) * y_stack, dim=1)
        rel_l2 = torch.sqrt(
            torch.mean((y_pred_blend - y_true) ** 2)
            / torch.clamp(torch.mean(y_true**2), min=torch.as_tensor(1e-12, device=device, dtype=dtype))
        )

        denom = max(local_updates, 1)
        loss_data_e = data_acc / denom
        loss_if_e = if_acc / denom
        loss_reg_e = reg_acc / denom
        loss_det_e = det_acc / denom
        loss_total_e = args.w_data * loss_data_e + args.w_if * loss_if_e + args.w_reg * loss_reg_e + args.w_det * loss_det_e

        mu_mean = scalar_from_tensor(mu_mean_t, device=device)
        K_mean = scalar_from_tensor(K_mean_t, device=device)
        mu_std = scalar_from_tensor(torch.std(mu_all_t), device=device)
        K_std = scalar_from_tensor(torch.std(K_all_t), device=device)
        rel_l2_v = scalar_from_tensor(rel_l2, device=device)
        mu_err = 100.0 * abs(mu_mean - args.mu_true) / max(args.mu_true, 1e-12)
        K_err = 100.0 * abs(K_mean - args.K_true) / max(args.K_true, 1e-12)

        history["loss"].append(loss_total_e)
        history["data"].append(loss_data_e)
        history["if"].append(loss_if_e)
        history["reg"].append(loss_reg_e)
        history["mu_mean"].append(mu_mean)
        history["K_mean"].append(K_mean)
        history["mu_std"].append(mu_std)
        history["K_std"].append(K_std)
        history["obs_rel_l2"].append(rel_l2_v)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch:04d}/{args.epochs} | loss={loss_total_e:.3e} data={loss_data_e:.3e} "
            f"if={loss_if_e:.3e} reg={loss_reg_e:.3e} "
            f"mu_mean={mu_mean:.6f} ({mu_err:.3f}%) K_mean={K_mean:.6f} ({K_err:.3f}%) "
            f"obs_relL2={rel_l2_v:.3e} charts_std(mu,K)=({mu_std:.3e},{K_std:.3e}) t={elapsed:.1f}s"
        )

        if (rel_l2_v < best_rel_l2 - 1e-12) or (
            abs(rel_l2_v - best_rel_l2) <= 1e-12 and (mu_err + K_err) < (best_mu_err + best_K_err)
        ):
            best_rel_l2 = rel_l2_v
            best_epoch = epoch
            best_mu_err = mu_err
            best_K_err = K_err
            best_mu_raw = [p.detach().clone() for p in mu_raw]
            best_K_raw = [p.detach().clone() for p in K_raw]

        if rel_l2_v <= args.target_rel_l2 and mu_err <= args.target_mu_rel_percent and K_err <= args.target_K_rel_percent:
            print(
                f"Stopping early: targets reached (relL2<={args.target_rel_l2}, "
                f"mu_err<={args.target_mu_rel_percent}%, K_err<={args.target_K_rel_percent}%)"
            )
            break

    # Restore best chart parameters before final evaluation/export.
    with torch.no_grad():
        for i in range(n_charts):
            mu_raw[i].copy_(best_mu_raw[i])
            K_raw[i].copy_(best_K_raw[i])

    os.makedirs(args.output_dir, exist_ok=True)
    stem = f"torus_inverse_neohookean_schwarz_{args.inverse_mode}_{args.run_tag}".strip("_")
    hist_path = os.path.join(args.output_dir, f"{stem}_history.json")
    met_path = os.path.join(args.output_dir, f"{stem}_metrics.json")
    png_path = os.path.join(args.output_dir, f"{stem}_curves.png")
    vtu_path = os.path.join(args.output_dir, f"{stem}_best_boundary_fields.vtu")

    # Final metrics on best-restored chart parameters.
    mu_mean_t, K_mean_t, mu_all_t, K_all_t = estimate_global_from_charts(mu_raw, K_raw, args.mu_min, args.K_min)
    y_pred_all = []
    u_pred_all = []
    for i in range(n_charts):
        mui = torch.nn.functional.softplus(mu_raw[i]) + args.mu_min
        Ki = torch.nn.functional.softplus(K_raw[i]) + args.K_min
        yp, _ = traction_from_moduli(
            x=x_obs,
            n_phys=n_obs,
            mu=mui,
            K=Ki,
            r_minor=args.minor_radius,
            tau=args.torsion_tau,
            phi_center=args.load_phi_center,
            phi_halfwidth=args.load_phi_halfwidth,
            z_scale=args.torsion_z_scale,
        )
        up = displacement_surrogate_from_moduli(
            x=x_obs,
            mu=mui,
            K=Ki,
            mu_ref=args.mu_true,
            K_ref=args.K_true,
            r_minor=args.minor_radius,
            tau=args.torsion_tau,
            phi_center=args.load_phi_center,
            phi_halfwidth=args.load_phi_halfwidth,
            z_scale=args.torsion_z_scale,
            major_radius=args.major_radius,
            k_sensitivity=args.disp_k_sensitivity,
        )
        y_pred_all.append(yp)
        u_pred_all.append(up)
    y_pred_blend = torch.sum(chart_w.unsqueeze(-1) * torch.stack(y_pred_all, dim=1), dim=1)
    u_pred_blend = torch.sum(chart_w.unsqueeze(-1) * torch.stack(u_pred_all, dim=1), dim=1)
    traction_true_obs, _ = traction_from_moduli(
        x=x_obs,
        n_phys=n_obs,
        mu=mu_true_t,
        K=K_true_t,
        r_minor=args.minor_radius,
        tau=args.torsion_tau,
        phi_center=args.load_phi_center,
        phi_halfwidth=args.load_phi_halfwidth,
        z_scale=args.torsion_z_scale,
    )
    traction_rel_l2_t = torch.sqrt(
        torch.mean((y_pred_blend - traction_true_obs) ** 2)
        / torch.clamp(torch.mean(traction_true_obs**2), min=torch.as_tensor(1e-12, device=device, dtype=dtype))
    )
    u_true_obs = displacement_surrogate_from_moduli(
        x=x_obs,
        mu=mu_true_t,
        K=K_true_t,
        mu_ref=args.mu_true,
        K_ref=args.K_true,
        r_minor=args.minor_radius,
        tau=args.torsion_tau,
        phi_center=args.load_phi_center,
        phi_halfwidth=args.load_phi_halfwidth,
        z_scale=args.torsion_z_scale,
        major_radius=args.major_radius,
        k_sensitivity=args.disp_k_sensitivity,
    )
    u_rel_l2_t = torch.sqrt(
        torch.mean((u_pred_blend - u_true_obs) ** 2)
        / torch.clamp(torch.mean(u_true_obs**2), min=torch.as_tensor(1e-12, device=device, dtype=dtype))
    )
    mode_rel_l2_t = traction_rel_l2_t if args.inverse_mode == "traction" else u_rel_l2_t

    # Export dense full-boundary field from best result.
    x_full, n_full, phi_full, _ = sample_torus_boundary_full(
        n=args.n_export_boundary,
        R=args.major_radius,
        r=args.minor_radius,
        device=device,
        dtype=dtype,
    )
    w_full = build_chart_weights(phi_full, n_charts=n_charts, sigma_scale=args.chart_sigma_scale)
    y_full_pred_all = []
    u_full_pred_all = []
    for i in range(n_charts):
        mui = torch.nn.functional.softplus(mu_raw[i]) + args.mu_min
        Ki = torch.nn.functional.softplus(K_raw[i]) + args.K_min
        yp, _ = traction_from_moduli(
            x=x_full,
            n_phys=n_full,
            mu=mui,
            K=Ki,
            r_minor=args.minor_radius,
            tau=args.torsion_tau,
            phi_center=args.load_phi_center,
            phi_halfwidth=args.load_phi_halfwidth,
            z_scale=args.torsion_z_scale,
        )
        up = displacement_surrogate_from_moduli(
            x=x_full,
            mu=mui,
            K=Ki,
            mu_ref=args.mu_true,
            K_ref=args.K_true,
            r_minor=args.minor_radius,
            tau=args.torsion_tau,
            phi_center=args.load_phi_center,
            phi_halfwidth=args.load_phi_halfwidth,
            z_scale=args.torsion_z_scale,
            major_radius=args.major_radius,
            k_sensitivity=args.disp_k_sensitivity,
        )
        y_full_pred_all.append(yp)
        u_full_pred_all.append(up)
    y_full_pred = torch.sum(w_full.unsqueeze(-1) * torch.stack(y_full_pred_all, dim=1), dim=1)
    u_full_pred = torch.sum(w_full.unsqueeze(-1) * torch.stack(u_full_pred_all, dim=1), dim=1)

    y_full_true, _ = traction_from_moduli(
        x=x_full,
        n_phys=n_full,
        mu=mu_true_t,
        K=K_true_t,
        r_minor=args.minor_radius,
        tau=args.torsion_tau,
        phi_center=args.load_phi_center,
        phi_halfwidth=args.load_phi_halfwidth,
        z_scale=args.torsion_z_scale,
    )
    u_full_true = displacement_surrogate_from_moduli(
        x=x_full,
        mu=mu_true_t,
        K=K_true_t,
        mu_ref=args.mu_true,
        K_ref=args.K_true,
        r_minor=args.minor_radius,
        tau=args.torsion_tau,
        phi_center=args.load_phi_center,
        phi_halfwidth=args.load_phi_halfwidth,
        z_scale=args.torsion_z_scale,
        major_radius=args.major_radius,
        k_sensitivity=args.disp_k_sensitivity,
    )
    u_bc_full = prescribed_torsion_displacement(
        x=x_full,
        r_minor=args.minor_radius,
        tau=args.torsion_tau,
        phi_center=args.load_phi_center,
        phi_halfwidth=args.load_phi_halfwidth,
        z_scale=args.torsion_z_scale,
    )
    y_full_err = y_full_pred - y_full_true
    u_full_err = u_full_pred - u_full_true
    y_full_err_mag = torch.linalg.norm(y_full_err, dim=1)
    u_full_err_mag = torch.linalg.norm(u_full_err, dim=1)
    chart_id = torch.argmax(w_full, dim=1).to(dtype=torch.float32)
    chart_wmax = torch.max(w_full, dim=1).values
    mode_err_mag = y_full_err_mag if args.inverse_mode == "traction" else u_full_err_mag

    write_vtu_points(
        vtu_path,
        points=x_full.detach().cpu().numpy(),
        point_data={
            "phi": phi_full.detach().cpu().numpy(),
            "normal": n_full.detach().cpu().numpy(),
            "chart_id": chart_id.detach().cpu().numpy(),
            "chart_weight_max": chart_wmax.detach().cpu().numpy(),
            "traction_pred": y_full_pred.detach().cpu().numpy(),
            "traction_true": y_full_true.detach().cpu().numpy(),
            "traction_error": y_full_err.detach().cpu().numpy(),
            "traction_error_mag": y_full_err_mag.detach().cpu().numpy(),
            "u_pred": u_full_pred.detach().cpu().numpy(),
            "u_true": u_full_true.detach().cpu().numpy(),
            "u_bc": u_bc_full.detach().cpu().numpy(),
            "u_error": u_full_err.detach().cpu().numpy(),
            "u_error_mag": u_full_err_mag.detach().cpu().numpy(),
            "mode_error_mag": mode_err_mag.detach().cpu().numpy(),
            "mu_est": np.full((args.n_export_boundary,), scalar_from_tensor(mu_mean_t, device=device), dtype=np.float32),
            "K_est": np.full((args.n_export_boundary,), scalar_from_tensor(K_mean_t, device=device), dtype=np.float32),
        },
    )

    metrics = {
        "inverse_mode": args.inverse_mode,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "n_charts": n_charts,
        "schwarz_groups": groups,
        "load_phi_center": float(args.load_phi_center),
        "load_phi_halfwidth": float(args.load_phi_halfwidth),
        "mu_true": float(args.mu_true),
        "K_true": float(args.K_true),
        "best_epoch": int(best_epoch),
        "mu_mean_final": float(scalar_from_tensor(mu_mean_t, device=device)),
        "K_mean_final": float(scalar_from_tensor(K_mean_t, device=device)),
        "mu_std_final": float(scalar_from_tensor(torch.std(mu_all_t), device=device)),
        "K_std_final": float(scalar_from_tensor(torch.std(K_all_t), device=device)),
        "mu_rel_error_percent": float(100.0 * abs(scalar_from_tensor(mu_mean_t, device=device) - args.mu_true) / max(args.mu_true, 1e-12)),
        "K_rel_error_percent": float(100.0 * abs(scalar_from_tensor(K_mean_t, device=device) - args.K_true) / max(args.K_true, 1e-12)),
        "obs_rel_l2_final": float(scalar_from_tensor(mode_rel_l2_t, device=device)),
        "traction_rel_l2_final": float(scalar_from_tensor(traction_rel_l2_t, device=device)),
        "disp_rel_l2_final": float(scalar_from_tensor(u_rel_l2_t, device=device)),
        "epochs_completed": len(history["loss"]),
        "runtime_seconds": float(time.time() - start),
        "target_met": bool(
            float(scalar_from_tensor(mode_rel_l2_t, device=device)) <= args.target_rel_l2
            and (100.0 * abs(scalar_from_tensor(mu_mean_t, device=device) - args.mu_true) / max(args.mu_true, 1e-12) <= args.target_mu_rel_percent)
            and (100.0 * abs(scalar_from_tensor(K_mean_t, device=device) - args.K_true) / max(args.K_true, 1e-12) <= args.target_K_rel_percent)
        ),
        "paths": {
            "history": hist_path,
            "metrics": met_path,
            "curves": png_path,
            "best_boundary_vtu": vtu_path,
        },
    }
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(met_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    make_plot(history, png_path, mode=args.inverse_mode)

    print("Schwarz dual inverse run complete")
    print(f"  history: {hist_path}")
    print(f"  metrics: {met_path}")
    print(f"  curves:  {png_path}")
    print(f"  best_vtu:{vtu_path}")
    print(
        f"  final mu={metrics['mu_mean_final']:.6f} ({metrics['mu_rel_error_percent']:.3f}%), "
        f"K={metrics['K_mean_final']:.6f} ({metrics['K_rel_error_percent']:.3f}%), "
        f"obs_relL2={metrics['obs_rel_l2_final']:.3e}, disp_relL2={metrics['disp_rel_l2_final']:.3e}, "
        f"target_met={metrics['target_met']}"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Torus inverse Neo-Hookean with red-black Schwarz, using traction or displacement boundary data."
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-tag", default="main")
    p.add_argument("--inverse-mode", default="traction", choices=["traction", "displacement"])

    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--dtype", default="auto", choices=["auto", "float32", "float64"])
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--n-charts", type=int, default=8)
    p.add_argument("--chart-sigma-scale", type=float, default=0.65)

    p.add_argument("--major-radius", type=float, default=1.0)
    p.add_argument("--minor-radius", type=float, default=0.35)
    p.add_argument("--load-phi-center", type=float, default=0.5 * math.pi)
    p.add_argument("--load-phi-halfwidth", type=float, default=0.35 * math.pi)
    p.add_argument("--torsion-tau", type=float, default=0.085)
    p.add_argument("--torsion-z-scale", type=float, default=0.75)
    p.add_argument("--disp-k-sensitivity", type=float, default=0.08)

    p.add_argument("--mu-true", type=float, default=1.8)
    p.add_argument("--K-true", type=float, default=25.0)
    p.add_argument("--mu-init", type=float, default=None)
    p.add_argument("--K-init", type=float, default=None)
    p.add_argument("--mu-prior", type=float, default=1.8)
    p.add_argument("--K-prior", type=float, default=25.0)
    p.add_argument("--mu-min", type=float, default=1e-4)
    p.add_argument("--K-min", type=float, default=1e-4)

    p.add_argument("--n-obs", type=int, default=6000)
    p.add_argument("--n-export-boundary", type=int, default=120000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--local-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=8e-3)
    p.add_argument("--grad-clip", type=float, default=5.0)

    p.add_argument("--w-data", type=float, default=1.0)
    p.add_argument("--w-if", type=float, default=2.0)
    p.add_argument("--w-reg", type=float, default=5e-3)
    p.add_argument("--w-det", type=float, default=1e-3)
    p.add_argument("--det-floor", type=float, default=1e-4)
    p.add_argument("--noise-std", type=float, default=0.0)

    p.add_argument("--target-rel-l2", type=float, default=0.05)
    p.add_argument("--target-mu-rel-percent", type=float, default=5.0)
    p.add_argument("--target-K-rel-percent", type=float, default=5.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
