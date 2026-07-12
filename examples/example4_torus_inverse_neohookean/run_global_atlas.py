#!/usr/bin/env python3
"""
Inverse Neo-Hookean parameter estimation on a 3D torus (donut) using an 8-chart atlas.

Setup:
- Domain: torus with major radius R and minor radius r.
- Load: localized torsional displacement field on a subsection in major-angle coordinate.
- Observation: traction vectors on boundary points within loaded subsection.
- Unknowns: global shear modulus mu and bulk modulus K.
- Atlas: 8 coordinate charts around the torus major loop with soft partition-of-unity weights.

The script prints guessed (estimated) moduli after every epoch.
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
        if not torch.cuda.is_available():
            print("Requested CUDA but unavailable; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    if device_arg == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            print("Requested MPS but unavailable; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "auto":
        return torch.float32 if device.type in ("cuda", "mps") else torch.float64
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float64":
        if device.type == "mps":
            raise RuntimeError("MPS + float64 is not recommended; use --dtype auto or float32.")
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype_arg}")


def wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def rand_uniform_cpu(
    n: int,
    low: float,
    high: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # MPS RNG can produce unstable samples on some builds; sample on CPU then move.
    r = torch.rand((n,), device=torch.device("cpu"), dtype=torch.float64)
    out = low + (high - low) * r
    return out.to(device=device, dtype=dtype)


def sanitize_tensor(x: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
    fill_t = torch.as_tensor(fill, device=x.device, dtype=x.dtype)
    return torch.nan_to_num(x, nan=fill_t, posinf=fill_t, neginf=fill_t)


def sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        return
    if device.type == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.synchronize()


def scalar_from_tensor(t: torch.Tensor, device: torch.device) -> float:
    sync_device(device)
    t_cpu = t.detach().to(device=torch.device("cpu"))
    if t_cpu.numel() == 0:
        raise ValueError("Cannot extract scalar from empty tensor.")
    return float(t_cpu.reshape(-1)[0].item())


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
    # phi: major-loop angle; theta: minor-tube angle; rho: distance from tube centerline.
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
    phi = rand_uniform_cpu(
        n=n,
        low=phi_center - phi_halfwidth,
        high=phi_center + phi_halfwidth,
        device=device,
        dtype=dtype,
    )
    theta = rand_uniform_cpu(
        n=n,
        low=0.0,
        high=2.0 * math.pi,
        device=device,
        dtype=dtype,
    )
    rho = torch.full((n,), float(r), device=device, dtype=dtype)
    x = sanitize_tensor(torus_from_angles(phi=phi, theta=theta, rho=rho, R=R))
    nvec = sanitize_tensor(torus_boundary_normals(phi=phi, theta=theta))
    phi_wrapped = sanitize_tensor(torch.remainder(phi, two_pi))
    theta = sanitize_tensor(theta)
    return x, nvec, phi_wrapped, theta


def sample_torus_boundary_full(
    n: int,
    R: float,
    r: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Stratified full-boundary sampling gives deterministic chart coverage and
    # avoids MPS random sampling artifacts.
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

    jitter_phi = (torch.rand((m,), device=torch.device("cpu"), dtype=torch.float64) - 0.5) * d_phi
    jitter_theta = (torch.rand((m,), device=torch.device("cpu"), dtype=torch.float64) - 0.5) * d_theta
    phi_grid = torch.remainder(phi_grid + jitter_phi, two_pi_f)
    theta_grid = torch.remainder(theta_grid + jitter_theta, two_pi_f)

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
    two_pi = torch.as_tensor(two_pi_f, device=device, dtype=dtype)
    rho = torch.full((n,), float(r), device=device, dtype=dtype)
    x = sanitize_tensor(torus_from_angles(phi=phi, theta=theta, rho=rho, R=R))
    nvec = sanitize_tensor(torus_boundary_normals(phi=phi, theta=theta))
    phi_wrapped = sanitize_tensor(torch.remainder(phi, two_pi))
    theta = sanitize_tensor(theta)
    return x, nvec, phi_wrapped, theta


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


def build_chart_weights(phi: torch.Tensor, n_charts: int = 8, sigma_scale: float = 0.65) -> torch.Tensor:
    two_pi = torch.as_tensor(2.0 * math.pi, device=phi.device, dtype=phi.dtype)
    centers = torch.linspace(0.0, float(two_pi.item()), steps=n_charts + 1, device=phi.device, dtype=phi.dtype)[:-1]
    d = wrap_to_pi(phi.unsqueeze(1) - centers.unsqueeze(0))
    sigma = torch.as_tensor(sigma_scale * (2.0 * math.pi / n_charts), device=phi.device, dtype=phi.dtype)
    logits = -torch.as_tensor(0.5, device=phi.device, dtype=phi.dtype) * (d / sigma) ** 2
    w = torch.softmax(logits, dim=1)
    return w


def build_chart_local_coords(
    phi: torch.Tensor,
    theta: torch.Tensor,
    n_charts: int = 8,
) -> torch.Tensor:
    two_pi = torch.as_tensor(2.0 * math.pi, device=phi.device, dtype=phi.dtype)
    pi_t = torch.as_tensor(math.pi, device=phi.device, dtype=phi.dtype)
    centers = torch.linspace(0.0, float(two_pi.item()), steps=n_charts + 1, device=phi.device, dtype=phi.dtype)[:-1]
    d = wrap_to_pi(phi.unsqueeze(1) - centers.unsqueeze(0))
    u = d / pi_t
    v = wrap_to_pi(theta).unsqueeze(1).repeat(1, n_charts) / pi_t
    w = torch.zeros_like(u)
    return torch.stack([u, v, w], dim=2)


def softplus_inv(y: float) -> float:
    y = max(y, 1e-8)
    return math.log(math.expm1(y))


def clamp_init(val: float, target: float, frac: float, name: str) -> float:
    lo = target * (1.0 - frac)
    hi = target * (1.0 + frac)
    if val < lo or val > hi:
        clamped = min(max(val, lo), hi)
        print(
            f"[Init] {name}={val:.6f} outside ±{100.0*frac:.1f}% of target {target:.6f}; "
            f"clamped to {clamped:.6f}"
        )
        return clamped
    return val


def make_plot(history: Dict[str, List[float]], out_path: str) -> None:
    ep = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].semilogy(ep, np.maximum(history["loss"], 1e-16), label="total")
    axes[0].semilogy(ep, np.maximum(history["traction"], 1e-16), label="traction")
    axes[0].semilogy(ep, np.maximum(history["det_barrier"], 1e-16), label="det_barrier")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Inverse Losses (Torus)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(ep, history["mu_guess"], label="mu_guess", color="tab:red")
    axes[1].plot(ep, history["K_guess"], label="K_guess", color="tab:blue")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Estimated Modulus")
    axes[1].set_title("Modulus Estimates")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_vtu_points(path: str, points: np.ndarray, point_data: Dict[str, np.ndarray]) -> None:
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be [N,3]")
    n0 = int(points.shape[0])
    if n0 <= 0:
        raise ValueError("points must be non-empty")

    # Drop invalid rows consistently across points and all point_data arrays.
    valid = np.all(np.isfinite(points), axis=1)
    cleaned: Dict[str, np.ndarray] = {}
    for name, arr in point_data.items():
        a = np.asarray(arr)
        if a.shape[0] != n0:
            raise ValueError(f"point_data '{name}' first dimension must be {n0}, got {a.shape[0]}")
        if a.ndim == 1:
            finite = np.isfinite(a)
        elif a.ndim == 2:
            finite = np.all(np.isfinite(a), axis=1)
        else:
            raise ValueError(f"Unsupported point_data shape for '{name}': {a.shape}")
        valid &= finite
        cleaned[name] = a

    valid_count = int(np.count_nonzero(valid))
    if valid_count <= 0:
        raise ValueError(f"All points are invalid for VTU export: {path}")
    dropped = n0 - valid_count
    if dropped > 0:
        print(f"[VTU] Dropping {dropped}/{n0} non-finite rows for {os.path.basename(path)}")

    points = points[valid]
    # Force float32 for compatibility and smaller files.
    if np.issubdtype(points.dtype, np.floating):
        points = points.astype(np.float32, copy=False)
    n = int(points.shape[0])

    for name, arr in cleaned.items():
        a = arr[valid]
        if np.issubdtype(a.dtype, np.floating):
            a = a.astype(np.float32, copy=False)
        cleaned[name] = a

    os.makedirs(os.path.dirname(path), exist_ok=True)

    connectivity = np.arange(n, dtype=np.int32)
    offsets = np.arange(1, n + 1, dtype=np.int32)
    cell_types = np.ones(n, dtype=np.uint8)  # VTK_VERTEX

    def vtk_type(arr: np.ndarray) -> str:
        a = np.asarray(arr)
        if np.issubdtype(a.dtype, np.integer):
            if a.dtype.itemsize <= 4:
                return "Int32"
            return "Int64"
        if a.dtype == np.float32:
            return "Float32"
        return "Float64"

    def write_ascii_dataarray_lines(fh, arr: np.ndarray, indent: str = "          ", vals_per_line: int = 24) -> None:
        flat = np.asarray(arr).reshape(-1)
        nvals = int(flat.shape[0])
        for s in range(0, nvals, vals_per_line):
            e = min(nvals, s + vals_per_line)
            fh.write(indent + " ".join(map(str, flat[s:e])) + "\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write(f'    <Piece NumberOfPoints="{n}" NumberOfCells="{n}">\n')
        f.write("      <Points>\n")
        f.write(f'        <DataArray type="{vtk_type(points)}" NumberOfComponents="3" format="ascii">\n')
        write_ascii_dataarray_lines(f, points)
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")
        f.write("      <Cells>\n")
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        write_ascii_dataarray_lines(f, connectivity)
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        write_ascii_dataarray_lines(f, offsets)
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        write_ascii_dataarray_lines(f, cell_types)
        f.write("        </DataArray>\n")
        f.write("      </Cells>\n")
        f.write("      <PointData>\n")
        for name, arr in cleaned.items():
            if arr.ndim == 1:
                ncomp = 1
            elif arr.ndim == 2:
                ncomp = int(arr.shape[1])
            else:
                raise ValueError(f"Unsupported point_data shape for '{name}': {arr.shape}")
            f.write(
                f'        <DataArray type="{vtk_type(arr)}" Name="{name}" '
                f'NumberOfComponents="{ncomp}" format="ascii">\n'
            )
            write_ascii_dataarray_lines(f, arr)
            f.write("        </DataArray>\n")
        f.write("      </PointData>\n")
        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")


def run(args: argparse.Namespace) -> Dict[str, float]:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device=device)
    # Keep implicit scalar/tensor construction consistent with requested runtime dtype.
    torch.set_default_dtype(dtype)
    n_charts = 8
    print(f"Device={device.type} dtype={dtype} n_charts={n_charts}")

    mu_init = args.mu_init if args.mu_init is not None else args.mu_true * (1.0 - 0.03)
    K_init = args.K_init if args.K_init is not None else args.K_true * (1.0 + 0.04)
    mu_init = clamp_init(mu_init, args.mu_true, 0.05, "mu_init")
    K_init = clamp_init(K_init, args.K_true, 0.05, "K_init")

    mu_raw = torch.nn.Parameter(
        torch.tensor(softplus_inv(mu_init - args.mu_min), device=device, dtype=dtype)
    )
    K_raw = torch.nn.Parameter(
        torch.tensor(softplus_inv(K_init - args.K_min), device=device, dtype=dtype)
    )
    optimizer = torch.optim.Adam([mu_raw, K_raw], lr=args.lr)

    # Fixed traction observations from the loaded boundary subsection.
    x_obs, n_obs, phi_obs, theta_obs = sample_torus_boundary_subsection(
        n=args.n_obs,
        R=args.major_radius,
        r=args.minor_radius,
        phi_center=args.load_phi_center,
        phi_halfwidth=args.load_phi_halfwidth,
        device=device,
        dtype=dtype,
    )
    chart_weights = build_chart_weights(phi_obs, n_charts=n_charts, sigma_scale=args.chart_sigma_scale).detach()
    chart_local_coords = build_chart_local_coords(phi_obs, theta_obs, n_charts=n_charts).detach()

    mu_true_t = torch.tensor(args.mu_true, device=device, dtype=dtype)
    K_true_t = torch.tensor(args.K_true, device=device, dtype=dtype)
    t_true, _ = traction_from_moduli(
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
    if args.traction_noise_std > 0.0:
        t_true = t_true + args.traction_noise_std * torch.std(t_true) * torch.randn_like(t_true)
    t_true = t_true.detach()

    history: Dict[str, List[float]] = {
        "loss": [],
        "traction": [],
        "det_barrier": [],
        "mu_guess": [],
        "K_guess": [],
    }

    start = time.time()
    best_loss = float("inf")
    stale = 0
    det_floor_t = torch.as_tensor(args.det_floor, device=device, dtype=dtype)
    det_barrier_weight_t = torch.as_tensor(args.det_barrier_weight, device=device, dtype=dtype)

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        sel = torch.randint(0, args.n_obs, (min(args.batch_size, args.n_obs),), device=device)
        xb = x_obs[sel]
        nb = n_obs[sel]
        tb = t_true[sel]
        wb = chart_weights[sel]

        mu_est = torch.nn.functional.softplus(mu_raw) + args.mu_min
        K_est = torch.nn.functional.softplus(K_raw) + args.K_min
        t_pred, det_f = traction_from_moduli(
            x=xb,
            n_phys=nb,
            mu=mu_est,
            K=K_est,
            r_minor=args.minor_radius,
            tau=args.torsion_tau,
            phi_center=args.load_phi_center,
            phi_halfwidth=args.load_phi_halfwidth,
            z_scale=args.torsion_z_scale,
        )

        per_point = torch.mean((t_pred - tb) ** 2, dim=1)
        num = torch.sum(wb * per_point.unsqueeze(1), dim=0)
        den = torch.sum(wb, dim=0) + torch.as_tensor(1e-12, device=device, dtype=dtype)
        per_chart = num / den
        loss_traction = torch.mean(per_chart)
        loss_det_barrier = torch.mean(torch.nn.functional.softplus(det_floor_t - det_f) ** 2)
        loss_total = loss_traction + det_barrier_weight_t * loss_det_barrier

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_([mu_raw, K_raw], max_norm=5.0)
        optimizer.step()

        lv = float(loss_total.item())
        history["loss"].append(lv)
        history["traction"].append(float(loss_traction.item()))
        history["det_barrier"].append(float(loss_det_barrier.item()))
        mu_guess = float((torch.nn.functional.softplus(mu_raw) + args.mu_min).item())
        K_guess = float((torch.nn.functional.softplus(K_raw) + args.K_min).item())
        history["mu_guess"].append(mu_guess)
        history["K_guess"].append(K_guess)

        mu_err = 100.0 * abs(mu_guess - args.mu_true) / max(args.mu_true, 1e-12)
        K_err = 100.0 * abs(K_guess - args.K_true) / max(args.K_true, 1e-12)
        elapsed = time.time() - start
        print(
            f"Epoch {epoch:04d}/{args.epochs} | loss={lv:.4e} traction={history['traction'][-1]:.4e} "
            f"| mu_guess={mu_guess:.6f} ({mu_err:.3f}% err) "
            f"| K_guess={K_guess:.6f} ({K_err:.3f}% err) | t={elapsed:.1f}s"
        )

        if lv + args.min_delta < best_loss:
            best_loss = lv
            stale = 0
        else:
            stale += 1

        if args.target_loss is not None and lv <= args.target_loss:
            print(f"Stopping early: target_loss={args.target_loss:.3e} reached at epoch {epoch}.")
            break
        if stale >= args.patience:
            print(f"Stopping early: plateau patience {args.patience} reached at epoch {epoch}.")
            break

    # Holdout evaluation on fresh boundary points in the loaded subsection.
    x_eval, n_eval, phi_eval, theta_eval = sample_torus_boundary_subsection(
        n=args.n_eval,
        R=args.major_radius,
        r=args.minor_radius,
        phi_center=args.load_phi_center,
        phi_halfwidth=args.load_phi_halfwidth,
        device=device,
        dtype=dtype,
    )
    w_eval = build_chart_weights(phi_eval, n_charts=n_charts, sigma_scale=args.chart_sigma_scale).detach()
    chart_local_coords_eval = build_chart_local_coords(phi_eval, theta_eval, n_charts=n_charts).detach()
    t_eval_true, _ = traction_from_moduli(
        x=x_eval,
        n_phys=n_eval,
        mu=mu_true_t,
        K=K_true_t,
        r_minor=args.minor_radius,
        tau=args.torsion_tau,
        phi_center=args.load_phi_center,
        phi_halfwidth=args.load_phi_halfwidth,
        z_scale=args.torsion_z_scale,
    )
    mu_final_raw = torch.nn.functional.softplus(mu_raw.detach()) + args.mu_min
    K_final_raw = torch.nn.functional.softplus(K_raw.detach()) + args.K_min
    mu_final_value = scalar_from_tensor(mu_final_raw, device=device)
    K_final_value = scalar_from_tensor(K_final_raw, device=device)
    if history["mu_guess"] and history["K_guess"]:
        mu_last = float(history["mu_guess"][-1])
        K_last = float(history["K_guess"][-1])
        mu_rel_diff = abs(mu_final_value - mu_last) / max(abs(mu_last), 1e-12)
        K_rel_diff = abs(K_final_value - K_last) / max(abs(K_last), 1e-12)
        if (
            (not math.isfinite(mu_final_value))
            or (not math.isfinite(K_final_value))
            or mu_final_value <= 0.0
            or K_final_value <= 0.0
            or mu_rel_diff > 5e-2
            or K_rel_diff > 5e-2
        ):
            print(
                "[Warn] Final parameter readback inconsistent with last epoch; "
                "using last-epoch modulus estimates for evaluation/export."
            )
            mu_final_value = mu_last
            K_final_value = K_last
    mu_final = torch.as_tensor(mu_final_value, device=device, dtype=dtype)
    K_final = torch.as_tensor(K_final_value, device=device, dtype=dtype)
    t_eval_pred, _ = traction_from_moduli(
        x=x_eval,
        n_phys=n_eval,
        mu=mu_final,
        K=K_final,
        r_minor=args.minor_radius,
        tau=args.torsion_tau,
        phi_center=args.load_phi_center,
        phi_halfwidth=args.load_phi_halfwidth,
        z_scale=args.torsion_z_scale,
    )
    t_obs_pred, _ = traction_from_moduli(
        x=x_obs,
        n_phys=n_obs,
        mu=mu_final,
        K=K_final,
        r_minor=args.minor_radius,
        tau=args.torsion_tau,
        phi_center=args.load_phi_center,
        phi_halfwidth=args.load_phi_halfwidth,
        z_scale=args.torsion_z_scale,
    )

    t_eval_err = t_eval_pred - t_eval_true
    t_obs_err = t_obs_pred - t_true
    t_eval_err_mag = torch.linalg.norm(t_eval_err, dim=1)
    t_obs_err_mag = torch.linalg.norm(t_obs_err, dim=1)
    traction_rel_l2_t = torch.sqrt(
        torch.mean((t_eval_pred - t_eval_true) ** 2)
        / torch.clamp(
            torch.mean(t_eval_true**2),
            min=torch.as_tensor(1e-12, device=device, dtype=dtype),
        )
    )
    traction_rel_l2 = scalar_from_tensor(traction_rel_l2_t, device=device)
    per_point_eval = torch.mean((t_eval_pred - t_eval_true) ** 2, dim=1)
    per_chart_eval = torch.sum(w_eval * per_point_eval.unsqueeze(1), dim=0) / (
        torch.sum(w_eval, dim=0) + torch.as_tensor(1e-12, device=device, dtype=dtype)
    )

    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.join(args.output_dir, "torus_inverse_neohookean_atlas_curves.png")
    metrics_path = os.path.join(args.output_dir, "torus_inverse_neohookean_atlas_metrics.json")
    history_path = os.path.join(args.output_dir, "torus_inverse_neohookean_atlas_history.json")
    model_path = os.path.join(args.output_dir, "torus_inverse_neohookean_atlas_state.pt")
    obs_path = os.path.join(args.output_dir, "torus_inverse_neohookean_atlas_obs.npz")
    eval_vtu_path = os.path.join(args.output_dir, "torus_inverse_neohookean_atlas_boundary_eval_subsection_error.vtu")
    obs_vtu_path = os.path.join(args.output_dir, "torus_inverse_neohookean_atlas_boundary_obs_subsection_error.vtu")
    full_vtu_path = os.path.join(args.output_dir, "torus_inverse_neohookean_atlas_boundary_full_error.vtu")
    support_vtu_path = os.path.join(args.output_dir, "torus_inverse_neohookean_atlas_boundary_full_chart_support.vtu")
    # Backward-compatible aliases.
    eval_vtu_legacy_path = os.path.join(args.output_dir, "torus_inverse_neohookean_atlas_boundary_eval_error.vtu")
    obs_vtu_legacy_path = os.path.join(args.output_dir, "torus_inverse_neohookean_atlas_boundary_obs_error.vtu")

    make_plot(history, plot_path)

    chart_id_eval = torch.argmax(w_eval, dim=1).to(dtype=torch.float32)
    chart_id_obs = torch.argmax(chart_weights, dim=1).to(dtype=torch.float32)
    chart_wmax_eval = torch.max(w_eval, dim=1).values
    chart_wmax_obs = torch.max(chart_weights, dim=1).values

    write_vtu_points(
        eval_vtu_path,
        points=x_eval.detach().cpu().numpy(),
        point_data={
            "phi": phi_eval.detach().cpu().numpy(),
            "theta": theta_eval.detach().cpu().numpy(),
            "chart_id": chart_id_eval.detach().cpu().numpy(),
            "chart_weight_max": chart_wmax_eval.detach().cpu().numpy(),
            "traction_true": t_eval_true.detach().cpu().numpy(),
            "traction_pred": t_eval_pred.detach().cpu().numpy(),
            "traction_error": t_eval_err.detach().cpu().numpy(),
            "traction_error_mag": t_eval_err_mag.detach().cpu().numpy(),
            **{f"chart_weight_{i:02d}": w_eval[:, i].detach().cpu().numpy() for i in range(n_charts)},
        },
    )
    write_vtu_points(
        eval_vtu_legacy_path,
        points=x_eval.detach().cpu().numpy(),
        point_data={
            "phi": phi_eval.detach().cpu().numpy(),
            "theta": theta_eval.detach().cpu().numpy(),
            "chart_id": chart_id_eval.detach().cpu().numpy(),
            "chart_weight_max": chart_wmax_eval.detach().cpu().numpy(),
            "traction_true": t_eval_true.detach().cpu().numpy(),
            "traction_pred": t_eval_pred.detach().cpu().numpy(),
            "traction_error": t_eval_err.detach().cpu().numpy(),
            "traction_error_mag": t_eval_err_mag.detach().cpu().numpy(),
            **{f"chart_weight_{i:02d}": w_eval[:, i].detach().cpu().numpy() for i in range(n_charts)},
        },
    )
    write_vtu_points(
        obs_vtu_path,
        points=x_obs.detach().cpu().numpy(),
        point_data={
            "phi": phi_obs.detach().cpu().numpy(),
            "theta": theta_obs.detach().cpu().numpy(),
            "chart_id": chart_id_obs.detach().cpu().numpy(),
            "chart_weight_max": chart_wmax_obs.detach().cpu().numpy(),
            "traction_true": t_true.detach().cpu().numpy(),
            "traction_pred": t_obs_pred.detach().cpu().numpy(),
            "traction_error": t_obs_err.detach().cpu().numpy(),
            "traction_error_mag": t_obs_err_mag.detach().cpu().numpy(),
            **{f"chart_weight_{i:02d}": chart_weights[:, i].detach().cpu().numpy() for i in range(n_charts)},
        },
    )
    write_vtu_points(
        obs_vtu_legacy_path,
        points=x_obs.detach().cpu().numpy(),
        point_data={
            "phi": phi_obs.detach().cpu().numpy(),
            "theta": theta_obs.detach().cpu().numpy(),
            "chart_id": chart_id_obs.detach().cpu().numpy(),
            "chart_weight_max": chart_wmax_obs.detach().cpu().numpy(),
            "traction_true": t_true.detach().cpu().numpy(),
            "traction_pred": t_obs_pred.detach().cpu().numpy(),
            "traction_error": t_obs_err.detach().cpu().numpy(),
            "traction_error_mag": t_obs_err_mag.detach().cpu().numpy(),
            **{f"chart_weight_{i:02d}": chart_weights[:, i].detach().cpu().numpy() for i in range(n_charts)},
        },
    )

    # Dense full-boundary export so all charts are represented in ParaView.
    n_export_full = max(args.n_export_points, 8 * 2000)
    x_full, n_full, phi_full, theta_full = sample_torus_boundary_full(
        n=n_export_full,
        R=args.major_radius,
        r=args.minor_radius,
        device=device,
        dtype=dtype,
    )
    w_full = build_chart_weights(phi_full, n_charts=n_charts, sigma_scale=args.chart_sigma_scale).detach()
    chart_local_coords_full = build_chart_local_coords(phi_full, theta_full, n_charts=n_charts).detach()
    chart_id_full_i = torch.argmax(w_full, dim=1)
    chart_id_full = chart_id_full_i.to(dtype=torch.float32)
    chart_wmax_full = torch.max(w_full, dim=1).values

    t_full_true, _ = traction_from_moduli(
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
    t_full_pred, _ = traction_from_moduli(
        x=x_full,
        n_phys=n_full,
        mu=mu_final,
        K=K_final,
        r_minor=args.minor_radius,
        tau=args.torsion_tau,
        phi_center=args.load_phi_center,
        phi_halfwidth=args.load_phi_halfwidth,
        z_scale=args.torsion_z_scale,
    )
    t_full_err = t_full_pred - t_full_true
    t_full_err_mag = torch.linalg.norm(t_full_err, dim=1)
    u_full = prescribed_torsion_displacement(
        x=x_full,
        r_minor=args.minor_radius,
        tau=args.torsion_tau,
        phi_center=args.load_phi_center,
        phi_halfwidth=args.load_phi_halfwidth,
        z_scale=args.torsion_z_scale,
    )
    u_full_mag = torch.linalg.norm(u_full, dim=1)
    x_full_deformed = x_full + u_full
    idx_full = torch.arange(x_full.shape[0], device=device)
    chart_u_dom = chart_local_coords_full[idx_full, chart_id_full_i, 0]
    chart_v_dom = chart_local_coords_full[idx_full, chart_id_full_i, 1]
    chart_w_dom = chart_local_coords_full[idx_full, chart_id_full_i, 2]
    support_thresh = torch.as_tensor(1.0 / (2.0 * n_charts), device=device, dtype=dtype)
    support_masks = (w_full >= support_thresh).to(dtype=torch.float32)

    write_vtu_points(
        full_vtu_path,
        points=x_full.detach().cpu().numpy(),
        point_data={
            "phi": phi_full.detach().cpu().numpy(),
            "theta": theta_full.detach().cpu().numpy(),
            "chart_id": chart_id_full.detach().cpu().numpy(),
            "chart_weight_max": chart_wmax_full.detach().cpu().numpy(),
            "traction_true": t_full_true.detach().cpu().numpy(),
            "traction_pred": t_full_pred.detach().cpu().numpy(),
            "traction_error": t_full_err.detach().cpu().numpy(),
            "traction_error_mag": t_full_err_mag.detach().cpu().numpy(),
            "displacement": u_full.detach().cpu().numpy(),
            "displacement_mag": u_full_mag.detach().cpu().numpy(),
            "x_deformed": x_full_deformed.detach().cpu().numpy(),
            **{f"chart_weight_{i:02d}": w_full[:, i].detach().cpu().numpy() for i in range(n_charts)},
            "chart_u_dominant": chart_u_dom.detach().cpu().numpy(),
            "chart_v_dominant": chart_v_dom.detach().cpu().numpy(),
            "chart_w_dominant": chart_w_dom.detach().cpu().numpy(),
        },
    )
    write_vtu_points(
        support_vtu_path,
        points=x_full.detach().cpu().numpy(),
        point_data={
            "phi": phi_full.detach().cpu().numpy(),
            "theta": theta_full.detach().cpu().numpy(),
            "chart_id": chart_id_full.detach().cpu().numpy(),
            "chart_weight_max": chart_wmax_full.detach().cpu().numpy(),
            "traction_error_mag": t_full_err_mag.detach().cpu().numpy(),
            "displacement": u_full.detach().cpu().numpy(),
            "displacement_mag": u_full_mag.detach().cpu().numpy(),
            "x_deformed": x_full_deformed.detach().cpu().numpy(),
            **{f"chart_weight_{i:02d}": w_full[:, i].detach().cpu().numpy() for i in range(n_charts)},
            **{f"chart_support_{i:02d}": support_masks[:, i].detach().cpu().numpy() for i in range(n_charts)},
        },
    )

    per_chart_vtu_paths: List[str] = []
    for i in range(n_charts):
        mask = w_full[:, i] >= torch.as_tensor(1.0 / (2.0 * n_charts), device=device, dtype=dtype)
        if int(torch.sum(mask).item()) == 0:
            mask = chart_id_full_i == i
        n_i = int(torch.sum(mask).item())
        if n_i == 0:
            continue
        p = os.path.join(
            args.output_dir, f"torus_inverse_neohookean_atlas_boundary_full_chart_{i:02d}.vtu"
        )
        write_vtu_points(
            p,
            points=x_full[mask].detach().cpu().numpy(),
            point_data={
                "phi": phi_full[mask].detach().cpu().numpy(),
                "theta": theta_full[mask].detach().cpu().numpy(),
                "chart_id": chart_id_full[mask].detach().cpu().numpy(),
                "chart_weight_max": chart_wmax_full[mask].detach().cpu().numpy(),
                "traction_true": t_full_true[mask].detach().cpu().numpy(),
                "traction_pred": t_full_pred[mask].detach().cpu().numpy(),
                "traction_error": t_full_err[mask].detach().cpu().numpy(),
                "traction_error_mag": t_full_err_mag[mask].detach().cpu().numpy(),
                "displacement": u_full[mask].detach().cpu().numpy(),
                "displacement_mag": u_full_mag[mask].detach().cpu().numpy(),
                "x_deformed": x_full_deformed[mask].detach().cpu().numpy(),
                "chart_weight_local": w_full[mask, i].detach().cpu().numpy(),
                "chart_u_local": chart_local_coords_full[mask, i, 0].detach().cpu().numpy(),
                "chart_v_local": chart_local_coords_full[mask, i, 1].detach().cpu().numpy(),
                "chart_w_local": chart_local_coords_full[mask, i, 2].detach().cpu().numpy(),
            },
        )
        per_chart_vtu_paths.append(p)

    metrics = {
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "n_charts": n_charts,
        "mu_true": float(args.mu_true),
        "K_true": float(args.K_true),
        "mu_init": float(mu_init),
        "K_init": float(K_init),
        "mu_final": float(mu_final_value),
        "K_final": float(K_final_value),
        "mu_rel_error_percent": float(100.0 * abs(mu_final_value - args.mu_true) / max(args.mu_true, 1e-12)),
        "K_rel_error_percent": float(100.0 * abs(K_final_value - args.K_true) / max(args.K_true, 1e-12)),
        "traction_rel_l2": traction_rel_l2,
        "per_chart_eval_mse": per_chart_eval.detach().to(device=torch.device("cpu")).numpy().tolist(),
        "paraview_eval_vtu": eval_vtu_path,
        "paraview_obs_vtu": obs_vtu_path,
        "paraview_full_boundary_vtu": full_vtu_path,
        "paraview_full_boundary_support_vtu": support_vtu_path,
        "paraview_full_boundary_per_chart_vtu": per_chart_vtu_paths,
        "n_export_points": int(n_export_full),
        "epochs_completed": len(history["loss"]),
        "runtime_seconds": float(time.time() - start),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    torch.save(
        {
            "mu_raw": mu_raw.detach().to(device=torch.device("cpu")),
            "K_raw": K_raw.detach().to(device=torch.device("cpu")),
            "mu_final": float(mu_final_value),
            "K_final": float(K_final_value),
            "args": vars(args),
            "metrics": metrics,
            "history": history,
        },
        model_path,
    )

    np.savez_compressed(
        obs_path,
        x_obs=x_obs.detach().cpu().numpy(),
        n_obs=n_obs.detach().cpu().numpy(),
        t_true=t_true.detach().cpu().numpy(),
        chart_weights=chart_weights.detach().cpu().numpy(),
        chart_local_coords=chart_local_coords.detach().cpu().numpy(),
        x_eval=x_eval.detach().cpu().numpy(),
        n_eval=n_eval.detach().cpu().numpy(),
        t_eval_true=t_eval_true.detach().cpu().numpy(),
        t_eval_pred=t_eval_pred.detach().cpu().numpy(),
        chart_weights_eval=w_eval.detach().cpu().numpy(),
        chart_local_coords_eval=chart_local_coords_eval.detach().cpu().numpy(),
        x_full=x_full.detach().cpu().numpy(),
        n_full=n_full.detach().cpu().numpy(),
        t_full_true=t_full_true.detach().cpu().numpy(),
        t_full_pred=t_full_pred.detach().cpu().numpy(),
        t_full_error=t_full_err.detach().cpu().numpy(),
        u_full=u_full.detach().cpu().numpy(),
        u_full_mag=u_full_mag.detach().cpu().numpy(),
        x_full_deformed=x_full_deformed.detach().cpu().numpy(),
        chart_weights_full=w_full.detach().cpu().numpy(),
        chart_local_coords_full=chart_local_coords_full.detach().cpu().numpy(),
    )

    print("Inverse torus run complete")
    print(f"  curves:   {plot_path}")
    print(f"  metrics:  {metrics_path}")
    print(f"  history:  {history_path}")
    print(f"  model:    {model_path}")
    print(f"  obs:      {obs_path}")
    print(f"  eval_vtu: {eval_vtu_path}")
    print(f"  obs_vtu:  {obs_vtu_path}")
    print(f"  full_vtu: {full_vtu_path}")
    print(f"  support_vtu: {support_vtu_path}")
    print("  note: *_subsection_* VTUs contain only the loaded torus arc; use *_full_error.vtu for full torus.")
    print(f"  chart_vtu_files: {len(per_chart_vtu_paths)}")
    print(f"  mu_true={args.mu_true:.6f}, mu_final={mu_final_value:.6f}")
    print(f"  K_true={args.K_true:.6f}, K_final={K_final_value:.6f}")
    print(f"  traction_rel_l2={traction_rel_l2:.6e}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inverse Neo-Hookean moduli on 3D torus with 8-chart atlas.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float64"])

    parser.add_argument("--major-radius", type=float, default=1.0)
    parser.add_argument("--minor-radius", type=float, default=0.35)
    parser.add_argument("--load-phi-center", type=float, default=0.5 * math.pi)
    parser.add_argument("--load-phi-halfwidth", type=float, default=0.35 * math.pi)
    parser.add_argument("--torsion-tau", type=float, default=0.085)
    parser.add_argument("--torsion-z-scale", type=float, default=0.75)

    parser.add_argument("--mu-true", type=float, default=1.8)
    parser.add_argument("--K-true", type=float, default=25.0)
    parser.add_argument("--mu-init", type=float, default=None)
    parser.add_argument("--K-init", type=float, default=None)
    parser.add_argument("--mu-min", type=float, default=1e-4)
    parser.add_argument("--K-min", type=float, default=1e-4)

    parser.add_argument("--n-obs", type=int, default=6000)
    parser.add_argument("--n-eval", type=int, default=3000)
    parser.add_argument(
        "--n-export-points",
        type=int,
        default=24000,
        help="Dense full-boundary sample count for ParaView error-map export.",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--det-floor", type=float, default=1e-4)
    parser.add_argument("--det-barrier-weight", type=float, default=1e-2)
    parser.add_argument("--chart-sigma-scale", type=float, default=0.65)
    parser.add_argument("--traction-noise-std", type=float, default=0.0)

    parser.add_argument("--target-loss", type=float, default=None)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--min-delta", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
