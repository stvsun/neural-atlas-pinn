"""
3D PINN example:
Poisson equation in an ellipsoid, mapped to the unit sphere.

Physical domain (x, y, z):
    E = {(x/a)^2 + (y/b)^2 + (z/c)^2 < 1}

Mapping to reference sphere coordinates (xi, eta, zeta):
    x = a*xi, y = b*eta, z = c*zeta
    B = {xi^2 + eta^2 + zeta^2 < 1}

Physical PDE:
    -Delta_x u = f in E, u = g on dE

Mapped PDE on B (constant-coefficient anisotropic diffusion):
    -[(1/a^2) u_xixi + (1/b^2) u_etaeta + (1/c^2) u_zetazeta] = f(a*xi, b*eta, c*zeta)
    u = g(a*xi, b*eta, c*zeta) on xi^2 + eta^2 + zeta^2 = 1

Manufactured solution used here:
    u_exact(xi, eta, zeta) = 1 - (xi^2 + eta^2 + zeta^2)
    f_map = 2*(1/a^2 + 1/b^2 + 1/c^2)
    BC: u = 0 on the unit sphere boundary.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import numpy as np
import torch

# Make the repository root importable so ``mapped_sphere.*`` resolves regardless of
# how this script is invoked. Once the repo is installed as a package this
# block can be deleted.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mapped_sphere.device import resolve_device, resolve_dtype
from mapped_sphere.utils import set_seed


class PINN3D(torch.nn.Module):
    def __init__(self, width=64, depth=4):
        super().__init__()
        layers = [torch.nn.Linear(3, width)]
        for _ in range(depth - 1):
            layers.append(torch.nn.Linear(width, width))
        self.hidden = torch.nn.ModuleList(layers)
        self.out = torch.nn.Linear(width, 1)

        for layer in self.hidden:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        torch.nn.init.xavier_normal_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h = x
        for layer in self.hidden:
            h = torch.tanh(layer(h))
        return self.out(h)


def sample_interior_unit_ball(n_points, device, dtype):
    direction = torch.randn((n_points, 3), device=device, dtype=dtype)
    direction = direction / torch.clamp(torch.linalg.norm(direction, dim=1, keepdim=True), min=1e-12)
    radius = torch.rand((n_points, 1), device=device, dtype=dtype) ** (1.0 / 3.0)
    return radius * direction


def sample_boundary_unit_sphere(n_points, device, dtype):
    direction = torch.randn((n_points, 3), device=device, dtype=dtype)
    direction = direction / torch.clamp(torch.linalg.norm(direction, dim=1, keepdim=True), min=1e-12)
    return direction


def exact_solution_mapped(x_ref):
    return 1.0 - torch.sum(x_ref**2, dim=1, keepdim=True)


def source_term_mapped_constant(a_axis, b_axis, c_axis, device, dtype):
    val = 2.0 * ((1.0 / (a_axis**2)) + (1.0 / (b_axis**2)) + (1.0 / (c_axis**2)))
    return torch.tensor(val, device=device, dtype=dtype)


def mapped_poisson_residual(model, x_ref, inv_sq, f_mapped):
    x_var = x_ref.clone().detach().requires_grad_(True)
    u = model(x_var)
    grad_u = torch.autograd.grad(
        u,
        x_var,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    second_derivs = []
    for i in range(3):
        d_ui = grad_u[:, i : i + 1]
        d2_ui = torch.autograd.grad(
            d_ui,
            x_var,
            grad_outputs=torch.ones_like(d_ui),
            create_graph=True,
        )[0][:, i : i + 1]
        second_derivs.append(d2_ui)

    operator = -(
        inv_sq[0] * second_derivs[0]
        + inv_sq[1] * second_derivs[1]
        + inv_sq[2] * second_derivs[2]
    )
    residual = operator - f_mapped
    return residual


def train_pinn(
    model,
    a_axis,
    b_axis,
    c_axis,
    n_epochs=8000,
    lr=1e-3,
    n_int=512,
    n_bc=256,
    bc_weight=5.0,
    target_total_loss=None,
    log_every=400,
):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    x_int = sample_interior_unit_ball(n_int, device=device, dtype=dtype)
    x_bc = sample_boundary_unit_sphere(n_bc, device=device, dtype=dtype)
    u_bc_target = torch.zeros((n_bc, 1), device=device, dtype=dtype)

    inv_sq = torch.tensor(
        [1.0 / (a_axis**2), 1.0 / (b_axis**2), 1.0 / (c_axis**2)],
        device=device,
        dtype=dtype,
    )
    f_mapped = source_term_mapped_constant(a_axis, b_axis, c_axis, device, dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "total_loss": [],
        "pde_loss": [],
        "bc_loss": [],
    }

    start_time = time.time()
    stop_reason = "max_epochs"

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()

        residual = mapped_poisson_residual(model, x_int, inv_sq, f_mapped)
        loss_pde = torch.mean(residual**2)

        u_bc_pred = model(x_bc)
        loss_bc = torch.mean((u_bc_pred - u_bc_target) ** 2)

        loss_total = loss_pde + bc_weight * loss_bc
        loss_total.backward()
        optimizer.step()

        history["total_loss"].append(float(loss_total.item()))
        history["pde_loss"].append(float(loss_pde.item()))
        history["bc_loss"].append(float(loss_bc.item()))

        if epoch % max(1, log_every) == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch}/{n_epochs} | "
                f"Total: {loss_total.item():.6e} | "
                f"PDE: {loss_pde.item():.6e} | "
                f"BC: {loss_bc.item():.6e} | "
                f"Time: {elapsed:.1f}s"
            )

        if target_total_loss is not None and loss_total.item() <= target_total_loss:
            stop_reason = f"target_total_loss reached ({loss_total.item():.3e})"
            print(
                f"Stopping at epoch {epoch}/{n_epochs}: "
                f"total_loss={loss_total.item():.3e} <= target={target_total_loss:.3e}"
            )
            break

    train_time = time.time() - start_time
    history["epochs_ran"] = len(history["total_loss"])
    history["train_time_sec"] = train_time
    history["stop_reason"] = stop_reason
    history["final_total_loss"] = history["total_loss"][-1]
    return history


def evaluate_model(model, n_eval=20000):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    x_eval = sample_interior_unit_ball(n_eval, device=device, dtype=dtype)
    with torch.no_grad():
        u_pred = model(x_eval)
        u_exact = exact_solution_mapped(x_eval)

    abs_err = torch.abs(u_pred - u_exact)
    l2 = torch.sqrt(torch.mean((u_pred - u_exact) ** 2)).item()
    rel_l2 = torch.sqrt(
        torch.mean((u_pred - u_exact) ** 2) / torch.clamp(torch.mean(u_exact**2), min=1e-12)
    ).item()
    max_err = torch.max(abs_err).item()
    return {
        "l2_error": float(l2),
        "relative_l2_error": float(rel_l2),
        "max_error": float(max_err),
    }


def mapped_slice_prediction(model, n_grid=180, zeta=0.0):
    grid = np.linspace(-1.0, 1.0, n_grid)
    XI, ETA = np.meshgrid(grid, grid)
    mask = (XI**2 + ETA**2 + zeta**2) <= 1.0

    pred_grid = np.full_like(XI, np.nan, dtype=float)
    exact_grid = np.full_like(XI, np.nan, dtype=float)
    err_grid = np.full_like(XI, np.nan, dtype=float)

    if np.any(mask):
        pts = np.column_stack([XI[mask], ETA[mask], np.full(np.sum(mask), zeta)])
        pts_t = torch.tensor(pts, dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)
        with torch.no_grad():
            pred = model(pts_t).cpu().numpy().reshape(-1)
        exact = 1.0 - np.sum(pts**2, axis=1)
        err = np.abs(pred - exact)

        pred_grid[mask] = pred
        exact_grid[mask] = exact
        err_grid[mask] = err

    return XI, ETA, pred_grid, exact_grid, err_grid


def physical_slice_prediction(model, a_axis, b_axis, c_axis, n_grid=200, z_phys=0.0):
    x_vals = np.linspace(-a_axis, a_axis, n_grid)
    y_vals = np.linspace(-b_axis, b_axis, n_grid)
    X, Y = np.meshgrid(x_vals, y_vals)
    mask = ((X / a_axis) ** 2 + (Y / b_axis) ** 2 + (z_phys / c_axis) ** 2) <= 1.0

    pred_grid = np.full_like(X, np.nan, dtype=float)
    if np.any(mask):
        xi = X[mask] / a_axis
        eta = Y[mask] / b_axis
        zeta = np.full_like(xi, z_phys / c_axis)
        pts_ref = np.column_stack([xi, eta, zeta])
        pts_t = torch.tensor(
            pts_ref,
            dtype=next(model.parameters()).dtype,
            device=next(model.parameters()).device,
        )
        with torch.no_grad():
            pred = model(pts_t).cpu().numpy().reshape(-1)
        pred_grid[mask] = pred
    return X, Y, pred_grid


def create_visualization(history, metrics, a_axis, b_axis, c_axis, bc_weight, output_path):
    XI, ETA, pred_mapped, exact_mapped, err_mapped = mapped_slice_prediction(model=metrics["model"])
    X_phys, Y_phys, pred_phys = physical_slice_prediction(
        model=metrics["model"],
        a_axis=a_axis,
        b_axis=b_axis,
        c_axis=c_axis,
    )

    fig = plt.figure(figsize=(18, 11))

    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.contourf(XI, ETA, pred_mapped, levels=18, cmap="viridis")
    ax1.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="red", linewidth=2))
    ax1.set_title("Mapped Domain: Predicted u (zeta=0)")
    ax1.set_aspect("equal")
    plt.colorbar(im1, ax=ax1, label="u")

    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.contourf(XI, ETA, exact_mapped, levels=18, cmap="viridis")
    ax2.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="red", linewidth=2))
    ax2.set_title("Mapped Domain: Exact u (zeta=0)")
    ax2.set_aspect("equal")
    plt.colorbar(im2, ax=ax2, label="u")

    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.contourf(XI, ETA, err_mapped, levels=18, cmap="magma")
    ax3.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="white", linewidth=2))
    ax3.set_title("Mapped Domain: |Error| (zeta=0)")
    ax3.set_aspect("equal")
    plt.colorbar(im3, ax=ax3, label="|u - u_exact|")

    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.contourf(X_phys, Y_phys, pred_phys, levels=18, cmap="viridis")
    ax4.add_patch(
        Ellipse(
            (0.0, 0.0),
            width=2.0 * a_axis,
            height=2.0 * b_axis,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
    )
    ax4.set_title("Physical Slice z=0 in Ellipsoid")
    ax4.set_aspect("equal")
    plt.colorbar(im4, ax=ax4, label="u")

    epochs = range(1, len(history["total_loss"]) + 1)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.semilogy(epochs, np.maximum(history["total_loss"], 1e-16), label="Total")
    ax5.semilogy(epochs, np.maximum(history["pde_loss"], 1e-16), label="PDE")
    ax5.semilogy(epochs, np.maximum(np.array(history["bc_loss"]) * bc_weight, 1e-16), label=f"BC x {bc_weight}")
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Loss")
    ax5.set_title("Training Curves")
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    summary = (
        "3D Ellipsoid -> Sphere Mapping\n\n"
        f"x = ({a_axis}*xi, {b_axis}*eta, {c_axis}*zeta)\n"
        "xi^2 + eta^2 + zeta^2 <= 1\n\n"
        "Mapped PDE:\n"
        "-[(1/a^2)u_xixi + (1/b^2)u_etaeta + (1/c^2)u_zetazeta] = f_map\n"
        "u=0 on sphere boundary\n\n"
        f"f_map = {2.0 * (1.0 / a_axis**2 + 1.0 / b_axis**2 + 1.0 / c_axis**2):.6f}\n"
        f"Epochs: {history['epochs_ran']}\n"
        f"Stop: {history['stop_reason']}\n"
        f"Final total loss: {history['final_total_loss']:.3e}\n"
        f"L2 error: {metrics['l2_error']:.3e}\n"
        f"Relative L2: {metrics['relative_l2_error']:.3e}\n"
        f"Max error: {metrics['max_error']:.3e}\n"
        f"Train time: {history['train_time_sec']:.1f}s"
    )
    ax6.text(0.02, 0.98, summary, va="top", family="monospace")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_experiment(args):
    set_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    # Set the default dtype BEFORE constructing PINN3D so that nn.Linear's
    # internal init draws from the RNG at the right precision. Without this,
    # the model is initialised at float32 and cast to dtype later, which
    # advances the RNG differently and breaks bit-exact reproducibility
    # against the pre-refactor (CPU+float64) baseline.
    torch.set_default_dtype(dtype)
    print(f"Device={device.type} dtype={dtype}")
    model = PINN3D(width=args.width, depth=args.depth).to(device=device, dtype=dtype)

    print("=" * 88)
    print("3D PINN: Poisson on Ellipsoid mapped to Unit Sphere")
    print("=" * 88)
    print(f"Ellipsoid semiaxes (a, b, c): ({args.a}, {args.b}, {args.c})")
    print("Mapped PDE on sphere:")
    print("  -[(1/a^2)u_xixi + (1/b^2)u_etaeta + (1/c^2)u_zetazeta] = f_map")
    print("  u = 0 on xi^2 + eta^2 + zeta^2 = 1")
    print(f"f_map = {2.0 * (1.0 / args.a**2 + 1.0 / args.b**2 + 1.0 / args.c**2):.6f}")
    print(f"Epoch cap: {args.epochs}, LR: {args.lr}, n_int: {args.n_int}, n_bc: {args.n_bc}")
    print(f"Target total loss: {args.target_loss}")

    history = train_pinn(
        model=model,
        a_axis=args.a,
        b_axis=args.b,
        c_axis=args.c,
        n_epochs=args.epochs,
        lr=args.lr,
        n_int=args.n_int,
        n_bc=args.n_bc,
        bc_weight=args.bc_weight,
        target_total_loss=args.target_loss,
        log_every=max(1, args.epochs // 10),
    )

    metrics = evaluate_model(model, n_eval=args.eval_points)
    print("\nFinal metrics")
    print("-" * 88)
    print(f"L2 error:        {metrics['l2_error']:.6e}")
    print(f"Relative L2:     {metrics['relative_l2_error']:.6e}")
    print(f"Max error:       {metrics['max_error']:.6e}")
    print(f"Final total loss:{history['final_total_loss']:.6e}")

    if not args.no_plot:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "pinn_3d_ellipsoid_mapped_sphere.png")
        metrics_with_model = dict(metrics)
        metrics_with_model["model"] = model
        create_visualization(
            history=history,
            metrics=metrics_with_model,
            a_axis=args.a,
            b_axis=args.b,
            c_axis=args.c,
            bc_weight=args.bc_weight,
            output_path=output_path,
        )
        print(f"Saved visualization: {output_path}")

    return model, history, metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="3D PINN for ellipsoid Poisson mapped to unit sphere."
    )
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-int", type=int, default=512)
    parser.add_argument("--n-bc", type=int, default=256)
    parser.add_argument("--bc-weight", type=float, default=5.0)
    parser.add_argument("--a", type=float, default=2.0)
    parser.add_argument("--b", type=float, default=1.4)
    parser.add_argument("--c", type=float, default=0.9)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-loss", type=float, default=1e-4)
    parser.add_argument("--eval-points", type=int, default=20000)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device. 'auto' prefers CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float64"],
        help="Tensor dtype. 'auto' chooses float32 on GPU backends and float64 on CPU.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()

