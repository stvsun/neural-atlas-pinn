"""Differentiable return mapping for finite-strain von Mises elastoplasticity.

Implements a regularised (softplus) radial-return algorithm in logarithmic
strain space following de Souza Neto et al., Ch. 14.  The entire map is
smooth and compatible with PyTorch autograd, so material parameters such as
the yield stress can be learned end-to-end via gradient descent.

Supports arbitrary leading batch dimensions and CPU / CUDA / MPS backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EYE3: torch.Tensor | None = None  # lazily cached identity


def _eye3(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a cached (3, 3) identity on the requested device/dtype."""
    global _EYE3
    if _EYE3 is None or _EYE3.device != device or _EYE3.dtype != dtype:
        _EYE3 = torch.eye(3, device=device, dtype=dtype)
    return _EYE3


def dev(T: torch.Tensor) -> torch.Tensor:
    """Deviatoric part of a batched (..., 3, 3) tensor.

    dev(T) = T - (1/3) tr(T) I
    """
    I = _eye3(T.device, T.dtype)
    tr = torch.diagonal(T, dim1=-2, dim2=-1).sum(dim=-1)  # (...)
    return T - (tr / 3.0)[..., None, None] * I


def frobenius_norm(T: torch.Tensor) -> torch.Tensor:
    """Frobenius norm ||T||_F for batched (..., 3, 3) tensors.

    Returns a tensor of shape (...).
    """
    return torch.sqrt((T * T).sum(dim=(-2, -1)).clamp(min=1e-30))


# ---------------------------------------------------------------------------
# Symmetric matrix logarithm / exponential via eigendecomposition
# ---------------------------------------------------------------------------

def _divided_diff(f_lam: torch.Tensor, fp_lam: torch.Tensor,
                   lam: torch.Tensor) -> torch.Tensor:
    """Compute the divided-difference matrix L for eigenvalue-based matrix functions.

    L_ij = (f(λ_i) - f(λ_j)) / (λ_i - λ_j)   when |λ_i - λ_j| > tol,
    L_ij = f'(λ_i)                              otherwise (L'Hôpital).

    This avoids the NaN gradient that ``torch.linalg.eigh`` produces when
    eigenvalues are degenerate.

    Parameters
    ----------
    f_lam : (..., 3)   f evaluated at eigenvalues
    fp_lam : (..., 3)  f' evaluated at eigenvalues
    lam : (..., 3)     eigenvalues
    """
    # pairwise differences  (..., 3, 3)
    lam_i = lam.unsqueeze(-1)                     # (..., 3, 1)
    lam_j = lam.unsqueeze(-2)                     # (..., 1, 3)
    diff = lam_i - lam_j                          # (..., 3, 3)

    f_i = f_lam.unsqueeze(-1)                     # (..., 3, 1)
    f_j = f_lam.unsqueeze(-2)                     # (..., 1, 3)

    tol = 1e-10 * (lam.abs().max(dim=-1, keepdim=True).values.unsqueeze(-1) + 1e-30)
    safe = diff.abs() > tol

    # Standard divided difference where eigenvalues are distinct
    L = torch.where(safe, (f_i - f_j) / (diff + (~safe) * 1.0), torch.zeros_like(diff))

    # L'Hôpital fallback: use f'(λ_i) on the diagonal and degenerate off-diags
    fp_diag = fp_lam.unsqueeze(-1).expand_as(L)   # (..., 3, 3) — broadcast f'(λ_i)
    L = torch.where(safe, L, fp_diag)

    return L


class _SymLogm(torch.autograd.Function):
    """Matrix logarithm with gradient stable under degenerate eigenvalues."""

    @staticmethod
    def forward(ctx, A):
        lam, Q = torch.linalg.eigh(A)
        lam = lam.clamp(min=1e-12)
        log_lam = torch.log(lam)
        result = (Q * log_lam.unsqueeze(-2)) @ Q.transpose(-2, -1)
        ctx.save_for_backward(lam, Q)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        lam, Q = ctx.saved_tensors
        # f = log, f' = 1/lam
        f_lam = torch.log(lam)
        fp_lam = 1.0 / lam
        L = _divided_diff(f_lam, fp_lam, lam)
        # grad_A = Q @ (L ⊙ (Qᵀ @ grad_output @ Q)) @ Qᵀ
        inner = Q.transpose(-2, -1) @ grad_output @ Q
        grad_A = Q @ (L * inner) @ Q.transpose(-2, -1)
        # Symmetrise to stay in symmetric-matrix space
        grad_A = 0.5 * (grad_A + grad_A.transpose(-2, -1))
        return grad_A


class _SymExpm(torch.autograd.Function):
    """Matrix exponential with gradient stable under degenerate eigenvalues."""

    @staticmethod
    def forward(ctx, A):
        lam, Q = torch.linalg.eigh(A)
        exp_lam = torch.exp(lam)
        result = (Q * exp_lam.unsqueeze(-2)) @ Q.transpose(-2, -1)
        ctx.save_for_backward(lam, Q)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        lam, Q = ctx.saved_tensors
        # f = exp, f' = exp
        f_lam = torch.exp(lam)
        fp_lam = f_lam  # exp' = exp
        L = _divided_diff(f_lam, fp_lam, lam)
        inner = Q.transpose(-2, -1) @ grad_output @ Q
        grad_A = Q @ (L * inner) @ Q.transpose(-2, -1)
        grad_A = 0.5 * (grad_A + grad_A.transpose(-2, -1))
        return grad_A


def sym_logm(A: torch.Tensor) -> torch.Tensor:
    """Matrix logarithm of a batched SPD matrix via eigen-decomposition.

    Uses a custom backward pass with L'Hôpital-stable divided differences
    to avoid NaN gradients when eigenvalues are degenerate.

    Parameters
    ----------
    A : torch.Tensor, shape (..., 3, 3)
        Symmetric positive-definite matrices.

    Returns
    -------
    torch.Tensor, shape (..., 3, 3)
        Q diag(log lam_i) Q^T
    """
    return _SymLogm.apply(A)


def sym_expm(A: torch.Tensor) -> torch.Tensor:
    """Matrix exponential of a batched symmetric matrix via eigen-decomposition.

    Uses a custom backward pass with L'Hôpital-stable divided differences
    to avoid NaN gradients when eigenvalues are degenerate.

    Parameters
    ----------
    A : torch.Tensor, shape (..., 3, 3)
        Symmetric matrices.

    Returns
    -------
    torch.Tensor, shape (..., 3, 3)
        Q diag(exp lam_i) Q^T
    """
    return _SymExpm.apply(A)


# ---------------------------------------------------------------------------
# Plastic state
# ---------------------------------------------------------------------------

@dataclass
class ReturnMappingState:
    """Internal variables for multiplicative elastoplasticity.

    Attributes
    ----------
    Be : torch.Tensor, shape (..., 3, 3)
        Elastic left Cauchy--Green tensor.  Initialised to I.
    ep_bar : torch.Tensor, shape (...)
        Accumulated equivalent plastic strain.  Initialised to 0.
    beta : torch.Tensor, shape (..., 3, 3)
        Deviatoric back-stress tensor (kinematic hardening).  Initialised to 0.
    """
    Be: torch.Tensor       # (..., 3, 3)
    ep_bar: torch.Tensor   # (...)
    beta: torch.Tensor     # (..., 3, 3)

    # ----- utilities -----

    def clone(self) -> ReturnMappingState:
        """Return a deep copy (new leaf tensors)."""
        return ReturnMappingState(
            Be=self.Be.clone(),
            ep_bar=self.ep_bar.clone(),
            beta=self.beta.clone(),
        )

    @staticmethod
    def zeros(
        shape: Tuple[int, ...],
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> ReturnMappingState:
        """Factory: create a zero-initialised state (Be = I, rest = 0).

        Parameters
        ----------
        shape : tuple of int
            Leading batch dimensions, e.g. ``(N,)`` or ``(B, N)``.
        device, dtype : torch device / dtype forwarded to tensor constructors.
        """
        I = torch.eye(3, device=device, dtype=dtype)
        Be = I.expand(*shape, 3, 3).clone()
        ep_bar = torch.zeros(shape, device=device, dtype=dtype)
        beta = torch.zeros(*shape, 3, 3, device=device, dtype=dtype)
        return ReturnMappingState(Be=Be, ep_bar=ep_bar, beta=beta)


# ---------------------------------------------------------------------------
# Smooth return mapping
# ---------------------------------------------------------------------------

def smooth_return_map(
    F_delta: torch.Tensor,
    state: ReturnMappingState,
    mu: torch.Tensor,
    K: torch.Tensor,
    tau_y: torch.Tensor,
    H_kin: torch.Tensor,
    epsilon: float = 1e-2,
) -> Tuple[torch.Tensor, ReturnMappingState]:
    """Regularised radial-return in logarithmic strain space.

    Parameters
    ----------
    F_delta : torch.Tensor, shape (..., 3, 3)
        Incremental deformation gradient.
    state : ReturnMappingState
        Current plastic internal variables.
    mu : torch.Tensor
        Shear modulus (scalar or batched).
    K : torch.Tensor
        Bulk modulus (scalar or batched).
    tau_y : torch.Tensor
        Yield stress (scalar, may require grad).
    H_kin : torch.Tensor
        Kinematic hardening modulus (scalar or batched).
    epsilon : float
        Softplus regularisation sharpness (smaller = sharper).

    Returns
    -------
    tau : torch.Tensor, shape (..., 3, 3)
        Kirchhoff stress.
    new_state : ReturnMappingState
        Updated internal variables.
    """
    I = _eye3(F_delta.device, F_delta.dtype)

    Be_n = state.Be
    ep_bar_n = state.ep_bar
    beta_n = state.beta

    # --- elastic predictor ---
    Be_trial = F_delta @ Be_n @ F_delta.transpose(-2, -1)

    eps_e_trial = 0.5 * sym_logm(Be_trial)                   # (...,3,3)

    # trace and deviatoric elastic strain
    tr_eps = torch.diagonal(eps_e_trial, dim1=-2, dim2=-1).sum(-1)  # (...)
    dev_eps_trial = eps_e_trial - (tr_eps / 3.0)[..., None, None] * I

    # trial Kirchhoff stress  (only deviatoric part needed for yield check)
    tau_dev_trial = 2.0 * mu * dev_eps_trial                  # (...,3,3)

    # --- shifted stress ---
    s = tau_dev_trial - beta_n                                # (...,3,3)
    s_norm = frobenius_norm(s)                                # (...)

    q = torch.sqrt(torch.tensor(1.5, device=s.device, dtype=s.dtype)) * s_norm
    Phi = q - tau_y                                           # yield function

    # --- regularised plastic corrector (no branching) ---
    eta = F.softplus(Phi / epsilon) * epsilon                 # smooth max(Phi,0)
    Delta_gamma = eta / (3.0 * mu + H_kin)                   # (...,)

    # flow direction — guard against division by zero when s_norm ≈ 0
    # (elastic regime: Δγ ≈ 0, so Δγ·N ≈ 0 regardless of N's direction)
    _safe_norm = torch.where(
        s_norm > 1e-10,
        s_norm,
        torch.ones_like(s_norm),       # dummy denominator (replaced by 0 below)
    )
    N = torch.where(
        (s_norm > 1e-10)[..., None, None],
        s / _safe_norm[..., None, None],
        torch.zeros_like(s),
    )                                                        # (...,3,3)

    sqrt_1p5 = torch.sqrt(torch.tensor(1.5, device=s.device, dtype=s.dtype))
    sqrt_2o3 = torch.sqrt(torch.tensor(2.0 / 3.0, device=s.device, dtype=s.dtype))

    # --- corrected logarithmic strain ---
    eps_e = eps_e_trial - (Delta_gamma * sqrt_1p5)[..., None, None] * N

    # --- updated elastic left Cauchy-Green ---
    Be_new = sym_expm(2.0 * eps_e)

    # --- updated plastic strain and back stress ---
    ep_bar_new = ep_bar_n + sqrt_2o3 * Delta_gamma
    beta_new = beta_n + (2.0 / 3.0) * H_kin * Delta_gamma[..., None, None] * N

    # --- updated Kirchhoff stress ---
    tr_eps_e = torch.diagonal(eps_e, dim1=-2, dim2=-1).sum(-1)
    dev_eps_e = eps_e - (tr_eps_e / 3.0)[..., None, None] * I
    tau = 2.0 * mu * dev_eps_e + (K * tr_eps_e)[..., None, None] * I

    new_state = ReturnMappingState(Be=Be_new, ep_bar=ep_bar_new, beta=beta_new)
    return tau, new_state


# ===================================================================
# Unit tests
# ===================================================================

if __name__ == "__main__":
    import math

    torch.set_default_dtype(torch.float64)
    device = "cpu"

    # material parameters
    E = 200.0       # Young's modulus (GPa-ish units)
    nu = 0.3
    mu_val = E / (2.0 * (1.0 + nu))
    K_val = E / (3.0 * (1.0 - 2.0 * nu))
    tau_y_val = 0.4  # yield stress

    mu = torch.tensor(mu_val, device=device)
    K = torch.tensor(K_val, device=device)
    tau_y = torch.tensor(tau_y_val, device=device)

    def _uniaxial_F(eps: float) -> torch.Tensor:
        """Incremental deformation gradient for uniaxial tension (incompressible lateral)."""
        lam = 1.0 + eps
        lat = 1.0 / math.sqrt(lam)
        return torch.diag(torch.tensor([lam, lat, lat], device=device))

    # ---------------------------------------------------------------
    # Test 1: elastic regime (perfect plasticity, H_kin = 0)
    # ---------------------------------------------------------------
    print("=" * 60)
    print("TEST 1: Elastic regime (H_kin=0, below yield)")
    print("=" * 60)

    H_kin_zero = torch.tensor(0.0, device=device)
    state = ReturnMappingState.zeros((), device=device)

    eps_small = 0.0005  # well below yield
    F_inc = _uniaxial_F(eps_small).unsqueeze(0)  # (1,3,3) — batch dim
    state_b = ReturnMappingState.zeros((1,), device=device)

    tau, state_b = smooth_return_map(F_inc, state_b, mu, K, tau_y, H_kin_zero, epsilon=1e-4)
    tau_11 = tau[0, 0, 0].item()
    # For isochoric uniaxial F = diag(1+eps, 1/sqrt(1+eps), 1/sqrt(1+eps)):
    # eps_e = 1/2 log(Be) = diag(log(1+eps), -1/2 log(1+eps), -1/2 log(1+eps))
    # tr(eps_e) = 0, so tau = 2*mu * eps_e, giving tau_11 = 2*mu*log(1+eps) ~ 2*mu*eps
    expected = 2.0 * mu_val * math.log(1.0 + eps_small)
    err = abs(tau_11 - expected) / abs(expected)
    passed = err < 0.01
    print(f"  tau_11 = {tau_11:.6f},  expected ~ {expected:.6f},  rel err = {err:.2e}  "
          f"{'[PASS]' if passed else '[FAIL]'}")

    # ---------------------------------------------------------------
    # Test 2: stress saturation (perfect plasticity, H_kin=0)
    # ---------------------------------------------------------------
    print()
    print("=" * 60)
    print("TEST 2: Stress saturation with perfect plasticity (H_kin=0)")
    print("=" * 60)

    state = ReturnMappingState.zeros((), device=device)
    n_steps = 80
    dep = 0.002
    stresses = []
    for i in range(n_steps):
        F_inc = _uniaxial_F(dep)
        tau, state = smooth_return_map(F_inc, state, mu, K, tau_y, H_kin_zero, epsilon=1e-3)
        sig_11 = tau[0, 0].item()
        stresses.append(sig_11)

    # In perfect plasticity the axial deviatoric Kirchhoff stress should
    # saturate near tau_y * sqrt(2/3)^{-1} = tau_y * sqrt(3/2).
    # Actually for von Mises: q = sqrt(3/2)*||s||, and yield is q = tau_y,
    # so ||s|| = tau_y / sqrt(3/2).  The axial deviatoric stress s_11 for
    # uniaxial loading is (2/3)*sigma_dev_axial, so sigma_11_dev saturates
    # at tau_y * sqrt(2/3).  Total sigma_11 includes volumetric part which
    # keeps growing with K*eps_vol.  For the *deviatoric* component check:
    final_tau = stresses[-1]
    final_tr = tau.diagonal(dim1=-2, dim2=-1).sum().item()
    final_dev_11 = final_tau - final_tr / 3.0
    # The deviatoric component should not grow unboundedly — it saturates.
    mid_dev_11_list = []
    state2 = ReturnMappingState.zeros((), device=device)
    for i in range(n_steps):
        F_inc = _uniaxial_F(dep)
        tau2, state2 = smooth_return_map(F_inc, state2, mu, K, tau_y, H_kin_zero, epsilon=1e-3)
        tr2 = tau2.diagonal(dim1=-2, dim2=-1).sum().item()
        mid_dev_11_list.append(tau2[0, 0].item() - tr2 / 3.0)

    # Check that deviatoric stress plateaus: last 20 values should be nearly constant
    late_devs = mid_dev_11_list[-20:]
    dev_range = max(late_devs) - min(late_devs)
    passed = dev_range < 0.05 * abs(late_devs[-1])
    print(f"  Deviatoric sigma_11 at end: {late_devs[-1]:.4f}")
    print(f"  Range over last 20 steps:   {dev_range:.6f}")
    print(f"  Saturation check:           {'[PASS]' if passed else '[FAIL]'}")

    # ---------------------------------------------------------------
    # Test 3: kinematic hardening gives reduced post-yield slope
    # ---------------------------------------------------------------
    print()
    print("=" * 60)
    print("TEST 3: Post-yield slope with kinematic hardening (H_kin > 0)")
    print("=" * 60)

    H_kin_pos = torch.tensor(20.0, device=device)
    state_h = ReturnMappingState.zeros((), device=device)
    dev_stresses_h = []
    for i in range(n_steps):
        F_inc = _uniaxial_F(dep)
        tau_h, state_h = smooth_return_map(F_inc, state_h, mu, K, tau_y, H_kin_pos, epsilon=1e-3)
        tr_h = tau_h.diagonal(dim1=-2, dim2=-1).sum().item()
        dev_stresses_h.append(tau_h[0, 0].item() - tr_h / 3.0)

    # With H_kin > 0 the deviatoric stress should keep increasing (not saturate)
    slope_late = dev_stresses_h[-1] - dev_stresses_h[-21]
    slope_early_elastic = dev_stresses_h[1] - dev_stresses_h[0]
    passed = slope_late > 0.01 * slope_early_elastic  # still increasing, but slower
    print(f"  Elastic-regime dev slope:   {slope_early_elastic:.6f}")
    print(f"  Late plastic-regime slope (20 steps): {slope_late:.6f}")
    print(f"  Hardening gives continued increase: {'[PASS]' if passed else '[FAIL]'}")

    # ---------------------------------------------------------------
    # Test 4: Bauschinger effect under cyclic loading (H_kin > 0)
    # ---------------------------------------------------------------
    print()
    print("=" * 60)
    print("TEST 4: Bauschinger effect (cyclic loading, H_kin > 0)")
    print("=" * 60)

    H_kin_cyc = torch.tensor(30.0, device=device)
    state_c = ReturnMappingState.zeros((), device=device)

    # Phase 1: load in tension
    n_load = 40
    dep_cyc = 0.003
    for _ in range(n_load):
        F_inc = _uniaxial_F(dep_cyc)
        tau_c, state_c = smooth_return_map(F_inc, state_c, mu, K, tau_y, H_kin_cyc, epsilon=1e-3)

    tr_c = tau_c.diagonal(dim1=-2, dim2=-1).sum().item()
    peak_tension_dev = tau_c[0, 0].item() - tr_c / 3.0
    print(f"  Peak tension dev sigma_11: {peak_tension_dev:.4f}")

    # Phase 2: unload and reverse-load (compression)
    # Use inverse incremental F to go back
    dep_rev = -dep_cyc
    yield_comp_dev = None
    for i in range(2 * n_load):
        F_inc = _uniaxial_F(dep_rev)
        tau_c, state_c = smooth_return_map(F_inc, state_c, mu, K, tau_y, H_kin_cyc, epsilon=1e-3)
        tr_c = tau_c.diagonal(dim1=-2, dim2=-1).sum().item()
        current_dev = tau_c[0, 0].item() - tr_c / 3.0
        # Detect approximate re-yield in compression (deviatoric goes negative and
        # plastic strain starts accumulating again)
        if yield_comp_dev is None and current_dev < 0 and state_c.ep_bar.item() > 0.01:
            yield_comp_dev = current_dev

    if yield_comp_dev is not None:
        # Bauschinger: |yield in compression| < |peak in tension| (back-stress shifts)
        bauschinger = abs(yield_comp_dev) < abs(peak_tension_dev)
        print(f"  Approx yield in compression dev: {yield_comp_dev:.4f}")
        print(f"  |comp yield| < |tension peak|:   {'[PASS]' if bauschinger else '[FAIL]'}")
    else:
        print("  Could not detect compression yield  [FAIL]")

    # ---------------------------------------------------------------
    # Test 5: autograd — tau_y gradient flows
    # ---------------------------------------------------------------
    print()
    print("=" * 60)
    print("TEST 5: Autograd — gradient w.r.t. tau_y")
    print("=" * 60)

    tau_y_ad = torch.tensor(tau_y_val, device=device, requires_grad=True)
    state_ad = ReturnMappingState.zeros((), device=device)
    for _ in range(10):
        F_inc = _uniaxial_F(0.005)
        tau_ad, state_ad = smooth_return_map(
            F_inc, state_ad, mu, K, tau_y_ad,
            torch.tensor(10.0, device=device), epsilon=1e-3,
        )
    loss = tau_ad[0, 0]
    loss.backward()
    grad_val = tau_y_ad.grad
    passed = grad_val is not None and torch.isfinite(grad_val).all()
    print(f"  d(tau_11)/d(tau_y) = {grad_val}")
    print(f"  Gradient exists and finite: {'[PASS]' if passed else '[FAIL]'}")

    # ---------------------------------------------------------------
    # Test 6: batched operation
    # ---------------------------------------------------------------
    print()
    print("=" * 60)
    print("TEST 6: Batched operation (B=8)")
    print("=" * 60)

    B = 8
    state_batch = ReturnMappingState.zeros((B,), device=device)
    # Different strain increments per batch element
    eps_vals = torch.linspace(0.001, 0.01, B, device=device)
    F_batch = torch.zeros(B, 3, 3, device=device)
    for b in range(B):
        F_batch[b] = _uniaxial_F(eps_vals[b].item())

    tau_batch, state_batch = smooth_return_map(
        F_batch, state_batch, mu, K, tau_y, H_kin_zero, epsilon=1e-3,
    )
    passed = tau_batch.shape == (B, 3, 3) and state_batch.Be.shape == (B, 3, 3)
    print(f"  Output tau shape: {tau_batch.shape}")
    print(f"  Output Be shape:  {state_batch.Be.shape}")
    print(f"  Shapes correct: {'[PASS]' if passed else '[FAIL]'}")

    print()
    print("=" * 60)
    print("All tests complete.")
    print("=" * 60)
