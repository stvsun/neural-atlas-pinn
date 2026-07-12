#!/usr/bin/env python3
"""Incremental load-stepping forward solver for finite-strain elastoplasticity.

Couples the differentiable return mapping (return_mapping.py) with the P1
vector FEM solver (chart_vector_fem.py) to simulate cyclic loading on a
single chart.  The entire pipeline is differentiable via PyTorch autograd,
enabling inverse parameter identification by backpropagating through the
full load history.

Supports epsilon annealing via cosine schedule (PNAS Bark & Sun 2025, Eq. 12).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

from .return_mapping import (
    ReturnMappingState,
    smooth_return_map,
)
from .chart_vector_fem import ChartVectorFEMSolver


# ---------------------------------------------------------------------------
# Material wrapper: converts (F_total, state) -> (P, new_state) via return map
# ---------------------------------------------------------------------------

def _incremental_F(F_total_new: torch.Tensor, F_total_old: torch.Tensor) -> torch.Tensor:
    """Compute incremental deformation gradient  F_delta = F_new @ F_old^{-1}."""
    return F_total_new @ torch.linalg.inv(F_total_old)


# ---------------------------------------------------------------------------
# Cosine annealing for softplus sharpness (PNAS Bark & Sun 2025, Eq. 12)
# ---------------------------------------------------------------------------

def cosine_anneal(epoch: int, total_epochs: int,
                  val_start: float, val_end: float) -> float:
    """Cosine annealing schedule: val_start -> val_end over total_epochs.

    m(t) = m_0 + 0.5*(m_max - m_0)*(1 - cos(pi*t/t_max))

    For epsilon annealing: start warm (large epsilon), end sharp (small epsilon).
    """
    if total_epochs <= 1:
        return val_end
    t = min(epoch, total_epochs - 1)
    return val_start + 0.5 * (val_end - val_start) * (1.0 - math.cos(math.pi * t / (total_epochs - 1)))


# ---------------------------------------------------------------------------
# Elastoplastic equilibrium solver for a single load step
# ---------------------------------------------------------------------------

class ElastoplasticStepSolver:
    """Newton-Raphson solver for one load step of finite-strain elasto-plasticity.

    At each Newton iteration:
    1. Compute F = I + grad(u) from current nodal displacements
    2. Compute F_delta = F @ F_old^{-1}
    3. Run smooth_return_map -> Kirchhoff stress tau
    4. Convert tau -> first Piola-Kirchhoff P = tau @ F^{-T}
    5. Assemble internal forces
    6. Compute analytical tangent dP/dF via autograd (Section 14.5 of de Souza Neto)
    7. Solve increment and update u
    """

    def __init__(
        self,
        fem_solver: ChartVectorFEMSolver,
        mu: torch.Tensor,
        K: torch.Tensor,
        tau_y: torch.Tensor,
        H_kin: torch.Tensor,
        epsilon: float = 1e-2,
    ):
        self.fem = fem_solver
        self.mu = mu
        self.K = K
        self.tau_y = tau_y
        self.H_kin = H_kin
        self.epsilon = epsilon

    def _compute_residual(
        self,
        u: torch.Tensor,
        F_old: torch.Tensor,
        state: ReturnMappingState,
        f_ext: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, ReturnMappingState]:
        """Compute residual R = f_int - f_ext and return (R_flat, P, new_state)."""
        F_total = self.fem.compute_F(u)
        F_delta = _incremental_F(F_total, F_old)

        tau, new_state = smooth_return_map(
            F_delta, state, self.mu, self.K, self.tau_y, self.H_kin, self.epsilon,
        )

        # tau -> P: P = tau @ F^{-T}
        F_inv_T = torch.linalg.inv(F_total).transpose(-2, -1)
        P = tau @ F_inv_T

        # Assemble internal forces using pre-computed P
        # f_int[a,i] = sum_e vol_e * P[e,i,j] * dNdx[e,a,j]
        f_elem = self.fem.vol[:, None, None] * torch.einsum(
            "eij, eaj -> eai", P, self.fem.dNdx
        )
        f_int = torch.zeros(self.fem.n_nodes, 3, device=u.device, dtype=u.dtype)
        f_elem_flat = f_elem.reshape(-1, 3)
        idx_flat = self.fem.elements.reshape(-1).unsqueeze(-1).expand(-1, 3)
        f_int.scatter_add_(0, idx_flat, f_elem_flat)

        R = (f_int - f_ext).reshape(-1)
        return R, P, new_state

    def _element_tangent_dPdF(
        self,
        F_total: torch.Tensor,
        F_old: torch.Tensor,
        state: ReturnMappingState,
    ) -> torch.Tensor:
        """Compute element-level material tangent dP/dF via autograd.

        Uses the chain rule from de Souza Neto Section 14.5 (Eq. 14.98):
            dτ/dF = (dτ/dε^{e,trial}) : (dε^{e,trial}/dB^{e,trial}) : (dB^{e,trial}/dF)
        which is computed automatically through our smooth return mapping.

        This replaces the O(n_dof) FD tangent with O(9) backward passes
        vectorized over all M elements — roughly 40x faster.

        Parameters
        ----------
        F_total : (M, 3, 3) total deformation gradient at current u
        F_old : (M, 3, 3) total deformation gradient from previous step
        state : plastic state from previous step

        Returns
        -------
        C : (M, 9, 9)  dP_iJ / dF_kL in row-major ordering
        """
        # Delegate to impl with @enable_grad for no_grad contexts
        return self._element_tangent_dPdF_impl(F_total, F_old, state)

    @torch.enable_grad()
    def _element_tangent_dPdF_impl(
        self, F_total: torch.Tensor, F_old: torch.Tensor, state: ReturnMappingState,
    ) -> torch.Tensor:
        M = F_total.shape[0]
        device = F_total.device
        dtype = F_total.dtype
        F_var = F_total.detach().clone().requires_grad_(True)

        # Forward: F_delta -> return mapping -> tau -> P
        # Detach everything except F_var so autograd only tracks dP/dF
        F_delta = F_var @ torch.linalg.inv(F_old.detach())
        mu_d = self.mu.detach() if isinstance(self.mu, torch.Tensor) else self.mu
        K_d = self.K.detach() if isinstance(self.K, torch.Tensor) else self.K
        tau_y_d = self.tau_y.detach() if isinstance(self.tau_y, torch.Tensor) else self.tau_y
        H_kin_d = self.H_kin.detach() if isinstance(self.H_kin, torch.Tensor) else self.H_kin
        state_d = ReturnMappingState(
            Be=state.Be.detach(), ep_bar=state.ep_bar.detach(), beta=state.beta.detach(),
        )
        tau, _ = smooth_return_map(
            F_delta, state_d, mu_d, K_d, tau_y_d, H_kin_d, self.epsilon,
        )
        F_inv_T = torch.linalg.inv(F_var).transpose(-2, -1)
        P = tau @ F_inv_T  # (M, 3, 3)

        # Compute dP/dF via 9 backward passes (one per component of P)
        # P is (M, 3, 3), F_var is (M, 3, 3)
        C = torch.zeros(M, 9, 9, device=device, dtype=dtype)

        for r in range(9):
            i, J = r // 3, r % 3
            # Select P_{iJ} for all elements
            grad_out = torch.zeros(M, 3, 3, device=device, dtype=dtype)
            grad_out[:, i, J] = 1.0
            grads = torch.autograd.grad(
                P, F_var,
                grad_outputs=grad_out,
                retain_graph=(r < 8),
                create_graph=False,
            )[0]  # (M, 3, 3)
            C[:, r, :] = grads.reshape(M, 9)

        return C  # (M, 9, 9)

    def solve_step(
        self,
        u_prev: torch.Tensor,
        state: ReturnMappingState,
        F_old: torch.Tensor,
        u_bc: torch.Tensor,
        bc_mask: torch.Tensor,
        f_ext: Optional[torch.Tensor] = None,
        max_iter: int = 25,
        tol: float = 1e-8,
    ) -> Tuple[torch.Tensor, ReturnMappingState, torch.Tensor]:
        """Solve one load step.

        Returns (u, new_state, F_new).
        """
        device = u_prev.device
        dtype = u_prev.dtype
        N = self.fem.n_nodes

        if f_ext is None:
            f_ext = torch.zeros(N, 3, device=device, dtype=dtype)

        u = u_prev.clone()
        u[bc_mask] = u_bc[bc_mask]

        bc_mask_dof = bc_mask.unsqueeze(1).expand(-1, 3).reshape(-1)
        free_dof = ~bc_mask_dof

        converged_state = state
        K_tan = None

        for it in range(max_iter):
            R, P, trial_state = self._compute_residual(u, F_old, state, f_ext)
            R[bc_mask_dof] = 0.0
            res_norm = torch.norm(R[free_dof]).item()

            if it == 0:
                res_norm0 = max(res_norm, 1e-30)

            if res_norm < tol or res_norm / res_norm0 < tol:
                converged_state = trial_state
                break

            # Analytical tangent via autograd through smooth return mapping
            # (de Souza Neto Section 14.5: dP/dF computed per element, then assembled)
            F_total_cur = self.fem.compute_F(u.detach())
            C_ep = self._element_tangent_dPdF(F_total_cur, F_old, state)  # (M, 9, 9)
            K_tan = self.fem.tangent_stiffness(u.detach(), lambda F: C_ep)

            # Apply BCs
            K_tan[bc_mask_dof, :] = 0.0
            K_tan[:, bc_mask_dof] = 0.0
            K_tan[bc_mask_dof, bc_mask_dof] = 1.0
            R[bc_mask_dof] = 0.0

            du = torch.linalg.solve(K_tan, -R.detach())
            du_3d = du.reshape(-1, 3)

            # Backtracking line search to ensure residual decrease
            alpha = 1.0
            u_trial = u + alpha * du_3d
            for ls_it in range(5):
                R_trial, _, _ = self._compute_residual(u_trial, F_old, state, f_ext)
                R_trial[bc_mask_dof] = 0.0
                res_trial = torch.norm(R_trial[free_dof]).item()
                if res_trial < res_norm:
                    break
                alpha *= 0.5
                u_trial = u + alpha * du_3d
            u = u_trial

            if it == max_iter - 1:
                print(f"  [EP-Newton] WARNING: did not converge after {max_iter} iters, |R| = {res_norm:.2e}")

        # --- Differentiable correction for inverse problems ---
        # At convergence, R(u*; theta) ≈ 0.  To propagate gradients w.r.t.
        # material parameters theta (tau_y, H_kin, etc.) through the
        # converged displacement u*, we apply the implicit function theorem:
        #   du*/d(theta) = -[dR/du]^{-1} dR/d(theta)
        # This is implemented as one differentiable linear solve:
        #   u_diff = u* - K^{-1} R(u*; theta)
        # where R carries the theta dependency through the return mapping.
        if K_tan is None:
            # Newton converged immediately — compute tangent for IFT step
            F_total_cur = self.fem.compute_F(u.detach())
            C_ep = self._element_tangent_dPdF(F_total_cur, F_old, state)
            K_tan = self.fem.tangent_stiffness(u.detach(), lambda F: C_ep)
            K_tan[bc_mask_dof, :] = 0.0
            K_tan[:, bc_mask_dof] = 0.0
            K_tan[bc_mask_dof, bc_mask_dof] = 1.0

        R_diff, _, converged_state = self._compute_residual(u, F_old, state, f_ext)
        R_diff[bc_mask_dof] = 0.0
        du_corr = torch.linalg.solve(K_tan.detach(), -R_diff)
        u = u.detach() + du_corr.reshape(-1, 3)

        F_new = self.fem.compute_F(u)
        return u, converged_state, F_new


# ---------------------------------------------------------------------------
# Full load-history solver
# ---------------------------------------------------------------------------

class IncrementalSolver:
    """Multi-step forward solver for elastoplastic loading histories.

    For the inverse problem, gradients flow through the material parameters
    (tau_y, H_kin, mu, K) at each step's converged state.  The Newton
    iterations themselves are not differentiated (implicit function theorem
    approach: we solve R(u*; theta)=0 then differentiate u*(theta)).
    """

    def __init__(
        self,
        fem_solver: ChartVectorFEMSolver,
        mu: torch.Tensor,
        K: torch.Tensor,
        tau_y: torch.Tensor,
        H_kin: torch.Tensor,
        epsilon: float = 1e-2,
        epsilon_schedule: Optional[Tuple[float, float, int]] = None,
    ):
        """
        Parameters
        ----------
        epsilon : float
            Default softplus sharpness.
        epsilon_schedule : (eps_start, eps_end, total_steps), optional
            Cosine annealing schedule for epsilon across load steps.
            Inspired by PNAS Bark & Sun (2025) Eq. 12.
        """
        self.fem = fem_solver
        self.mu = mu
        self.K = K
        self.tau_y = tau_y
        self.H_kin = H_kin
        self.epsilon = epsilon
        self.epsilon_schedule = epsilon_schedule

    def _get_epsilon(self, step: int, total_steps: int) -> float:
        if self.epsilon_schedule is not None:
            eps_start, eps_end, n_anneal = self.epsilon_schedule
            return cosine_anneal(step, n_anneal, eps_start, eps_end)
        return self.epsilon

    def solve_history(
        self,
        bc_schedule: List[torch.Tensor],
        bc_mask: torch.Tensor,
        sensor_nodes: Optional[torch.Tensor] = None,
        f_ext: Optional[torch.Tensor] = None,
        max_newton_iter: int = 25,
        tol: float = 1e-8,
        verbose: bool = True,
    ) -> Tuple[List[torch.Tensor], List[ReturnMappingState]]:
        """Solve the full load history.

        Returns (u_history, state_history).
        """
        device = self.fem.device
        dtype = self.fem.dtype
        N = self.fem.n_nodes
        M = self.fem.n_elements

        u = torch.zeros(N, 3, device=device, dtype=dtype)
        I33 = torch.eye(3, device=device, dtype=dtype)
        F_old = I33.unsqueeze(0).expand(M, 3, 3).clone()
        state = ReturnMappingState.zeros((M,), device=device, dtype=dtype)

        u_history: List[torch.Tensor] = []
        state_history: List[ReturnMappingState] = []

        n_steps = len(bc_schedule)

        for step_idx, u_bc in enumerate(bc_schedule):
            if verbose and step_idx % max(1, n_steps // 10) == 0:
                print(f"  [Incremental] step {step_idx+1}/{n_steps}")

            eps_current = self._get_epsilon(step_idx, n_steps)
            step_solver = ElastoplasticStepSolver(
                self.fem, self.mu, self.K, self.tau_y, self.H_kin, eps_current,
            )

            u, state, F_new = step_solver.solve_step(
                u, state, F_old, u_bc, bc_mask, f_ext,
                max_iter=max_newton_iter, tol=tol,
            )

            # Detach F_old to avoid graph growth; gradients flow through
            # material parameters at the converged state evaluation
            F_old = F_new.detach()

            u_history.append(u)
            state_history.append(state)

        return u_history, state_history


# ---------------------------------------------------------------------------
# Utility: cyclic torsion+bending load schedule for a torus
# ---------------------------------------------------------------------------

def torus_cyclic_bc_schedule(
    nodes: torch.Tensor,
    n_steps_per_half: int = 20,
    n_cycles: int = 2,
    max_twist_angle: float = 0.05,
    R_major: float = 1.0,
    r_minor: float = 0.3,
) -> List[torch.Tensor]:
    """Generate a cyclic torsion/bending load schedule on a torus cross-section."""
    device = nodes.device
    dtype = nodes.dtype
    N = nodes.shape[0]

    phi = torch.atan2(nodes[:, 1], nodes[:, 0])
    r_proj = torch.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2)
    x_local = r_proj - R_major
    z_local = nodes[:, 2]

    total_steps = 4 * n_cycles * n_steps_per_half
    schedule = []

    for step in range(total_steps):
        t = step / (2 * n_steps_per_half)
        lam = max_twist_angle * math.sin(2.0 * math.pi * t / (2.0 * n_cycles))

        angle = lam * phi
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        x_new = x_local * cos_a - z_local * sin_a
        z_new = x_local * sin_a + z_local * cos_a

        dx_local = x_new - x_local
        dz = z_new - z_local

        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        u = torch.zeros(N, 3, device=device, dtype=dtype)
        u[:, 0] = dx_local * cos_phi
        u[:, 1] = dx_local * sin_phi
        u[:, 2] = dz

        schedule.append(u)

    return schedule


# ===================================================================
# Self-tests
# ===================================================================

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    device = "cpu"

    print("=" * 60)
    print("IncrementalSolver self-test")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Test 1: elastic-only loading (tau_y very high -> no plasticity)
    # ---------------------------------------------------------------
    print()
    print("--- Test 1: Elastic-only (high yield stress) ---")

    n_cells = 4
    solver = ChartVectorFEMSolver(
        n_cells=n_cells, support_r=0.5, device=device, dtype=torch.float64,
    )

    E_val, nu_val = 200.0, 0.3
    mu_val = E_val / (2 * (1 + nu_val))
    K_val = E_val / (3 * (1 - 2 * nu_val))

    mu = torch.tensor(mu_val, device=device)
    K = torch.tensor(K_val, device=device)
    tau_y = torch.tensor(100.0, device=device)  # very high — elastic only
    H_kin = torch.tensor(0.0, device=device)

    inc_solver = IncrementalSolver(solver, mu, K, tau_y, H_kin, epsilon=1e-3)

    bc_mask = solver.boundary_mask
    nodes = solver.nodes

    eps_max = 0.005
    n_steps = 5
    bc_schedule = []
    for step in range(n_steps):
        lam = (step + 1) / n_steps
        u_bc = torch.zeros_like(nodes)
        u_bc[:, 0] = eps_max * lam * nodes[:, 0]
        bc_schedule.append(u_bc)

    u_hist, state_hist = inc_solver.solve_history(
        bc_schedule, bc_mask, verbose=True, max_newton_iter=15, tol=1e-10,
    )

    final_ep = state_hist[-1].ep_bar
    max_ep = final_ep.max().item()
    passed = max_ep < 1e-6
    print(f"  Max accumulated plastic strain: {max_ep:.2e} {'[PASS]' if passed else '[FAIL]'}")

    # ---------------------------------------------------------------
    # Test 2: plastic loading — verify plastic strain accumulation
    # ---------------------------------------------------------------
    print()
    print("--- Test 2: Plastic loading (low yield stress) ---")

    tau_y_low = torch.tensor(0.5, device=device)
    H_kin_val = torch.tensor(10.0, device=device)

    inc_solver2 = IncrementalSolver(
        solver, mu, K, tau_y_low, H_kin_val, epsilon=1e-3,
    )

    eps_max2 = 0.03
    n_steps2 = 10
    bc_schedule2 = []
    for step in range(n_steps2):
        lam = (step + 1) / n_steps2
        u_bc = torch.zeros_like(nodes)
        u_bc[:, 0] = eps_max2 * lam * nodes[:, 0]
        bc_schedule2.append(u_bc)

    u_hist2, state_hist2 = inc_solver2.solve_history(
        bc_schedule2, bc_mask, verbose=True, max_newton_iter=25, tol=1e-8,
    )

    final_ep2 = state_hist2[-1].ep_bar
    max_ep2 = final_ep2.max().item()
    has_plasticity = max_ep2 > 0.001
    print(f"  Max accumulated plastic strain: {max_ep2:.4f} "
          f"{'[PASS]' if has_plasticity else '[FAIL]'}")

    # ---------------------------------------------------------------
    # Test 3: autograd through tau_y (differentiable forward solve)
    # ---------------------------------------------------------------
    print()
    print("--- Test 3: Autograd through tau_y ---")

    tau_y_ad = torch.tensor(0.5, device=device, requires_grad=True)

    inc_solver3 = IncrementalSolver(
        solver, mu, K, tau_y_ad, H_kin_val, epsilon=1e-3,
    )

    # Just 2 steps for speed
    bc_schedule3 = bc_schedule2[:2]
    u_hist3, _ = inc_solver3.solve_history(
        bc_schedule3, bc_mask, verbose=False, max_newton_iter=20, tol=1e-8,
    )

    target_node = solver.n_nodes // 2
    loss = u_hist3[-1][target_node, 0]
    loss.backward()
    grad = tau_y_ad.grad
    passed = grad is not None and torch.isfinite(grad).all()
    print(f"  d(u_x)/d(tau_y) at node {target_node} = {grad}")
    print(f"  Gradient exists and finite: {'[PASS]' if passed else '[FAIL]'}")

    # ---------------------------------------------------------------
    # Test 4: epsilon annealing schedule
    # ---------------------------------------------------------------
    print()
    print("--- Test 4: Cosine epsilon annealing ---")
    eps_vals = [cosine_anneal(i, 100, 0.1, 0.001) for i in range(100)]
    assert abs(eps_vals[0] - 0.1) < 1e-10, f"Start: {eps_vals[0]}"
    assert abs(eps_vals[-1] - 0.001) < 1e-10, f"End: {eps_vals[-1]}"
    monotonic = all(eps_vals[i] >= eps_vals[i+1] - 1e-15 for i in range(len(eps_vals)-1))
    print(f"  eps[0]={eps_vals[0]:.4f}, eps[49]={eps_vals[49]:.4f}, eps[99]={eps_vals[99]:.4f}")
    print(f"  Monotonically decreasing: {'[PASS]' if monotonic else '[FAIL]'}")

    print()
    print("=" * 60)
    print("IncrementalSolver self-test complete.")
    print("=" * 60)
