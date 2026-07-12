#!/usr/bin/env python3
"""Multi-chart Schwarz domain decomposition solver for 3D vector elastoplasticity.

Manages multiple ChartVectorFEMSolver instances, exchanging displacement BCs
at chart overlaps via multiplicative alternating Schwarz iteration.  Plastic
state (Be, ep_bar, beta) is element-local and never communicated between charts.

Follows the patterns from the scalar Poisson Schwarz solver
(run_poisson_rabbit_atlas_schwarz_fem.py) extended to 3 DOFs per node.

All PyTorch, dense matrices, CPU, float64.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch

from .chart_vector_fem import ChartVectorFEMSolver
from .return_mapping import ReturnMappingState


# ---------------------------------------------------------------------------
# Color-group assignment (greedy graph coloring from overlap adjacency)
# ---------------------------------------------------------------------------

def choose_color_groups(
    n_charts: int,
    adjacency: Dict[int, set],
) -> List[List[int]]:
    """Greedy coloring: group non-adjacent charts for parallel update.

    Parameters
    ----------
    n_charts : int
        Total number of charts.
    adjacency : dict
        adjacency[i] is the set of chart indices that overlap with chart i.

    Returns
    -------
    List of groups; charts within one group share no overlap.
    """
    color_of = {}
    for i in range(n_charts):
        used = {color_of[j] for j in adjacency.get(i, set()) if j in color_of}
        c = 0
        while c in used:
            c += 1
        color_of[i] = c
    max_color = max(color_of.values(), default=0)
    groups: List[List[int]] = [[] for _ in range(max_color + 1)]
    for i, c in sorted(color_of.items()):
        groups[c].append(i)
    return [g for g in groups if g]


# ---------------------------------------------------------------------------
# Interface geometry precomputation
# ---------------------------------------------------------------------------

def _nodes_in_box(nodes: torch.Tensor, half_width: float) -> torch.Tensor:
    """Boolean mask: which nodes lie inside [-hw, hw]^3."""
    return torch.all(torch.abs(nodes) <= half_width, dim=1)


def precompute_interface_geometry(
    solvers: List[ChartVectorFEMSolver],
    chart_decoders: Optional[List[Callable]] = None,
    decoder_kwargs_list: Optional[List[dict]] = None,
) -> Dict:
    """Find, for each boundary node of chart A, its location in chart B's
    reference domain via Newton inversion.

    When chart_decoders are provided, Newton inversion maps physical
    coordinates back to each neighbor's reference domain.  When they are
    None (simple test cases without decoders), the charts share physical
    space directly and we just check coordinate containment.

    Returns
    -------
    cache : dict
        cache[(i, j)] = {
            "bdy_nodes_i": LongTensor of boundary node indices in chart i,
            "xi_j": (N, 3) coordinates in chart j's reference domain,
            "valid": BoolTensor mask of which nodes successfully mapped,
        }
    """
    n_charts = len(solvers)
    cache: Dict = {}

    for i in range(n_charts):
        si = solvers[i]
        bdy_idx_i = torch.where(si.boundary_mask)[0]
        if bdy_idx_i.numel() == 0:
            continue
        bdy_xi_i = si.nodes[bdy_idx_i]  # (Nb, 3)

        # Map to physical space
        if chart_decoders is not None:
            dkw = (decoder_kwargs_list[i] if decoder_kwargs_list else {})
            with torch.no_grad():
                x_phys = chart_decoders[i](bdy_xi_i, **dkw)  # (Nb, 3)
        else:
            x_phys = bdy_xi_i  # identity map

        for j in range(n_charts):
            if j == i:
                continue
            sj = solvers[j]

            # Invert chart j's decoder to find xi_j coords
            if chart_decoders is not None:
                dkw_j = (decoder_kwargs_list[j] if decoder_kwargs_list else {})
                xi_j = _newton_invert_decoder(
                    chart_decoders[j], x_phys, dkw_j,
                    device=si.device, dtype=si.dtype,
                )
            else:
                xi_j = x_phys.clone()  # identity

            # Check which points land inside chart j's mesh bounding box
            node_min = sj.nodes.min(dim=0).values
            node_max = sj.nodes.max(dim=0).values
            tol_bb = sj.h * 0.1 if sj.n_elements > 0 else 0.0
            valid = torch.all(xi_j >= node_min - tol_bb, dim=1) & \
                    torch.all(xi_j <= node_max + tol_bb, dim=1)

            if valid.any():
                cache[(i, j)] = {
                    "bdy_nodes_i": bdy_idx_i[valid],
                    "xi_j": xi_j[valid],
                }

    return cache


def _newton_invert_decoder(
    decoder: Callable,
    x_target: torch.Tensor,
    decoder_kwargs: dict,
    device: torch.device,
    dtype: torch.dtype,
    max_iter: int = 15,
    tol: float = 1e-8,
) -> torch.Tensor:
    """Find xi such that decoder(xi) = x_target via Newton iteration."""
    xi = x_target.clone().detach()
    N = xi.shape[0]

    for it in range(max_iter):
        xi_var = xi.clone().detach().requires_grad_(True)
        x_pred = decoder(xi_var, **decoder_kwargs)
        residual = x_pred - x_target
        res_norm = torch.linalg.norm(residual, dim=1)

        if res_norm.max().item() < tol:
            break

        grads = []
        for d in range(3):
            g = torch.autograd.grad(
                x_pred[:, d], xi_var,
                grad_outputs=torch.ones(N, device=device, dtype=dtype),
                create_graph=False, retain_graph=True,
            )[0]
            grads.append(g)
        J = torch.stack(grads, dim=1)  # (N, 3, 3)

        try:
            delta = torch.linalg.solve(J, residual.unsqueeze(-1)).squeeze(-1)
        except Exception:
            delta = torch.bmm(
                torch.linalg.pinv(J), residual.unsqueeze(-1)
            ).squeeze(-1)

        xi = xi - delta

    return xi.detach()


# ---------------------------------------------------------------------------
# Interpolate displacement from chart j at given reference coords
# ---------------------------------------------------------------------------

def _interpolate_u_at(
    solver: ChartVectorFEMSolver,
    u: torch.Tensor,
    xi_query: torch.Tensor,
) -> torch.Tensor:
    """Evaluate P1 displacement at query points via barycentric interpolation.

    Falls back to nearest node if no containing tet is found.
    """
    Q = xi_query.shape[0]
    u_interp = torch.zeros(Q, 3, device=solver.device, dtype=solver.dtype)
    elem_nodes = solver.nodes[solver.elements]  # (M, 4, 3)
    centroids = elem_nodes.mean(dim=1)  # (M, 3)

    for q_idx in range(Q):
        xq = xi_query[q_idx]
        dists = torch.norm(centroids - xq.unsqueeze(0), dim=1)
        _, nearest = torch.topk(dists, min(8, solver.n_elements), largest=False)
        found = False
        for e_idx in nearest:
            e = e_idx.item()
            verts = elem_nodes[e]
            B = (verts[1:] - verts[0]).T
            try:
                lam = torch.linalg.solve(B, xq - verts[0])
            except Exception:
                continue
            bary = torch.cat([(1.0 - lam.sum()).unsqueeze(0), lam])
            if torch.all(bary >= -1e-6):
                bary = bary.clamp(min=0.0)
                bary = bary / bary.sum()
                u_interp[q_idx] = (bary.unsqueeze(1) * u[solver.elements[e]]).sum(0)
                found = True
                break
        if not found:
            nn_idx = torch.argmin(torch.norm(solver.nodes - xq.unsqueeze(0), dim=1))
            u_interp[q_idx] = u[nn_idx]
    return u_interp


# ---------------------------------------------------------------------------
# SchwarzVectorSolver
# ---------------------------------------------------------------------------

class SchwarzVectorSolver:
    """Multi-chart Schwarz domain decomposition for 3D vector elastoplasticity."""

    def __init__(
        self,
        solvers: List[ChartVectorFEMSolver],
        chart_decoders: Optional[List[Callable]] = None,
        decoder_kwargs_list: Optional[List[dict]] = None,
        sdf_oracles: Optional[list] = None,
    ):
        self.solvers = solvers
        self.n_charts = len(solvers)
        self.chart_decoders = chart_decoders
        self.decoder_kwargs_list = decoder_kwargs_list or [{}] * self.n_charts
        self.sdf_oracles = sdf_oracles

        device = solvers[0].device
        dtype = solvers[0].dtype
        self.device = device
        self.dtype = dtype

        # Per-chart displacement fields
        self.u_charts: List[torch.Tensor] = [
            torch.zeros(s.n_nodes, 3, device=device, dtype=dtype)
            for s in solvers
        ]

        # Per-chart plastic state
        self.states: List[ReturnMappingState] = [
            ReturnMappingState.zeros((s.n_elements,), device=device, dtype=dtype)
            for s in solvers
        ]

        # Per-chart previous deformation gradient (for incremental plasticity)
        I33 = torch.eye(3, device=device, dtype=dtype)
        self.F_old: List[torch.Tensor] = [
            I33.unsqueeze(0).expand(s.n_elements, 3, 3).clone()
            for s in solvers
        ]

        # Build adjacency and color groups
        self.adjacency = self._build_adjacency()
        self.color_groups = choose_color_groups(self.n_charts, self.adjacency)

        # Precompute interface geometry
        self.iface_cache = precompute_interface_geometry(
            solvers, chart_decoders, decoder_kwargs_list,
        )

    def _build_adjacency(self) -> Dict[int, set]:
        """Determine chart overlap from interface cache candidates."""
        adj: Dict[int, set] = {i: set() for i in range(self.n_charts)}
        # Two charts overlap if any boundary node of one maps into the other
        for i in range(self.n_charts):
            for j in range(self.n_charts):
                if i == j:
                    continue
                key = (i, j)
                # We check both directions during precompute; just mark if present
                # Try a quick geometric test: bounding boxes overlap
                si, sj = self.solvers[i], self.solvers[j]
                if self.chart_decoders is None:
                    # Identity map: check if cube domains overlap
                    sep = 0.0
                    for d in range(3):
                        lo_i = si.nodes[:, d].min().item()
                        hi_i = si.nodes[:, d].max().item()
                        lo_j = sj.nodes[:, d].min().item()
                        hi_j = sj.nodes[:, d].max().item()
                        if hi_i < lo_j or hi_j < lo_i:
                            sep = 1.0
                            break
                    if sep == 0.0 and i != j:
                        adj[i].add(j)
                        adj[j].add(i)
                else:
                    # With decoders, assume all charts may overlap
                    adj[i].add(j)
        return adj

    def _interpolate_interface_bc(
        self, chart_i: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute displacement BCs at chart_i's boundary from neighbors."""
        si = self.solvers[chart_i]
        bc_mask = torch.zeros(si.n_nodes, device=self.device, dtype=torch.bool)
        bc_vals = torch.zeros(si.n_nodes, 3, device=self.device, dtype=self.dtype)
        weight_sum = torch.zeros(si.n_nodes, device=self.device, dtype=self.dtype)

        for j in range(self.n_charts):
            if j == chart_i:
                continue
            key = (chart_i, j)
            if key not in self.iface_cache:
                continue

            entry = self.iface_cache[key]
            bdy_nodes = entry["bdy_nodes_i"]
            xi_j = entry["xi_j"]

            # Interpolate chart j's displacement at these reference coords
            u_j_interp = _interpolate_u_at(
                self.solvers[j], self.u_charts[j], xi_j,
            )

            bc_mask[bdy_nodes] = True
            bc_vals[bdy_nodes] += u_j_interp
            weight_sum[bdy_nodes] += 1.0

        # Average contributions from multiple overlapping neighbors
        has_bc = weight_sum > 0.5
        bc_vals[has_bc] /= weight_sum[has_bc].unsqueeze(1)

        return bc_mask, bc_vals

    def solve_load_step(
        self,
        stress_fn: Callable,
        tangent_fn: Callable,
        f_ext_charts: Optional[List[torch.Tensor]] = None,
        u_bc_charts: Optional[List[torch.Tensor]] = None,
        phys_bc_masks: Optional[List[torch.Tensor]] = None,
        n_schwarz: int = 5,
        max_newton: int = 20,
        newton_tol: float = 1e-8,
    ) -> List[torch.Tensor]:
        """Solve one load step: outer Schwarz loop wraps inner Newton per chart."""
        for sweep in range(n_schwarz):
            for group in self.color_groups:
                for chart_i in group:
                    si = self.solvers[chart_i]
                    if si.n_elements == 0:
                        continue

                    # Combine physical BCs and interface BCs
                    iface_mask, iface_vals = self._interpolate_interface_bc(chart_i)

                    bc_mask = torch.zeros(
                        si.n_nodes, device=self.device, dtype=torch.bool
                    )
                    bc_vals = torch.zeros(
                        si.n_nodes, 3, device=self.device, dtype=self.dtype
                    )

                    # Physical BCs
                    if phys_bc_masks is not None and phys_bc_masks[chart_i] is not None:
                        bc_mask |= phys_bc_masks[chart_i]
                        if u_bc_charts is not None and u_bc_charts[chart_i] is not None:
                            phys_m = phys_bc_masks[chart_i]
                            bc_vals[phys_m] = u_bc_charts[chart_i][phys_m]

                    # Interface BCs (only on nodes without physical BCs)
                    iface_only = iface_mask & ~bc_mask
                    bc_mask |= iface_only
                    bc_vals[iface_only] = iface_vals[iface_only]

                    # External forces
                    f_ext = (
                        f_ext_charts[chart_i]
                        if f_ext_charts is not None
                        else torch.zeros(si.n_nodes, 3, device=self.device, dtype=self.dtype)
                    )

                    # Newton solve on this chart
                    u_new = si.solve_nonlinear(
                        stress_fn=stress_fn,
                        tangent_fn=tangent_fn,
                        f_ext=f_ext,
                        u_bc=bc_vals,
                        bc_mask=bc_mask,
                        u_init=self.u_charts[chart_i],
                        max_iter=max_newton,
                        tol=newton_tol,
                    )

                    self.u_charts[chart_i] = u_new

        return self.u_charts

    def interface_jump(self) -> float:
        """Max L2 displacement jump at interface nodes across all chart pairs."""
        max_jump = 0.0
        for (i, j), entry in self.iface_cache.items():
            bdy_nodes_i = entry["bdy_nodes_i"]
            xi_j = entry["xi_j"]

            u_i_vals = self.u_charts[i][bdy_nodes_i]
            u_j_vals = _interpolate_u_at(
                self.solvers[j], self.u_charts[j], xi_j,
            )

            jump = torch.norm(u_i_vals - u_j_vals, dim=1).max().item()
            max_jump = max(max_jump, jump)

        return max_jump


# ======================================================================
# Self-test: 2-chart overlapping cubes, Neo-Hookean
# ======================================================================

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    device = "cpu"

    print("=" * 70)
    print("SchwarzVectorSolver self-test")
    print("=" * 70)

    # Two cubes with overlap in x: A centered at -0.35, B at +0.35, r=0.65
    n_cells = 6
    support_r = 0.65

    solver_a = ChartVectorFEMSolver(
        n_cells=n_cells, support_r=support_r, mesh_extent=1.0,
        device=device, dtype=torch.float64,
    )
    solver_a.nodes = solver_a.nodes.clone()
    solver_a.nodes[:, 0] -= 0.35

    solver_b = ChartVectorFEMSolver(
        n_cells=n_cells, support_r=support_r, mesh_extent=1.0,
        device=device, dtype=torch.float64,
    )
    solver_b.nodes = solver_b.nodes.clone()
    solver_b.nodes[:, 0] += 0.35

    # Recompute element quantities and boundary after shifting
    for s in [solver_a, solver_b]:
        s._precompute_element_quantities()
        tol_bdy = s.h * 0.01
        on_face = torch.zeros(s.n_nodes, device=device, dtype=torch.bool)
        for d in range(3):
            lo = s.nodes[:, d].min()
            hi = s.nodes[:, d].max()
            on_face |= (torch.abs(s.nodes[:, d] - lo) < tol_bdy)
            on_face |= (torch.abs(s.nodes[:, d] - hi) < tol_bdy)
        s.boundary_mask = on_face

    solvers = [solver_a, solver_b]

    print(f"\nChart A: {solver_a.n_nodes} nodes, domain x in "
          f"[{solver_a.nodes[:, 0].min():.2f}, {solver_a.nodes[:, 0].max():.2f}]")
    print(f"Chart B: {solver_b.n_nodes} nodes, domain x in "
          f"[{solver_b.nodes[:, 0].min():.2f}, {solver_b.nodes[:, 0].max():.2f}]")

    # Material: Neo-Hookean
    mu_val, K_val = 1.0, 10.0
    stress_fn_a, tangent_fn_a = solver_a.make_neo_hookean(mu_val, K_val)

    # BCs: fix left face, prescribe u_x on right face
    global_x_min = min(solver_a.nodes[:, 0].min().item(),
                       solver_b.nodes[:, 0].min().item())
    global_x_max = max(solver_a.nodes[:, 0].max().item(),
                       solver_b.nodes[:, 0].max().item())

    tol_face = solver_a.h * 0.1
    prescribed_ux = 0.02

    phys_bc_masks = []
    u_bc_list = []
    for s in solvers:
        mask = torch.zeros(s.n_nodes, device=device, dtype=torch.bool)
        u_bc = torch.zeros(s.n_nodes, 3, device=device, dtype=torch.float64)
        left = torch.abs(s.nodes[:, 0] - global_x_min) < tol_face
        right = torch.abs(s.nodes[:, 0] - global_x_max) < tol_face
        mask |= left
        mask |= right
        u_bc[right, 0] = prescribed_ux
        phys_bc_masks.append(mask)
        u_bc_list.append(u_bc)

    schwarz = SchwarzVectorSolver(solvers=solvers)

    print(f"\nColor groups: {schwarz.color_groups}")
    print(f"Interface cache entries: {len(schwarz.iface_cache)}")
    for key, entry in schwarz.iface_cache.items():
        print(f"  ({key[0]}->{key[1]}): {entry['bdy_nodes_i'].numel()} interface nodes")

    print("\n--- Schwarz convergence test ---")

    jumps = []
    for n_sweeps in [1, 2, 3, 4, 5, 6, 8]:
        # Reset
        schwarz.u_charts = [
            torch.zeros(s.n_nodes, 3, device=device, dtype=torch.float64)
            for s in solvers
        ]

        schwarz.solve_load_step(
            stress_fn=stress_fn_a,
            tangent_fn=tangent_fn_a,
            u_bc_charts=u_bc_list,
            phys_bc_masks=phys_bc_masks,
            n_schwarz=n_sweeps,
            max_newton=15,
            newton_tol=1e-10,
        )

        jump = schwarz.interface_jump()
        jumps.append(jump)
        print(f"  sweeps={n_sweeps:2d}  interface jump = {jump:.6e}")

    print("\n--- Convergence check ---")
    decreasing = all(jumps[i + 1] <= jumps[i] * 1.05 for i in range(len(jumps) - 1))
    final_decreased = jumps[-1] < jumps[0] * 0.5
    passed = decreasing and final_decreased

    print(f"  First jump:  {jumps[0]:.6e}")
    print(f"  Final jump:  {jumps[-1]:.6e}")
    print(f"  Ratio:       {jumps[-1] / max(jumps[0], 1e-30):.4f}")
    print(f"  Monotonically decreasing: {decreasing}")
    print(f"  Final < 50% of first:     {final_decreased}")
    print(f"  RESULT: {'PASSED' if passed else 'FAILED'}")

    # ------------------------------------------------------------------
    # Verify displacement continuity improves
    # ------------------------------------------------------------------
    print("\n--- Displacement field check ---")
    # Check that solutions are nonzero
    u_a_norm = torch.norm(schwarz.u_charts[0]).item()
    u_b_norm = torch.norm(schwarz.u_charts[1]).item()
    print(f"  |u_A| = {u_a_norm:.6e}")
    print(f"  |u_B| = {u_b_norm:.6e}")
    nontrivial = u_a_norm > 1e-6 and u_b_norm > 1e-6
    print(f"  Nontrivial solution: {'PASSED' if nontrivial else 'FAILED'}")

    print("\n" + "=" * 70)
    print("Self-test complete.")
    print("=" * 70)
