#!/usr/bin/env python3
"""
P1 tetrahedral FEM solver for 3D vector-valued nonlinear elastostatics on a
single chart's reference domain.

Extends the scalar Poisson solver (ChartFEMSolver) to vector elasticity with:
  - Neo-Hookean hyperelastic material (built-in)
  - Newton-Raphson nonlinear solver
  - Differentiable force/stiffness assembly in PyTorch

Mesh: structured hex grid in [-r, r]^3, subdivided into 6 tets/hex
(Freudenthal decomposition), optionally SDF-filtered.
"""

import math
from typing import Callable, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Freudenthal 6-tet decomposition of a unit cube
# 8 corners indexed by (dz*4 + dy*2 + dx).
# ---------------------------------------------------------------------------
_FREUDENTHAL_TETS = [
    [0, 1, 3, 7],
    [0, 1, 5, 7],
    [0, 2, 3, 7],
    [0, 2, 6, 7],
    [0, 4, 5, 7],
    [0, 4, 6, 7],
]

_HEX_CORNERS = [
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
]


class ChartVectorFEMSolver:
    """P1 tetrahedral FEM solver for 3D vector nonlinear elastostatics on one
    chart's reference domain.

    The solver works entirely in PyTorch for GPU support and differentiability.

    Parameters
    ----------
    n_cells : int
        Grid resolution per axis (number of hex cells).
    support_r : float
        Chart radius; the mesh spans [-r, r]^3 where r = support_r * mesh_extent.
    sdf_oracle : optional
        Object with ``.sdf(x)`` method returning signed distance. Used to
        filter elements and classify boundary nodes.
    chart_decoder : optional
        ``torch.nn.Module`` mapping chart coords xi -> physical coords x.
        When provided, physical mapped geometry is used for SDF filtering,
        deformation gradients, internal forces, and tangent assembly.
    decoder_kwargs : dict, optional
        Extra kwargs passed to chart_decoder (seed, t1, t2, n, chart_scale, ...).
    sdf_threshold : float
        Elements with centroid SDF above this are removed.
    mesh_extent : float
        Multiplier on support_r to set actual mesh half-width.
    device : str or torch.device
        Compute device.
    dtype : torch.dtype
        Floating-point precision (default float64 for elasticity).
    """

    def __init__(
        self,
        n_cells: int = 8,
        support_r: float = 1.0,
        sdf_oracle: Optional[object] = None,
        chart_decoder: Optional["torch.nn.Module"] = None,
        decoder_kwargs: Optional[dict] = None,
        sdf_threshold: float = -0.005,
        mesh_extent: float = 1.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.n_cells = n_cells
        self.support_r = support_r
        self.sdf_oracle = sdf_oracle
        self.chart_decoder = chart_decoder
        self.decoder_kwargs = decoder_kwargs or {}
        self.sdf_threshold = sdf_threshold
        self.mesh_extent = mesh_extent
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype

        self.r = support_r * mesh_extent
        self.h = 2.0 * self.r / n_cells

        # Populated by _build_mesh
        self.nodes: torch.Tensor = torch.empty(0, 3, device=self.device, dtype=self.dtype)
        self.nodes_phys: torch.Tensor = torch.empty(0, 3, device=self.device, dtype=self.dtype)
        self.elements: torch.Tensor = torch.empty(0, 4, device=self.device, dtype=torch.long)
        self.n_nodes: int = 0
        self.n_elements: int = 0
        self.boundary_mask: torch.Tensor = torch.empty(0, device=self.device, dtype=torch.bool)

        # Cached per-element quantities (set by _build_mesh)
        self.dNdx: torch.Tensor = torch.empty(0, 4, 3, device=self.device, dtype=self.dtype)
        self.vol: torch.Tensor = torch.empty(0, device=self.device, dtype=self.dtype)
        self.dNdx_ref: torch.Tensor = torch.empty(0, 4, 3, device=self.device, dtype=self.dtype)
        self.dNdx_phys: torch.Tensor = torch.empty(0, 4, 3, device=self.device, dtype=self.dtype)
        self.vol_ref: torch.Tensor = torch.empty(0, device=self.device, dtype=self.dtype)
        self.vol_phys: torch.Tensor = torch.empty(0, device=self.device, dtype=self.dtype)
        self.elem_centroids_ref: torch.Tensor = torch.empty(0, 3, device=self.device, dtype=self.dtype)
        self.elem_centroids_phys: torch.Tensor = torch.empty(0, 3, device=self.device, dtype=self.dtype)
        self.geom_J: torch.Tensor = torch.empty(0, 3, 3, device=self.device, dtype=self.dtype)
        self.geom_J_inv: torch.Tensor = torch.empty(0, 3, 3, device=self.device, dtype=self.dtype)
        self.geom_detJ: torch.Tensor = torch.empty(0, device=self.device, dtype=self.dtype)

        self._build_mesh()

    # ------------------------------------------------------------------
    # Mesh generation
    # ------------------------------------------------------------------
    def _build_mesh(self) -> None:
        """Generate structured hex grid, subdivide into tets, filter by SDF,
        compute and cache shape function gradients and element volumes."""
        nc = self.n_cells
        r = self.r
        npa = nc + 1  # nodes per axis

        # Regular grid nodes
        lin = torch.linspace(-r, r, npa, device=self.device, dtype=self.dtype)
        gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
        all_nodes = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)

        # Build hex cell indices
        ix = torch.arange(nc, device=self.device)
        iy = torch.arange(nc, device=self.device)
        iz = torch.arange(nc, device=self.device)
        gix, giy, giz = torch.meshgrid(ix, iy, iz, indexing="ij")
        hex_i = gix.reshape(-1)
        hex_j = giy.reshape(-1)
        hex_k = giz.reshape(-1)
        n_hex = hex_i.shape[0]

        # 8 corner node indices for each hex
        corners_offsets = torch.tensor(_HEX_CORNERS, device=self.device, dtype=torch.long)  # (8, 3)
        # node_idx(ix, iy, iz) = ix * npa^2 + iy * npa + iz
        hex_corners = torch.zeros(n_hex, 8, device=self.device, dtype=torch.long)
        for c in range(8):
            dx, dy, dz = corners_offsets[c]
            hex_corners[:, c] = (hex_i + dx) * npa * npa + (hex_j + dy) * npa + (hex_k + dz)

        # Subdivide each hex into 6 tets
        tet_templates = torch.tensor(_FREUDENTHAL_TETS, device=self.device, dtype=torch.long)  # (6, 4)
        all_tets = hex_corners[:, tet_templates.reshape(-1)].reshape(n_hex * 6, 4)

        # SDF filtering
        if self.sdf_oracle is not None and self.chart_decoder is not None:
            centroids_xi = all_nodes[all_tets].mean(dim=1)  # (M, 3)
            with torch.no_grad():
                x_phys = self.chart_decoder(centroids_xi, **self.decoder_kwargs)
                sdf_vals = self.sdf_oracle.sdf(x_phys)
            keep = sdf_vals < self.sdf_threshold
            all_tets = all_tets[keep]

        if all_tets.shape[0] == 0:
            self.nodes = all_nodes
            if self.chart_decoder is not None:
                with torch.no_grad():
                    self.nodes_phys = self.chart_decoder(self.nodes, **self.decoder_kwargs)
            else:
                self.nodes_phys = self.nodes.clone()
            self.n_nodes = all_nodes.shape[0]
            self.n_elements = 0
            self.boundary_mask = torch.zeros(self.n_nodes, device=self.device, dtype=torch.bool)
            return

        # Compact nodes: keep only referenced nodes
        used_mask = torch.zeros(all_nodes.shape[0], device=self.device, dtype=torch.bool)
        used_mask[all_tets.reshape(-1)] = True
        used_indices = torch.where(used_mask)[0]
        old_to_new = torch.full((all_nodes.shape[0],), -1, device=self.device, dtype=torch.long)
        old_to_new[used_indices] = torch.arange(used_indices.shape[0], device=self.device)

        self.nodes = all_nodes[used_indices]
        self.elements = old_to_new[all_tets]
        self.n_nodes = self.nodes.shape[0]
        self.n_elements = self.elements.shape[0]
        if self.chart_decoder is not None:
            with torch.no_grad():
                self.nodes_phys = self.chart_decoder(self.nodes, **self.decoder_kwargs)
        else:
            self.nodes_phys = self.nodes.clone()

        # Classify boundary nodes
        self._classify_boundary()

        # Pre-compute shape function gradients and volumes
        self._precompute_element_quantities()

        print(
            f"  [ChartVectorFEM] mesh: {self.n_nodes} nodes, {self.n_elements} tets, "
            f"{int(self.boundary_mask.sum())} boundary nodes"
        )

    def _classify_boundary(self) -> None:
        """Identify boundary nodes.

        For structured grids (no SDF), boundary = nodes on cube faces.
        For SDF-filtered meshes, boundary = nodes on faces shared by only one tet.
        """
        if self.n_elements == 0:
            self.boundary_mask = torch.zeros(self.n_nodes, device=self.device, dtype=torch.bool)
            return

        if self.sdf_oracle is None:
            # Fast path: cube face detection
            r = self.r
            tol = self.h * 0.01
            on_face = torch.any(torch.abs(torch.abs(self.nodes) - r) < tol, dim=1)
            self.boundary_mask = on_face
        else:
            # Face-counting approach
            face_combos = torch.tensor(
                [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
                device=self.device, dtype=torch.long
            )
            n_el = self.n_elements
            # Extract all faces, sorted
            all_faces = []
            for combo in face_combos:
                f = self.elements[:, combo]
                f, _ = f.sort(dim=1)
                all_faces.append(f)
            all_faces = torch.cat(all_faces, dim=0)  # (4*n_el, 3)

            # Encode faces as unique integers for hashing
            max_node = self.n_nodes
            face_keys = (all_faces[:, 0] * max_node * max_node +
                         all_faces[:, 1] * max_node +
                         all_faces[:, 2])
            unique_keys, counts = torch.unique(face_keys, return_counts=True)
            boundary_keys = unique_keys[counts == 1]

            # Find which faces are boundary
            boundary_set = set(boundary_keys.cpu().tolist())
            is_boundary_face = torch.tensor(
                [k.item() in boundary_set for k in face_keys],
                device=self.device, dtype=torch.bool
            )
            boundary_face_nodes = all_faces[is_boundary_face].reshape(-1)
            self.boundary_mask = torch.zeros(self.n_nodes, device=self.device, dtype=torch.bool)
            self.boundary_mask[boundary_face_nodes] = True

    def _precompute_element_quantities(self) -> None:
        """Pre-compute shape function gradients dN/dx and element volumes.

        For P1 tets, the shape function gradients are constant per element.
        Given tet with vertices v0, v1, v2, v3:
            B = [v1-v0 | v2-v0 | v3-v0]  (3x3, columns are edge vectors)
            Volume = |det(B)| / 6
            Barycentric coords lambda_k = row k of B^{-1} dot (x - v0)
            dN_0/dx = -sum of rows of B^{-1}
            dN_k/dx = row (k-1) of B^{-1}, k=1,2,3
        """
        def _metrics_from_nodes(xe: torch.Tensor):
            n_elem = xe.shape[0]
            B = (xe[:, 1:, :] - xe[:, 0:1, :]).permute(0, 2, 1)  # (M, 3, 3)
            detB = torch.det(B)  # (M,)
            invB = torch.linalg.inv(B)  # (M, 3, 3)
            dN = torch.zeros(n_elem, 4, 3, device=self.device, dtype=self.dtype)
            dN[:, 1:, :] = invB
            dN[:, 0, :] = -invB.sum(dim=1)
            vol = torch.abs(detB) / 6.0
            return B, detB, invB, dN, vol

        xe_ref = self.nodes[self.elements]  # (M, 4, 3)
        self.elem_centroids_ref = xe_ref.mean(dim=1)
        _, _, _, dN_ref, vol_ref = _metrics_from_nodes(xe_ref)
        self.dNdx_ref = dN_ref
        self.vol_ref = vol_ref

        if self.chart_decoder is not None:
            # Use the exact decoder Jacobian at the element centroid.  This avoids
            # singular nodal-map tets at the torus centerline while still giving a
            # consistent physical push-forward for P1 chart gradients.
            self.geom_J = self.decoder_jacobian(self.elem_centroids_ref)
            self.geom_J_inv = torch.linalg.inv(self.geom_J)
            self.geom_detJ = torch.det(self.geom_J)
            self.elem_centroids_phys = self.chart_decoder(
                self.elem_centroids_ref, **self.decoder_kwargs
            )
            self.dNdx_phys = torch.einsum("eaj,ejk->eak", self.dNdx_ref, self.geom_J_inv)
            self.vol_phys = self.vol_ref * torch.abs(self.geom_detJ)
            self.dNdx = self.dNdx_phys
            self.vol = self.vol_phys
        else:
            self.elem_centroids_phys = self.elem_centroids_ref.clone()
            eye = torch.eye(3, device=self.device, dtype=self.dtype)
            self.geom_J = eye.unsqueeze(0).expand(self.n_elements, 3, 3).clone()
            self.geom_J_inv = self.geom_J.clone()
            self.geom_detJ = torch.ones(self.n_elements, device=self.device, dtype=self.dtype)
            self.dNdx_phys = self.dNdx_ref.clone()
            self.vol_phys = self.vol_ref.clone()
            self.dNdx = self.dNdx_ref
            self.vol = self.vol_ref

    def decoder_jacobian(self, xi: torch.Tensor) -> torch.Tensor:
        """Compute the exact decoder Jacobian dx/dxi at the supplied points."""
        if self.chart_decoder is None:
            I = torch.eye(3, device=xi.device, dtype=xi.dtype)
            return I.unsqueeze(0).expand(xi.shape[0], 3, 3).clone()

        if hasattr(self.chart_decoder, "jacobian"):
            return self.chart_decoder.jacobian(xi, **self.decoder_kwargs)

        xi_var = xi.detach().clone().requires_grad_(True)
        x = self.chart_decoder(xi_var, **self.decoder_kwargs)
        rows = []
        for d in range(3):
            grad_out = torch.zeros_like(x)
            grad_out[:, d] = 1.0
            g = torch.autograd.grad(
                x,
                xi_var,
                grad_outputs=grad_out,
                retain_graph=(d < 2),
                create_graph=False,
            )[0]
            rows.append(g)
        return torch.stack(rows, dim=1)

    def compute_grad_u_ref(self, u: torch.Tensor) -> torch.Tensor:
        """Compute the chart-coordinate displacement gradient grad_xi(u)."""
        u_elem = u[self.elements]
        return torch.einsum("eai, eaj -> eij", u_elem, self.dNdx_ref)

    def compute_grad_u_phys(self, u: torch.Tensor) -> torch.Tensor:
        """Compute the physical displacement gradient grad_x(u)."""
        u_elem = u[self.elements]
        basis_grads = self.dNdx_phys if self.chart_decoder is not None else self.dNdx_ref
        return torch.einsum("eai, eaj -> eij", u_elem, basis_grads)

    def compute_F_ref(self, u: torch.Tensor) -> torch.Tensor:
        """Compute the reference-space deformation gradient I + grad_xi(u)."""
        grad_u = self.compute_grad_u_ref(u)
        I = torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0)
        return I + grad_u

    def compute_F_phys(self, u: torch.Tensor) -> torch.Tensor:
        """Compute the physical deformation gradient I + grad_x(u)."""
        grad_u = self.compute_grad_u_phys(u)
        I = torch.eye(3, device=self.device, dtype=self.dtype).unsqueeze(0)
        return I + grad_u

    # ------------------------------------------------------------------
    # Deformation gradient
    # ------------------------------------------------------------------
    def compute_F(self, u: torch.Tensor, physical: Optional[bool] = None) -> torch.Tensor:
        """Compute deformation gradient F = I + grad(u) at each element.

        Parameters
        ----------
        u : torch.Tensor, shape (N, 3)
            Nodal displacement vector.
        physical : bool, optional
            When ``True``, use the physical mapped geometry.  When ``False``,
            use the chart/reference coordinates.  Defaults to physical space
            whenever a chart decoder is present.

        Returns
        -------
        F : torch.Tensor, shape (M, 3, 3)
            Deformation gradient per element.
        """
        if physical is None:
            physical = self.chart_decoder is not None
        return self.compute_F_phys(u) if physical else self.compute_F_ref(u)

    # ------------------------------------------------------------------
    # Internal force vector
    # ------------------------------------------------------------------
    def internal_forces(
        self,
        u: torch.Tensor,
        stress_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Compute internal force vector from current displacement.

        Parameters
        ----------
        u : torch.Tensor, shape (N, 3)
            Nodal displacements.
        stress_fn : callable
            Maps F (M, 3, 3) -> P (M, 3, 3), first Piola-Kirchhoff stress.

        Returns
        -------
        f_int : torch.Tensor, shape (N, 3)
            Internal force vector (assembled).
        """
        F = self.compute_F(u)  # (M, 3, 3)
        P = stress_fn(F)       # (M, 3, 3)

        # Element force contribution: f_e[a, i] = vol_e * sum_j P[e, i, j] * dNdx[e, a, j]
        # (M, 4, 3) = vol (M,1,1) * einsum P(M,i,j) * dNdx(M,a,j) -> (M,a,i)
        f_elem = self.vol[:, None, None] * torch.einsum(
            "eij, eaj -> eai", P, self.dNdx
        )  # (M, 4, 3)

        # Scatter to global — flatten (M,4) → (M*4) for scatter into (N, 3)
        f_int = torch.zeros(self.n_nodes, 3, device=self.device, dtype=self.dtype)
        f_elem_flat = f_elem.reshape(-1, 3)                        # (M*4, 3)
        idx_flat = self.elements.reshape(-1).unsqueeze(-1).expand(-1, 3)  # (M*4, 3)
        f_int.scatter_add_(0, idx_flat, f_elem_flat)
        return f_int

    # ------------------------------------------------------------------
    # Tangent stiffness matrix
    # ------------------------------------------------------------------
    def tangent_stiffness(
        self,
        u: torch.Tensor,
        tangent_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Compute assembled tangent stiffness matrix (dense).

        Parameters
        ----------
        u : torch.Tensor, shape (N, 3)
            Nodal displacements.
        tangent_fn : callable
            Maps F (M, 3, 3) -> C (M, 9, 9), the material tangent dP/dF in
            Voigt-like ordering (row-major flattening of 3x3).

        Returns
        -------
        K : torch.Tensor, shape (3N, 3N)
            Global tangent stiffness matrix (dense).
        """
        F = self.compute_F(u)  # (M, 3, 3)
        C = tangent_fn(F)      # (M, 9, 9)

        n_dof = 3 * self.n_nodes

        # Build element B-matrix: B_e has shape (9, 12) for each element
        # Maps 12-DOF element displacement to 9-component grad(u) (row-major F)
        # F_iJ = delta_iJ + sum_a u_a_i * dN_a / dx_J
        # In row-major: component (3*i + J) = sum_a u_{a,i} * dNdx_{a,J}
        # So B[(3*i+J), (3*a+i)] = dNdx[a, J]
        # More systematically: for node a, dof offset 3*a+i contributes to
        # F component (3*i+J) with weight dNdx[a, J].

        # B_elem: (M, 9, 12) — build vectorized
        B = torch.zeros(self.n_elements, 9, 12, device=self.device, dtype=self.dtype)
        for a in range(4):
            for i in range(3):
                for J in range(3):
                    # F_{iJ} <-> row 3*i+J;  u_{a,i} <-> col 3*a+i
                    B[:, 3 * i + J, 3 * a + i] = self.dNdx[:, a, J]

        # Element stiffness: K_e = vol_e * B^T @ C @ B  (12x12)
        # (M, 12, 9) @ (M, 9, 9) @ (M, 9, 12) -> (M, 12, 12)
        BtC = torch.einsum("eji, ejk -> eik", B, C)  # (M, 12, 9)
        K_elem = self.vol[:, None, None] * torch.einsum("eij, ejk -> eik", BtC, B)  # (M, 12, 12)

        # Build global DOF index map: (M, 12)
        # For element e, local DOF 3*a+i maps to global DOF 3*elements[e,a]+i
        dof_map = torch.zeros(self.n_elements, 12, device=self.device, dtype=torch.long)
        for a in range(4):
            for i in range(3):
                dof_map[:, 3 * a + i] = 3 * self.elements[:, a] + i

        # Scatter into dense global matrix
        K_global = torch.zeros(n_dof, n_dof, device=self.device, dtype=self.dtype)

        # Use index_put_ with accumulate
        row_idx = dof_map.unsqueeze(2).expand_as(K_elem).reshape(-1)
        col_idx = dof_map.unsqueeze(1).expand_as(K_elem).reshape(-1)
        vals = K_elem.reshape(-1)
        K_global.index_put_((row_idx, col_idx), vals, accumulate=True)

        return K_global

    # ------------------------------------------------------------------
    # Newton-Raphson nonlinear solve
    # ------------------------------------------------------------------
    def solve_nonlinear(
        self,
        stress_fn: Callable[[torch.Tensor], torch.Tensor],
        tangent_fn: Callable[[torch.Tensor], torch.Tensor],
        f_ext: torch.Tensor,
        u_bc: torch.Tensor,
        bc_mask: torch.Tensor,
        u_init: Optional[torch.Tensor] = None,
        max_iter: int = 20,
        tol: float = 1e-8,
    ) -> torch.Tensor:
        """Newton-Raphson iteration for nonlinear equilibrium R(u) = f_int - f_ext = 0.

        Parameters
        ----------
        stress_fn : callable
            F (M,3,3) -> P (M,3,3), first Piola-Kirchhoff stress.
        tangent_fn : callable
            F (M,3,3) -> dP/dF (M,9,9), material tangent.
        f_ext : torch.Tensor, shape (N, 3)
            External force vector.
        u_bc : torch.Tensor, shape (N, 3)
            Prescribed displacement at Dirichlet nodes.
        bc_mask : torch.Tensor, shape (N,)
            Boolean mask; True for Dirichlet-constrained nodes.
        u_init : torch.Tensor, shape (N, 3), optional
            Initial guess for displacement.
        max_iter : int
            Maximum Newton iterations.
        tol : float
            Convergence tolerance on residual norm.

        Returns
        -------
        u : torch.Tensor, shape (N, 3)
            Converged nodal displacement field.
        """
        if u_init is not None:
            u = u_init.clone()
        else:
            u = torch.zeros(self.n_nodes, 3, device=self.device, dtype=self.dtype)

        # Enforce Dirichlet BCs on initial guess
        u[bc_mask] = u_bc[bc_mask]

        # Build free DOF mask (3N,) from node mask (N,)
        bc_mask_dof = bc_mask.unsqueeze(1).expand(-1, 3).reshape(-1)  # (3N,)
        free_dof = ~bc_mask_dof

        def residual_vector(u_cur: torch.Tensor) -> torch.Tensor:
            f_int = self.internal_forces(u_cur, stress_fn)
            R_cur = (f_int - f_ext).reshape(-1)
            R_cur[bc_mask_dof] = 0.0
            return R_cur

        for it in range(max_iter):
            # Residual: R = f_int(u) - f_ext
            R = residual_vector(u)  # (3N,)

            res_norm = torch.norm(R[free_dof]).item()
            if it == 0:
                res_norm0 = max(res_norm, 1e-30)
            rel_norm = res_norm / res_norm0

            if res_norm < tol or rel_norm < tol:
                print(f"  [Newton] converged at iter {it}: |R| = {res_norm:.2e}")
                break

            # Tangent stiffness
            K = self.tangent_stiffness(u, tangent_fn)  # (3N, 3N)

            # Apply Dirichlet BCs to tangent system: K[bc,:] = I[bc,:], R[bc] = 0
            K[bc_mask_dof, :] = 0.0
            K[:, bc_mask_dof] = 0.0
            K[bc_mask_dof, bc_mask_dof] = 1.0
            R[bc_mask_dof] = 0.0

            # Solve K @ du = -R
            du = torch.linalg.solve(K, -R)  # (3N,)

            # Backtracking line search improves robustness for the atlas-coupled
            # elastoplastic solves where a full Newton step can overshoot.
            du_3d = du.reshape(-1, 3)
            alpha = 1.0
            accepted = False
            u_trial = u + du_3d
            u_trial[bc_mask] = u_bc[bc_mask]

            for _ in range(8):
                R_trial = residual_vector(u_trial)
                res_trial = torch.norm(R_trial[free_dof]).item()
                if res_trial < res_norm:
                    accepted = True
                    break
                alpha *= 0.5
                u_trial = u + alpha * du_3d
                u_trial[bc_mask] = u_bc[bc_mask]

            u = u_trial if accepted else (u + du_3d)
            u[bc_mask] = u_bc[bc_mask]

            if it == max_iter - 1:
                print(f"  [Newton] WARNING: did not converge after {max_iter} iters, |R| = {res_norm:.2e}")

        return u

    # ------------------------------------------------------------------
    # Built-in Neo-Hookean material
    # ------------------------------------------------------------------
    @staticmethod
    def neo_hookean_stress(
        F: torch.Tensor, mu: "torch.Tensor | float", K: "torch.Tensor | float"
    ) -> torch.Tensor:
        """Compressible Neo-Hookean first Piola-Kirchhoff stress.

        P = mu * (F - F^{-T}) + K * ln(J) * F^{-T}

        Parameters
        ----------
        F : torch.Tensor, shape (M, 3, 3)
            Deformation gradient.
        mu : torch.Tensor or float
            Shear modulus (scalar tensor for differentiability).
        K : torch.Tensor or float
            Bulk modulus (scalar tensor for differentiability).

        Returns
        -------
        P : torch.Tensor, shape (M, 3, 3)
            First Piola-Kirchhoff stress.
        """
        J = torch.det(F)
        J_safe = torch.clamp(J, min=1e-8)
        F_inv_T = torch.linalg.inv(F).transpose(-2, -1)
        log_J = torch.log(J_safe).unsqueeze(-1).unsqueeze(-1)
        return mu * (F - F_inv_T) + K * log_J * F_inv_T

    @staticmethod
    def neo_hookean_tangent(
        F: torch.Tensor, mu: "torch.Tensor | float", K: "torch.Tensor | float"
    ) -> torch.Tensor:
        """Material tangent dP/dF for compressible Neo-Hookean, returned as (M, 9, 9).

        Uses row-major flattening: component (i,J) -> index 3*i + J.

        The tangent is:
            dP_{iJ}/dF_{kL} = mu * delta_{ik} delta_{JL}
                             + (mu - K*ln(J)) * A_{Li} A_{Jk}
                             + K * A_{Ji} A_{Lk}

        where A = F^{-1}.

        Derivation:
            d(F^{-T}_{iJ})/dF_{kL} = -A_{Li} A_{Jk}   (standard identity)
            d(ln J)/dF_{kL} = A_{Lk}

        Parameters
        ----------
        F : torch.Tensor, shape (M, 3, 3)
        mu : torch.Tensor or float
            Shear modulus (scalar tensor for differentiability).
        K : torch.Tensor or float
            Bulk modulus (scalar tensor for differentiability).

        Returns
        -------
        C : torch.Tensor, shape (M, 9, 9)
        """
        M = F.shape[0]
        dev = F.device
        dt = F.dtype

        A = torch.linalg.inv(F)  # (M, 3, 3), A = F^{-1}
        J = torch.det(F)
        J_safe = torch.clamp(J, min=1e-8)
        log_J = torch.log(J_safe)  # (M,)

        C = torch.zeros(M, 9, 9, device=dev, dtype=dt)

        # Fill using the formula:
        # C[(3i+J), (3k+L)] = mu * d_{ik} d_{JL}
        #                    + (mu - K*lnJ) * A_{Li} * A_{Jk}
        #                    + K * A_{Ji} * A_{Lk}
        #
        # We vectorize over M (elements) and loop over the 9x9 index pairs.
        coeff = mu - K * log_J  # (M,)

        for i in range(3):
            for J_idx in range(3):
                row = 3 * i + J_idx
                for k in range(3):
                    for L in range(3):
                        col = 3 * k + L
                        val = torch.zeros(M, device=dev, dtype=dt)
                        if i == k and J_idx == L:
                            val = val + mu
                        val = val + coeff * A[:, L, i] * A[:, J_idx, k]
                        val = val + K * A[:, J_idx, i] * A[:, L, k]
                        C[:, row, col] = val

        return C

    # ------------------------------------------------------------------
    # Convenience: make material closures
    # ------------------------------------------------------------------
    def make_neo_hookean(
        self, mu: "torch.Tensor | float", K: "torch.Tensor | float"
    ) -> Tuple[
        Callable[[torch.Tensor], torch.Tensor],
        Callable[[torch.Tensor], torch.Tensor],
    ]:
        """Return (stress_fn, tangent_fn) closures for Neo-Hookean material.

        Parameters
        ----------
        mu : torch.Tensor or float
            Shear modulus. Use a tensor with requires_grad=True for differentiability.
        K : torch.Tensor or float
            Bulk modulus. Use a tensor with requires_grad=True for differentiability.

        Returns
        -------
        stress_fn : callable, F -> P
        tangent_fn : callable, F -> dP/dF (M,9,9)
        """
        def stress_fn(F: torch.Tensor) -> torch.Tensor:
            return self.neo_hookean_stress(F, mu, K)

        def tangent_fn(F: torch.Tensor) -> torch.Tensor:
            return self.neo_hookean_tangent(F, mu, K)

        return stress_fn, tangent_fn


# ======================================================================
# Self-test
# ======================================================================
if __name__ == "__main__":
    import sys

    torch.set_default_dtype(torch.float64)
    device = "cpu"

    print("=" * 70)
    print("ChartVectorFEMSolver self-test")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Test 1: Patch test — linear displacement field
    # ------------------------------------------------------------------
    print("\n--- Test 1: Patch test (linear displacement) ---")

    n = 4
    solver = ChartVectorFEMSolver(n_cells=n, support_r=1.0, device=device)

    # Linear displacement: u(x) = A @ x + b
    A_deform = torch.tensor(
        [[0.01, 0.02, -0.01],
         [0.005, -0.01, 0.015],
         [-0.02, 0.01, 0.005]],
        device=device, dtype=torch.float64,
    )
    b_deform = torch.tensor([0.1, -0.05, 0.03], device=device, dtype=torch.float64)

    u_exact = solver.nodes @ A_deform.T + b_deform.unsqueeze(0)  # (N, 3)

    # For linear u, F = I + A is constant everywhere -> P should be constant
    # -> div(P) = 0, so f_ext = 0 at free nodes
    mu_test, K_test = 1.0, 10.0
    stress_fn, tangent_fn = solver.make_neo_hookean(mu_test, K_test)

    f_int = solver.internal_forces(u_exact, stress_fn)

    # Interior nodes should have zero net force (since P is constant, div P = 0)
    interior_mask = ~solver.boundary_mask
    f_int_interior = f_int[interior_mask]
    max_interior_force = torch.max(torch.abs(f_int_interior)).item()
    print(f"  Max |f_int| at interior nodes: {max_interior_force:.2e}")
    assert max_interior_force < 1e-10, f"Patch test FAILED: {max_interior_force}"
    print("  PASSED")

    # ------------------------------------------------------------------
    # Test 2: MMS convergence test
    # ------------------------------------------------------------------
    print("\n--- Test 2: MMS convergence (Neo-Hookean) ---")

    # Manufactured displacement: u(x) = eps * sin(pi*x) * sin(pi*y) * sin(pi*z) * [1,1,1]
    # Small eps so we stay in near-linear regime for Newton convergence.
    eps_mms = 0.005
    pi = math.pi
    mu_mms, K_mms = 1.0, 10.0

    def u_manufactured(x: torch.Tensor) -> torch.Tensor:
        """Manufactured displacement field, shape (N, 3)."""
        s = eps_mms * (
            torch.sin(pi * x[:, 0]) *
            torch.sin(pi * x[:, 1]) *
            torch.sin(pi * x[:, 2])
        )
        return torch.stack([s, s, s], dim=1)

    def body_force_mms(
        x: torch.Tensor, mu: float, K: float, eps: float
    ) -> torch.Tensor:
        """Compute body force f = -div(P(I + grad(u_mms))) via automatic differentiation.

        This computes the body force numerically using autograd, which is exact
        for the manufactured solution.
        """
        x_ad = x.clone().requires_grad_(True)
        u = u_manufactured(x_ad)
        # Compute grad(u): (N, 3, 3)
        grads = []
        for i in range(3):
            g = torch.autograd.grad(
                u[:, i], x_ad,
                grad_outputs=torch.ones(x_ad.shape[0], device=x_ad.device, dtype=x_ad.dtype),
                create_graph=True, retain_graph=True,
            )[0]
            grads.append(g)
        grad_u = torch.stack(grads, dim=1)  # (N, 3, 3)
        F_val = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0) + grad_u

        # P(F)
        P = ChartVectorFEMSolver.neo_hookean_stress(F_val, mu, K)

        # div(P): (N, 3) — divergence of each row of P
        div_P = torch.zeros_like(u)
        for i in range(3):
            for J in range(3):
                dP_iJ_dxJ = torch.autograd.grad(
                    P[:, i, J], x_ad,
                    grad_outputs=torch.ones(x_ad.shape[0], device=x_ad.device, dtype=x_ad.dtype),
                    create_graph=False, retain_graph=True,
                )[0][:, J]
                div_P[:, i] = div_P[:, i] + dP_iJ_dxJ

        # f = -div(P) (static equilibrium: div(P) + f = 0)
        return -div_P.detach()

    resolutions = [4, 8, 16]
    errors = []

    for nc in resolutions:
        solver_mms = ChartVectorFEMSolver(
            n_cells=nc, support_r=1.0, device=device
        )
        stress_fn_mms, tangent_fn_mms = solver_mms.make_neo_hookean(mu_mms, K_mms)

        # Compute body force at nodes
        f_body = body_force_mms(solver_mms.nodes, mu_mms, K_mms, eps_mms)

        # Dirichlet BC: prescribe exact displacement on boundary
        u_bc = u_manufactured(solver_mms.nodes)
        bc_mask = solver_mms.boundary_mask

        # External force: assemble from body force using FEM mass-like lumping
        # For P1 tets, f_ext[a] = sum_elem vol_e/4 * f(x_centroid) for node a in elem
        centroids = solver_mms.nodes[solver_mms.elements].mean(dim=1)
        f_cent = body_force_mms(centroids, mu_mms, K_mms, eps_mms)  # (M, 3)
        f_ext = torch.zeros(solver_mms.n_nodes, 3, device=device, dtype=torch.float64)
        f_ext_elem = (solver_mms.vol[:, None, None] / 4.0) * f_cent.unsqueeze(1).expand(-1, 4, -1)
        f_ext_flat = f_ext_elem.reshape(-1, 3)
        idx_flat = solver_mms.elements.reshape(-1).unsqueeze(-1).expand(-1, 3)
        f_ext.scatter_add_(0, idx_flat, f_ext_flat)

        # Solve
        u_sol = solver_mms.solve_nonlinear(
            stress_fn_mms, tangent_fn_mms, f_ext, u_bc, bc_mask,
            max_iter=30, tol=1e-12,
        )

        # Error
        u_ref = u_manufactured(solver_mms.nodes)
        err = torch.norm(u_sol - u_ref) / max(torch.norm(u_ref).item(), 1e-30)
        errors.append(err.item())
        print(f"  n_cells={nc:3d}: L2 error = {err:.6e}")

    # Check convergence rates
    print("\n  Convergence rates:")
    passed = True
    for i in range(1, len(errors)):
        if errors[i - 1] > 1e-15 and errors[i] > 1e-15:
            rate = math.log(errors[i - 1] / errors[i]) / math.log(
                resolutions[i] / resolutions[i - 1]
            )
            print(f"  h/{resolutions[i]//resolutions[i-1]}: rate = {rate:.2f}")
            if rate < 1.5:
                print(f"  WARNING: expected O(h^2), got rate {rate:.2f}")
                passed = False

    if passed:
        print("\n  MMS convergence test PASSED")
    else:
        print("\n  MMS convergence test: rates below expected O(h^2)")

    # ------------------------------------------------------------------
    # Test 3: Autograd through material parameters
    # ------------------------------------------------------------------
    print("\n--- Test 3: Autograd through material parameters ---")

    nc_ad = 4
    solver_ad = ChartVectorFEMSolver(n_cells=nc_ad, support_r=1.0, device=device)

    mu_ad = torch.tensor(1.0, device=device, dtype=torch.float64, requires_grad=True)
    K_ad = torch.tensor(10.0, device=device, dtype=torch.float64, requires_grad=True)

    # Use autograd-based stress (not the closed-form tangent) so that the
    # forward solve is fully differentiable w.r.t. mu_ad, K_ad.
    # Strategy: solve a single Newton step from zero displacement with a
    # prescribed body force, using torch.linalg.solve for differentiability.

    # Simple body force: uniform downward
    f_ext_ad = torch.zeros(solver_ad.n_nodes, 3, device=device, dtype=torch.float64)
    f_ext_ad[:, 2] = -0.01  # small load in z

    # Dirichlet BC: fix boundary
    bc_mask_ad = solver_ad.boundary_mask
    u_bc_ad = torch.zeros(solver_ad.n_nodes, 3, device=device, dtype=torch.float64)

    # One Newton step from u=0 (linear solve, fully differentiable):
    #   K_tan(0) @ u = f_ext  with BCs
    u_zero = torch.zeros(solver_ad.n_nodes, 3, device=device, dtype=torch.float64)
    u_zero[bc_mask_ad] = u_bc_ad[bc_mask_ad]

    def stress_fn_ad(F: torch.Tensor) -> torch.Tensor:
        return ChartVectorFEMSolver.neo_hookean_stress(F, mu_ad, K_ad)

    def tangent_fn_ad(F: torch.Tensor) -> torch.Tensor:
        return ChartVectorFEMSolver.neo_hookean_tangent(F, mu_ad, K_ad)

    # Build tangent at u=0
    K_mat = solver_ad.tangent_stiffness(u_zero, tangent_fn_ad)  # (3N, 3N)
    f_int_0 = solver_ad.internal_forces(u_zero, stress_fn_ad)  # (N, 3)

    rhs = (f_ext_ad - f_int_0).reshape(-1)  # (3N,)

    # Apply Dirichlet BCs
    bc_mask_dof_ad = bc_mask_ad.unsqueeze(1).expand(-1, 3).reshape(-1)
    K_mat[bc_mask_dof_ad, :] = 0.0
    K_mat[:, bc_mask_dof_ad] = 0.0
    K_mat[bc_mask_dof_ad, bc_mask_dof_ad] = 1.0
    rhs[bc_mask_dof_ad] = 0.0

    u_sol_ad = torch.linalg.solve(K_mat, rhs)  # (3N,)

    # Pick a scalar objective: displacement at a specific interior node
    interior_indices = torch.where(~bc_mask_ad)[0]
    probe_node = interior_indices[len(interior_indices) // 2].item()
    objective = u_sol_ad[3 * probe_node + 2]  # z-displacement at probe node

    objective.backward()

    grad_mu = mu_ad.grad
    grad_K = K_ad.grad

    print(f"  Objective (u_z at node {probe_node}): {objective.item():.6e}")
    print(f"  d(objective)/d(mu): {grad_mu.item():.6e}")
    print(f"  d(objective)/d(K):  {grad_K.item():.6e}")

    test3_passed = True
    if grad_mu is None or not torch.isfinite(grad_mu) or grad_mu.item() == 0.0:
        print("  FAILED: gradient w.r.t. mu is zero or non-finite")
        test3_passed = False
    if grad_K is None or not torch.isfinite(grad_K) or grad_K.item() == 0.0:
        print("  FAILED: gradient w.r.t. K is zero or non-finite")
        test3_passed = False

    if test3_passed:
        print("  PASSED")
    else:
        print("  FAILED")

    print("\n" + "=" * 70)
    print("Self-test complete.")
    print("=" * 70)
