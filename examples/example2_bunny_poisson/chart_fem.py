"""P1 finite elements on one chart of the neural atlas.

The chart's reference cube [-R, R]^3 (R = 1.5 r) carries a structured hex grid
split into 6 tetrahedra per hex. Its elements are mapped to physical space by the
chart decoder phi, and the Poisson problem is pulled back:

    -div_xi( A grad_xi u ) = j f(phi(xi)),   A = j J^-1 J^-T,  j = |det J|.

Only elements whose centroid lies inside the body are kept. A mesh-boundary face
that lies on the reference cube is an *interface* (artificial) face and receives
Schwarz data from neighbouring charts; every other boundary face was cut by the
geometry and approximates the surface. Its nodes are *snapped* onto the surface
(boundary-fitted P1, second order); slivers whose four vertices all end up on the
surface are dropped, and any other element that would invert or flatten halves
the moves of its nodes until it is valid.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import torch

from mapped_sphere.geometry import ChartGeometry, jacobian
from mapped_sphere.piola import diffusion_pullback

# Freudenthal split of the unit cube; corners indexed dz*4 + dy*2 + dx.
_TETS = np.array([[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]])
_CORNERS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
_NEIGHBOUR_CELLS = np.array(sorted(([a, b, c] for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
                                    if (a, b, c) != (0, 0, 0)), key=lambda o: sum(map(abs, o))))


class ChartFEM:
    def __init__(self, atlas, i: int, geometry, n_cells: int, sdf_threshold: float = -0.005,
                 mesh_extent: float = 1.5, snap_min_volume: float = 0.05, dtype=torch.float64):
        self.atlas, self.i, self.geometry = atlas, i, geometry
        self.dtype = dtype
        self.n_cells = n_cells
        self.r = float(atlas.radii[i]) * mesh_extent
        self.h = 2.0 * self.r / n_cells
        self.snap_min_volume = snap_min_volume
        self.u: Optional[np.ndarray] = None
        self._lu = self._lu_key = None
        self._build_mesh(sdf_threshold)

    # ------------------------------------------------------------------ geometry
    def decode(self, xi: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.atlas.decode(torch.as_tensor(xi, dtype=self.dtype), self.i).numpy()

    def pull_back(self, x: np.ndarray, xi0: np.ndarray, iters: int = 8) -> np.ndarray:
        """Newton-invert the chart decoder: the xi with phi(xi) = x."""
        xi = torch.as_tensor(xi0, dtype=self.dtype)
        xt = torch.as_tensor(x, dtype=self.dtype)
        for _ in range(iters):
            xr = xi.clone().requires_grad_(True)
            y = self.atlas.decode(xr, self.i)
            jac = jacobian(y, xr, create_graph=False)
            res = (y - xt).detach()
            xi = (xr - torch.linalg.solve(jac, res.unsqueeze(-1)).squeeze(-1)).detach()
            if float(res.norm(dim=1).max()) < 1e-13:
                break
        return xi.numpy()

    @staticmethod
    def signed_volumes(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
        xe = nodes[elements]
        return np.linalg.det(np.transpose(xe[:, 1:] - xe[:, :1], (0, 2, 1))) / 6.0

    def _build_mesh(self, sdf_threshold: float) -> None:
        nc, npa = self.n_cells, self.n_cells + 1
        lin = np.linspace(-self.r, self.r, npa)
        grid = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), axis=-1).reshape(-1, 3)
        hexes = np.stack(np.meshgrid(*(np.arange(nc),) * 3, indexing="ij"), axis=-1).reshape(-1, 3)
        corner = (hexes[:, None, :] + _CORNERS[None]) @ np.array([npa * npa, npa, 1])
        tets = corner[:, _TETS].reshape(-1, 4)
        hex_of_tet = np.repeat(np.arange(len(hexes)), 6)

        keep = self.geometry.sdf(self.decode(grid[tets].mean(axis=1))) < sdf_threshold
        tets, hex_of_tet = tets[keep], hex_of_tet[keep]
        used = np.unique(tets)
        remap = np.full(len(grid), -1)
        remap[used] = np.arange(len(used))
        self.nodes, self.elements, self.hex_of_tet = grid[used], remap[tets], hex_of_tet
        self.n_nodes = len(self.nodes)
        self._index_cells()
        self._classify_boundary()
        if len(self.phys_nodes):
            self._snap()

    def _index_cells(self) -> None:
        """(n_hex, 6) table of the tets in each hex cell, padded with -1."""
        n_hex = self.n_cells ** 3
        table = np.full((n_hex, 6), -1, dtype=np.int64)
        order = np.argsort(self.hex_of_tet, kind="stable")
        sh = self.hex_of_tet[order]
        table[sh, np.arange(len(sh)) - np.searchsorted(sh, sh)] = order
        self.cell_tets = table

    def _classify_boundary(self) -> None:
        faces = np.sort(self.elements[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3), axis=1)
        uniq, counts = np.unique(faces, axis=0, return_counts=True)
        bfaces = uniq[counts == 1]
        fx = self.nodes[bfaces]
        on_cube = np.zeros(len(bfaces), dtype=bool)
        for k in range(3):
            for sgn in (-1.0, 1.0):
                on_cube |= np.all(np.abs(fx[:, :, k] - sgn * self.r) < 1e-6 * self.h, axis=1)
        bnodes = np.unique(bfaces)
        is_phys = np.isin(bnodes, bfaces[~on_cube])
        self.phys_nodes, self.art_nodes = bnodes[is_phys], bnodes[~is_phys]

    def _snap(self) -> None:
        idx = self.phys_nodes
        xi0 = self.nodes[idx].copy()
        d = self.pull_back(self.geometry.closest_point(self.decode(xi0)), xi0) - xi0
        el = self.elements
        vol0 = self.signed_volumes(self.nodes, el)
        where = np.full(self.n_nodes, -1)
        where[idx] = np.arange(len(idx))
        on_surface = (where[el] >= 0).all(axis=1)
        t = np.ones(len(idx))
        nodes = self.nodes.copy()
        for _ in range(24):
            nodes[idx] = xi0 + t[:, None] * d
            bad = self.signed_volumes(nodes, el) * np.sign(vol0) < self.snap_min_volume * np.abs(vol0)
            blocking = bad & ~on_surface
            if not blocking.any():
                break
            k = where[np.unique(el[blocking])]
            k = k[k >= 0]
            t[k] = np.where(t[k] > 1.0 / 16.0, 0.5 * t[k], 0.0)
        else:
            raise RuntimeError(f"chart {self.i}: snapping did not reach a valid mesh")
        self.nodes, self.snap_fraction = nodes, t
        drop = bad & on_surface
        if drop.any():
            self.elements, self.hex_of_tet = self.elements[~drop], self.hex_of_tet[~drop]
            self._index_cells()

    # ------------------------------------------------------------------ assembly
    def assemble(self) -> None:
        cent = self.nodes[self.elements].mean(axis=1)
        xi = torch.as_tensor(cent, dtype=self.dtype).requires_grad_(True)
        x = self.atlas.decode(xi, self.i)
        geometry = ChartGeometry(jacobian(x, xi, create_graph=False))
        # Use the exact pullback; invalid maps raise instead of silently clipping
        # singular values or the determinant and changing the differential operator.
        A = diffusion_pullback(geometry).detach().numpy()
        det, x = geometry.volume.detach().numpy(), x.detach().numpy()

        xe = self.nodes[self.elements]
        B = np.transpose(xe[:, 1:] - xe[:, :1], (0, 2, 1))
        vol = np.abs(np.linalg.det(B)) / 6.0
        dN = np.zeros((len(self.elements), 4, 3))
        dN[:, 1:] = np.linalg.inv(B)
        dN[:, 0] = -dN[:, 1:].sum(axis=1)
        Ke = vol[:, None, None] * np.einsum("eai,ebi->eab", np.einsum("eij,eaj->eai", A, dN), dN)
        el = self.elements
        self.K = scipy.sparse.coo_matrix((Ke.ravel(), (np.repeat(el, 4, axis=1).ravel(), np.tile(el, (1, 4)).ravel())),
                                         shape=(self.n_nodes, self.n_nodes)).tocsr()
        from geometry import forcing
        self.F = np.zeros(self.n_nodes)
        np.add.at(self.F, el, (0.25 * vol * det * forcing(x))[:, None])

    def solve(self, dirichlet: Dict[int, float]) -> np.ndarray:
        """Solve with the given Dirichlet values; the free-free block is factorised once."""
        idx = np.fromiter(dirichlet.keys(), dtype=np.int64, count=len(dirichlet))
        val = np.fromiter(dirichlet.values(), dtype=float, count=len(dirichlet))
        order = np.argsort(idx)
        idx, val = idx[order], val[order]
        key = hash(idx.tobytes())
        if self._lu_key != key:
            free = np.setdiff1d(np.arange(self.n_nodes), idx)
            Kf = self.K[free]
            self._free, self._Kfb = free, Kf[:, idx].tocsr()
            self._lu = scipy.sparse.linalg.splu(Kf[:, free].tocsc(), permc_spec="MMD_AT_PLUS_A")
            self._lu_key = key
        u = np.empty(self.n_nodes)
        u[idx] = val
        u[self._free] = self._lu.solve(self.F[self._free] - self._Kfb @ val)
        self.u = u
        return u

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, xi: np.ndarray) -> np.ndarray:
        """P1 interpolant at reference points. A point in no element takes the linear
        extension of the best-fitting nearby element if it is within half an element
        of it, and the nearest node's value otherwise."""
        xi = np.atleast_2d(xi)
        nc = self.n_cells
        cell = np.clip(np.floor((xi + self.r) / self.h).astype(np.int64), 0, nc - 1)
        out = np.full(len(xi), np.nan)
        best_score, best_val = np.full(len(xi), -np.inf), np.zeros(len(xi))

        def try_cells(q, c):
            cand = self.cell_tets[c[:, 0] * nc * nc + c[:, 1] * nc + c[:, 2]]
            for t in range(6):
                sel = np.isnan(out[q]) & (cand[:, t] >= 0)
                if not sel.any():
                    continue
                qq, tid = q[sel], cand[sel, t]
                nid = self.elements[tid]
                xe = self.nodes[nid]
                T = np.transpose(xe[:, 1:] - xe[:, :1], (0, 2, 1))
                l123 = np.linalg.solve(T, (xi[qq] - xe[:, 0])[..., None])[..., 0]
                lam = np.column_stack([1.0 - l123.sum(axis=1), l123])
                score, val = lam.min(axis=1), np.sum(lam * self.u[nid], axis=1)
                inside = score >= -1e-10
                out[qq[inside]] = val[inside]
                better = ~inside & (score > best_score[qq])
                best_score[qq[better]], best_val[qq[better]] = score[better], val[better]

        try_cells(np.arange(len(xi)), cell)
        pending = np.where(np.isnan(out))[0]
        for off in _NEIGHBOUR_CELLS:
            if not len(pending):
                break
            c = cell[pending] + off
            ok = np.all((c >= 0) & (c < nc), axis=1)
            if ok.any():
                try_cells(pending[ok], c[ok])
            pending = pending[np.isnan(out[pending])]
        near = pending[best_score[pending] > -0.5]
        out[near] = best_val[near]
        miss = np.isnan(out)
        if miss.any():
            from scipy.spatial import cKDTree
            out[miss] = self.u[cKDTree(self.nodes).query(xi[miss])[1]]
        return out
