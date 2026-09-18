"""Stanford bunny geometry: download, hole closing, exact inside test and distances.

The domain is the region enclosed by ``bun_zipper.ply`` from the Stanford 3D
Scanning Repository. The scan is open at the base (five boundary loops), so each
loop is closed with a fan of triangles oriented consistently with the rest of the
surface. On the resulting closed surface:

* inside / outside is decided by ray-casting parity (VTK ``vtkSelectEnclosedPoints``),
  which is exact up to degenerate rays;
* distances and closest points come from VTK's point-to-triangle queries.

Coordinates are normalised as ``(x - center) / scale`` with ``center`` the centre of
the bounding box and ``scale`` its largest extent, so the bunny spans about
[-0.5, 0.5] and 1 normalised unit is 155.7 mm.
"""
from __future__ import annotations

import hashlib
import os
import struct
import tarfile
import urllib.request
from typing import Optional, Tuple

import numpy as np

BUNNY_URL = "https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
BUNNY_SHA256 = "b1acc63bece78444aa2e15bdcc72371a201279b98c6f5d4b74c993d02f0566fe"  # bun_zipper.ply
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# --------------------------------------------------------------------------- download
def fetch_bunny(ply: Optional[str] = None, data_dir: str = DATA_DIR) -> str:
    """Path to ``bun_zipper.ply``: ``ply`` if given, else a cached or downloaded copy."""
    if ply:
        return ply
    path = os.path.join(data_dir, "bunny", "reconstruction", "bun_zipper.ply")
    if not os.path.isfile(path):
        os.makedirs(data_dir, exist_ok=True)
        archive = os.path.join(data_dir, "bunny.tar.gz")
        print(f"Downloading the Stanford bunny from {BUNNY_URL} ...")
        urllib.request.urlretrieve(BUNNY_URL, archive)
        with tarfile.open(archive, "r:gz") as tf:
            member = next(m for m in tf.getmembers() if m.name.endswith("reconstruction/bun_zipper.ply"))
            tf.extract(member, path=data_dir)
        path = os.path.join(data_dir, member.name)
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    if digest != BUNNY_SHA256:
        print(f"WARNING: {path} differs from the mesh the reported results were computed on "
              f"(sha256 {digest[:12]}..., expected {BUNNY_SHA256[:12]}...).")
    return path


# --------------------------------------------------------------------------- PLY
_PLY_NP = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2", "int": "i4", "uint": "u4",
           "float": "f4", "double": "f8", "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
           "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8"}
_PLY_STRUCT = {"char": ("b", 1), "uchar": ("B", 1), "short": ("h", 2), "ushort": ("H", 2),
               "int": ("i", 4), "uint": ("I", 4), "float": ("f", 4), "double": ("d", 8),
               "int8": ("b", 1), "uint8": ("B", 1), "int32": ("i", 4), "uint32": ("I", 4),
               "float32": ("f", 4), "float64": ("d", 8)}


def parse_ply(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(vertices (V, 3) float64, triangles (F, 3) int32)`` from an ASCII or binary PLY."""
    with open(path, "rb") as fh:
        header = []
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"Invalid PLY (missing end_header): {path}")
            s = line.decode("ascii", errors="replace").strip()
            header.append(s)
            if s == "end_header":
                break
        fmt = next((l.split()[1] for l in header if l.startswith("format")), "ascii")
        endian = "<" if "little" in fmt else (">" if "big" in fmt else "=")
        n_verts = n_faces = 0
        vert_props: list = []
        count_t, index_t, cur = "uchar", "int", None
        for l in header:
            if l.startswith("element vertex"):
                n_verts, cur = int(l.split()[-1]), "vertex"
            elif l.startswith("element face"):
                n_faces, cur = int(l.split()[-1]), "face"
            elif l.startswith("element"):
                cur = "other"
            elif l.startswith("property list") and cur == "face":
                count_t, index_t = l.split()[2], l.split()[3]
            elif l.startswith("property") and "list" not in l and cur == "vertex":
                vert_props.append((l.split()[2], l.split()[1]))
        names = [n for n, _ in vert_props]
        if fmt == "ascii":
            cols = [names.index(c) for c in ("x", "y", "z")]
            verts = np.array([[float(r[c]) for c in cols]
                              for r in (fh.readline().split() for _ in range(n_verts))])
        else:
            dt = np.dtype([(n, endian + _PLY_NP[t]) for n, t in vert_props])
            va = np.frombuffer(fh.read(n_verts * dt.itemsize), dtype=dt)
            verts = np.column_stack([va["x"], va["y"], va["z"]]).astype(np.float64)
        tris: list = []
        if fmt == "ascii":
            for _ in range(n_faces):
                row = fh.readline().split()
                idx = [int(v) for v in row[1:1 + int(row[0])]]
                tris.extend([idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1))
        else:
            c_char, c_sz = _PLY_STRUCT[count_t]
            i_char, i_sz = _PLY_STRUCT[index_t]
            c_struct = struct.Struct(endian + c_char)
            for _ in range(n_faces):
                cnt = c_struct.unpack(fh.read(c_sz))[0]
                idx = list(struct.unpack(endian + i_char * cnt, fh.read(cnt * i_sz)))
                tris.extend([idx[0], idx[k], idx[k + 1]] for k in range(1, cnt - 1))
    return verts, np.asarray(tris, dtype=np.int32).reshape(-1, 3)


def close_holes(verts: np.ndarray, tris: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Close every boundary loop with a fan around its centroid.

    Each fan traverses the loop's edges opposite to the triangles that already own
    them, so it faces the same way as the rest of the surface. (A fan facing into
    the body makes any normal-based inside test call part of the interior outside.)
    """
    edges = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=0)
    uniq, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    boundary = uniq[counts == 1]
    if not len(boundary):
        return verts, tris
    adj: dict = {}
    for a, b in boundary:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    unused = {tuple(sorted((int(a), int(b)))) for a, b in boundary}
    loops = []
    while unused:
        a, b = next(iter(unused))
        unused.discard((a, b))
        loop = [a, b]
        while True:
            nxt = [n for n in adj.get(loop[-1], []) if tuple(sorted((loop[-1], n))) in unused]
            if not nxt:
                break
            unused.discard(tuple(sorted((loop[-1], nxt[0]))))
            if nxt[0] == loop[0]:
                break
            loop.append(nxt[0])
        if len(loop) >= 3:
            loops.append(loop)
    owned = {(int(a), int(b)) for a, b in edges}
    new_v, new_t = list(verts), list(tris)
    for loop in loops:
        if 2 * sum((loop[i], loop[(i + 1) % len(loop)]) in owned for i in range(len(loop))) > len(loop):
            loop = loop[::-1]
        c = len(new_v)
        new_v.append(verts[loop].mean(axis=0))
        new_t.extend([loop[i], loop[(i + 1) % len(loop)], c] for i in range(len(loop)))
    return np.asarray(new_v, dtype=np.float64), np.asarray(new_t, dtype=np.int32)


def sample_surface(verts: np.ndarray, tris: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` points distributed uniformly by area over the triangles."""
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    face = rng.choice(len(tris), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    w = 1.0 - u - v
    return w[:, None] * v0[face] + u[:, None] * v1[face] + v[:, None] * v2[face]


# --------------------------------------------------------------------------- geometry
class BunnyGeometry:
    """Exact geometry of the (hole-closed) bunny in the normalised frame."""

    def __init__(self, ply_path: str):
        import pyvista as pv

        self._pv = pv
        verts, tris = parse_ply(ply_path)
        lo, hi = verts.min(0), verts.max(0)
        self.center = 0.5 * (lo + hi)
        self.scale = float(np.max(hi - lo))
        self.scale_mm = 1000.0 * self.scale
        verts, tris = close_holes(verts, tris)
        self.verts = (verts - self.center) / self.scale
        self.tris = tris
        self.surface = pv.PolyData(self.verts, np.hstack([np.full((len(tris), 1), 3), tris]).ravel())
        n_open = self.surface.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                                    manifold_edges=False, non_manifold_edges=False).n_cells
        if n_open:
            raise ValueError(f"surface still has {n_open} open edges after closing its holes")
        self.lo, self.hi = self.verts.min(0), self.verts.max(0)

    def inside(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1, 3)
        if not len(x):
            return np.zeros(0, dtype=bool)
        enc = self._pv.PolyData(x).select_enclosed_points(self.surface, tolerance=1e-9, check_surface=False)
        in_box = np.all((x >= self.lo) & (x <= self.hi), axis=1)
        return np.asarray(enc["SelectedPoints"]).astype(bool) & in_box

    def distance(self, x: np.ndarray) -> np.ndarray:
        c = self._pv.PolyData(np.asarray(x, dtype=float).reshape(-1, 3))
        c.compute_implicit_distance(self.surface, inplace=True)
        return np.abs(np.asarray(c["implicit_distance"]))

    def sdf(self, x):
        """Signed distance (negative inside); accepts and returns numpy or torch."""
        is_torch = hasattr(x, "detach")
        xn = x.detach().cpu().numpy() if is_torch else np.asarray(x, dtype=float)
        s = np.where(self.inside(xn), -1.0, 1.0) * self.distance(xn)
        if is_torch:
            import torch

            return torch.as_tensor(s, dtype=x.dtype, device=x.device)
        return s

    def closest_point(self, x: np.ndarray) -> np.ndarray:
        _, cp = self.surface.find_closest_cell(np.asarray(x, dtype=float).reshape(-1, 3),
                                               return_closest_point=True)
        return np.asarray(cp, dtype=float).reshape(-1, 3)

    def sample_interior(self, n: int, seed: int) -> np.ndarray:
        """``n`` points uniform in the interior (rejection from the bounding box)."""
        rng = np.random.default_rng(seed)
        out, got = [], 0
        while got < n:
            c = self.lo + (self.hi - self.lo) * rng.random((max(8 * n, 400000), 3))
            c = c[self.inside(c)]
            out.append(c)
            got += len(c)
        return np.concatenate(out)[:n]

    def sample_surface(self, n: int, seed: int) -> np.ndarray:
        return sample_surface(self.verts, self.tris, n, np.random.default_rng(seed))


# --------------------------------------------------------------------------- manufactured solution
def u_exact(x):
    """u = sin(pi x) sin(pi y) sin(pi z) in normalised coordinates (numpy or torch)."""
    if hasattr(x, "detach"):
        import torch

        return torch.sin(np.pi * x[..., 0:1]) * torch.sin(np.pi * x[..., 1:2]) * torch.sin(np.pi * x[..., 2:3])
    x = np.asarray(x)
    return np.sin(np.pi * x[..., 0]) * np.sin(np.pi * x[..., 1]) * np.sin(np.pi * x[..., 2])


def forcing(x):
    """f = -Laplacian(u) = 3 pi^2 u."""
    return 3.0 * np.pi ** 2 * u_exact(x)
