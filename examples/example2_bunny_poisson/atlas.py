"""The frozen neural atlas of the Stanford bunny.

Each chart i has a seed point s_i, an orthonormal frame (t1, t2, n), a support
radius r_i and two small networks:

* a decoder  xi -> x = s_i + [t1 t2 n] xi + 0.2 tanh(a) r_i NN(xi / r_i)
  (the chart map; trained with a one-sided Jacobian barrier it is essentially
  rigid: det J = 1.000 +- 0.02, sub-millimetre displacement);
* a mask network whose logits, softmax-normalised over the charts, give the
  partition of unity that blends the chart solutions.

The atlas also carries its sample points and their chart membership, which set
the chart neighbourhoods, the multiplicative-Schwarz colouring and the interface
points. ``assets/atlas_bunny_{8,12}chart.npz`` hold the arrays and
``assets/atlas_bunny_{8,12}chart.pt`` the network weights.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int):
        super().__init__()
        self.hidden = torch.nn.ModuleList(
            [torch.nn.Linear(in_dim, width)] + [torch.nn.Linear(width, width) for _ in range(depth - 1)])
        self.out = torch.nn.Linear(width, out_dim)
        for layer in list(self.hidden) + [self.out]:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden:
            x = torch.tanh(layer(x))
        return self.out(x)


class ChartDecoder(torch.nn.Module):
    """x = s + [t1 t2 n] xi + 0.2 tanh(a) r NN(xi / r)."""

    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        self.net = MLP(3, 3, width, depth)
        self.raw_scale = torch.nn.Parameter(torch.tensor(-1.8))

    def forward(self, xi, seed, t1, t2, n, chart_scale):
        base = seed + xi[:, 0:1] * t1 + xi[:, 1:2] * t2 + xi[:, 2:3] * n
        r = torch.clamp(chart_scale, min=1e-6)
        return base + 0.20 * torch.tanh(self.raw_scale) * r * self.net(xi / r)


class MaskNet(torch.nn.Module):
    """Partition-of-unity logit of one chart."""

    def __init__(self, width: int = 48, depth: int = 3):
        super().__init__()
        self.net = MLP(3, 1, width, depth)

    def forward(self, xi, chart_scale):
        return self.net(xi / torch.clamp(chart_scale, min=1e-6)).squeeze(-1)


@dataclass
class Atlas:
    seeds: torch.Tensor          # (K, 3)
    t1: torch.Tensor             # (K, 3)
    t2: torch.Tensor             # (K, 3)
    nvec: torch.Tensor           # (K, 3)
    radii: torch.Tensor          # (K,)
    points: torch.Tensor         # (P, 3) atlas sample points
    membership: np.ndarray       # (P, K) bool
    color_groups: List[List[int]]
    decoders: List[ChartDecoder]
    masks: List[MaskNet]

    @property
    def n_charts(self) -> int:
        return int(self.seeds.shape[0])

    def local_coords(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """Frame coordinates of x in chart i: [t1 t2 n]^T (x - s_i)."""
        d = x - self.seeds[i]
        return torch.stack([d @ self.t1[i], d @ self.t2[i], d @ self.nvec[i]], dim=1)

    def from_local(self, xi: torch.Tensor, i: int) -> torch.Tensor:
        return self.seeds[i] + xi[:, 0:1] * self.t1[i] + xi[:, 1:2] * self.t2[i] + xi[:, 2:3] * self.nvec[i]

    def decode(self, xi: torch.Tensor, i: int) -> torch.Tensor:
        return self.decoders[i](xi, self.seeds[i], self.t1[i], self.t2[i], self.nvec[i], self.radii[i])

    def mask_logit(self, x: torch.Tensor, i: int) -> torch.Tensor:
        return self.masks[i](self.local_coords(x, i), self.radii[i])

    @torch.no_grad()
    def pou_weights(self, x: torch.Tensor) -> torch.Tensor:
        """(N, K) softmax of the mask logits: the partition of unity."""
        return torch.softmax(torch.stack([self.mask_logit(x, i) for i in range(self.n_charts)], dim=1), dim=1)

    def member_index(self, i: int) -> np.ndarray:
        return np.where(self.membership[:, i])[0]

    def neighbours(self) -> Tuple[List[List[int]], Dict[Tuple[int, int], np.ndarray]]:
        """Charts sharing atlas points, and the shared point indices per pair (i < j)."""
        k = self.n_charts
        nbr: List[List[int]] = [[] for _ in range(k)]
        shared: Dict[Tuple[int, int], np.ndarray] = {}
        for i in range(k):
            for j in range(i + 1, k):
                s = np.where(self.membership[:, i] & self.membership[:, j])[0]
                if len(s):
                    shared[(i, j)] = s
                    nbr[i].append(j)
                    nbr[j].append(i)
        return nbr, shared


def load_atlas(n_charts: int = 12, device="cpu", dtype=torch.float64, asset_dir: str = ASSET_DIR) -> Atlas:
    stem = os.path.join(asset_dir, f"atlas_bunny_{n_charts}chart")
    a = np.load(stem + ".npz")
    w = torch.load(stem + ".pt", map_location="cpu", weights_only=True)
    t = lambda k: torch.as_tensor(a[k], dtype=dtype, device=device)  # noqa: E731
    decoders, masks = [], []
    for sd in w["decoders"]:
        d = ChartDecoder(**w["decoder_kwargs"]).to(device=device, dtype=dtype)
        d.load_state_dict(sd)
        decoders.append(d.eval().requires_grad_(False))
    for sd in w["masks"]:
        m = MaskNet(**w["mask_kwargs"]).to(device=device, dtype=dtype)
        m.load_state_dict(sd)
        masks.append(m.eval().requires_grad_(False))
    flat, offs = a["color_groups_flat"], a["color_groups_offsets"]
    groups = [[int(c) for c in flat[offs[g]:offs[g + 1]]] for g in range(len(offs) - 1)]
    return Atlas(seeds=t("seed_points"), t1=t("frame_t1"), t2=t("frame_t2"), nvec=t("frame_n"),
                 radii=t("support_radii"), points=t("points"), membership=a["membership"].astype(bool),
                 color_groups=groups, decoders=decoders, masks=masks)
