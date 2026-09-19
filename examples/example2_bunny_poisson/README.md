# Example 2 — Poisson on the Stanford Bunny (FEM and PINN on a neural atlas)

![Example 2 results](figures/bunny_example2.png)

The steady heat-conduction (Poisson) problem on the Stanford Bunny,

```
-Δu = f  in the bunny,     u = g  on its surface,
u = sin(πx) sin(πy) sin(πz),   f = 3π² u        (normalised coordinates)
```

solved on a **frozen neural atlas** of overlapping charts, either with **P1 finite
elements** or with **physics-informed neural networks**, the charts coupled by
**multiplicative Schwarz**. The exact solution is used only as the Dirichlet data `g`
on the surface and to score the result.

## Run

```bash
pip install -e ".[bunny]"          # adds scipy + pyvista (VTK) to the base requirements

python examples/example2_bunny_poisson/run_fem.py  --output-dir out_ex2_fem    # ~2 min
python examples/example2_bunny_poisson/run_pinn.py --output-dir out_ex2_pinn   # ~25 min on a laptop CPU
python examples/example2_bunny_poisson/plot_results.py --fem-dir out_ex2_fem --pinn-dir out_ex2_pinn
```

The first run downloads `bun_zipper.ply` (3 MB) from the
[Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/) into
`examples/example2_bunny_poisson/data/` and checks its SHA-256; pass `--ply PATH` to use
a local copy. `run_fem.py --n-cells 8 16 24 32 48 56` reproduces the full refinement
study (about 10 min for the 8-chart atlas); `--atlas 12` selects the 12-chart atlas.

## Expected results

Relative L² error on 50,000 points drawn uniformly in the interior (same points for
every run; CPU, float64, seed 42):

**FEM** (boundary-fitted P1, multiplicative Schwarz)

| n (cells / chart axis) | 8 | 16 | 24 | 32 | 48 | 56 |
|---|---|---|---|---|---|---|
| 8-chart atlas | 15.9 % | 4.16 % | 1.79 % | 1.05 % | 0.538 % | **0.392 %** |
| 12-chart atlas | 9.15 % | 2.59 % | 1.16 % | 0.634 % | 0.334 % | **0.255 %** |

Observed order ≈ 1.9 (second order). Schwarz converges in 4–6 sweeps; n = 56 takes about
5 min (8 charts) and 14 min (12 charts).

**PINN** (one 4×64 tanh MLP per chart, 800 Schwarz sweeps)

| atlas | 12-chart | 8-chart |
|---|---|---|
| relative L² error | **0.182 %** (25 min) | **0.211 %** (18 min) |

The PINN numbers are for seed 42 and move by a few hundredths of a percent with the seed
and the BLAS backend.

## What is in this directory

| File | Contents |
|---|---|
| `geometry.py` | bunny download, closing the scan's open base, exact inside test (ray casting) and distances (VTK) |
| `atlas.py` | loads the frozen atlas: chart frames and radii, decoder and partition-of-unity networks |
| `chart_fem.py` | P1 FEM on one chart: mesh cut to the bunny, boundary nodes snapped onto the surface, cached factorisation |
| `run_fem.py` | Schwarz FEM driver (h-refinement sweep) |
| `run_pinn.py` | Schwarz PINN driver |
| `plot_results.py` | the figure above |
| `assets/` | the two frozen atlases (8 and 12 charts): arrays (`.npz`) and network weights (`.pt`), 5.4 MB |

## Method notes

**Atlas.** Each chart has a seed, an orthonormal frame, a support radius, a decoder network
(reference coordinates → physical point) and a mask network; the softmax of the mask logits
is the partition of unity that blends the chart solutions. The decoders were trained with a
one-sided Jacobian barrier and are essentially rigid (det J = 1.00 ± 0.02).

**Geometry.** The scan is open at its base; each open loop is closed with a fan of triangles
oriented like the rest of the surface. Inside/outside is decided by ray casting on the closed
surface, so the domain is the true bunny, not a learned approximation of it.

**FEM.** Each chart carries a structured tetrahedral mesh of its reference cube (half-width
1.5 r), mapped by the decoder and cut to the bunny. Nodes on cut faces are moved onto the
surface — without this a staircase boundary limits P1 to first order — and receive `g` there.
Nodes on the cube faces are Schwarz interfaces and receive the partition-of-unity blend of
every other chart whose mesh covers them. Every node starts at zero; the iteration stops when
the relative nodal update falls below 1e-6.

**PINN.** Each chart's network is trained on collocation points inside the bunny (its atlas
region, jittered, plus wherever its partition-of-unity weight exceeds 0.02) and on points of
the surface, with interface value and flux coupling to its frozen neighbours. Each loss term
is divided by the mean square of its data (`f` for the residual, `g` for boundary and
interface values), so no term dominates through its units. The learning rate follows a cosine
schedule. The state kept is the one with the lowest training loss on fixed samples; the error
against the exact solution is only logged.

**Verification.** The exact solution enters only as boundary data. It never sets an initial
guess, a stopping rule, an acceptance test or a checkpoint choice.
