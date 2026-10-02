# Differential geometry library: completed integration phase

The shared implementation is in `mapped_sphere/geometry.py`, `piola.py`, and
`maps.py`. All five examples use it. Public equations, tensor shapes, units,
orientation conventions, and limitations are in `NOTES.md`; a runnable usage
example is in `README.md`.

The Jacobian convention is physical components by reference derivatives.
Piola flux maps are signed by default; positive-volume weak forms explicitly use
absolute determinants. Singular/nonfinite maps raise instead of being clipped.
Bunny FEM's previous singular-value/determinant clipping was removed. Bunny PINN
keeps its frame-coordinate formulation. Constitutive updates and solver policies
were not changed. Example 4 still evaluates a prescribed displacement field.

## Executed checks

Empirical, 2026-10-02: **45 tests passed in 1.66 s**, local CPU. Tests cover
nonlinear Laplace–Beltrami and Piola identities, signed/reflected charts, Nanson
flux/traction, anisotropic diffusion, exact constant derivatives, parameter
gradients through both geometry and fields, and mapped affine FEM patches.
The patch assertions require gradient errors below `1e-15` and interior forces
below `1e-14`. Both saved bunny atlases are exercised without downloading a mesh.

```bash
pip install -e ".[test,bunny]"
python -m pytest -q
```

The execution environment was Python 3.14.6, PyTorch 2.14.1, NumPy 2.5.3,
SciPy 1.18.1, pytest 9.1.1. The interpreter was
`/Users/wsun/Documents/Softwares/WOS/.venv/bin/python`; no changes were made to that
environment. Full commands, source hashes, environment, and test results are in
`validation/geometry_library/manifest.json`, `pytest.xml`, and `pytest.txt`.

Empirical comparisons against commit `b74022111d3bf61d8428e0e1ebb279efca72a48f`
used CPU float64. Raw numbers are in the adjacent `scalar_parity.json` and
`mechanics_regression.json` reports.

| Regression | Maximum absolute difference |
|---|---:|
| Example 1 residual, 128 points, seed 142 | 8.88e-16 |
| Example 1 loss parameter gradient | 1.78e-15 |
| Bunny diffusion, 128 points per chart, all 20 saved charts | 1.07e-14 |
| Torus FEM tangent stiffness, 27 nodes / 48 tets, seed 20261002 | 3.56e-15 |
| Torus FEM material-parameter objective gradient | 8.68e-19 |
| Both Example 4 drivers: traction and parameter gradients, 64 points | 0 |

The sampled bunny singular values were at least `0.984704` (8-chart atlas) and
`0.988454` (12-chart atlas); the old clipping was inactive at these samples.
These are finite-sample checks, not bounds on the whole decoder domain.

Empirical packaging checks: a wheel built with `pip wheel --no-deps
--no-build-isolation` contained all three new modules and imported from an
isolated directory under `/tmp`. The README example returned a Laplacian of
`6.000000000000001`. All seven example entrypoints returned exit code zero for
`--help` with `/tmp` as the working directory.

## Open issues

Full paper training, inverse recovery, and bunny surface-mesh solves were not rerun.
No GPU, mesh-convergence, constitutive-validity, or global chart-injectivity claim follows.
Pre-existing centerline, inversion, and empty-mesh limitations are listed in NOTES.md.
