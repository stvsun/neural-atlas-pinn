# Chart calculus and Piola API

## Scope and convention

The library treats smooth, full-dimensional charts in Euclidean space with no
material symmetry restriction. A chart change does not prescribe small- or
finite-strain kinematics, temperature, or rate dependence. The existing examples
retain their material models and boundary conditions. The operators below apply
to scalar Dirichlet problems and vector/tensor fluxes across chart boundaries;
they introduce no initial conditions.

Let the reference point be the vector \(\xi\in\mathbb R^d\), taken dimensionless,
and the physical point be the vector \(x=\phi(\xi)\in\mathbb R^d\), with length
units \(L\). The second-order Jacobian is \(J_{iA}=\partial x_i/\partial\xi_A\)
with units \(L\). Its scalar signed determinant is \(j=\det J\), its positive
volume density is \(v=|j|\), both with units \(L^d\). The second-order metric
\(g=J^T J\) has units \(L^2\). Physical indices occupy matrix rows; reference
derivatives occupy columns. Repeated indices are summed.

```python
from mapped_sphere.geometry import (
    ChartGeometry, jacobian, scalar_gradient, divergence, laplace_beltrami,
)
from mapped_sphere import piola
from mapped_sphere.maps import TorusChart, torus_from_angles, torus_boundary_normals
```

`ChartGeometry(J)` accepts `(..., d, d)` tensors. Its `jacobian`, `inverse`,
`determinant`, `volume`, `metric`, and `inverse_metric` retain autograd graphs.
Transforms follow PyTorch broadcasting on leading dimensions. Do not mutate the
input Jacobian after constructing the geometry, since inverse and determinant
are cached. Scalar and vector gradient methods have distinct names to avoid
interpreting a vector batch as a matrix field.

## Differential operators

For a scalar field \(u\) with units \(U\), write
\(\widehat u(\xi)=u(\phi(\xi))\). The chain rule gives

\[
\partial_A\widehat u=J_{iA}\partial_i u,
\qquad \nabla_x u=J^{-T}\nabla_\xi\widehat u.
\]

For a vector field \(w\) with component units \(W\), its second-order gradient
has units \(W/L\) and satisfies
\(\nabla_x w=(\nabla_\xi\widehat w)J^{-1}\).
Use `geometry.scalar_gradient(grad_ref)` for `(..., d)` and
`geometry.vector_gradient(grad_ref)` for `(..., m, d)` component gradients.

For a second-order physical conductivity \(K\) with units \(\kappa\), the
reference diffusion tensor \(A\), with units \(\kappa L^{d-2}\), is

\[
A=vJ^{-1}KJ^{-T},\qquad
\operatorname{div}_x(K\nabla_x u)\circ\phi
=v^{-1}\operatorname{Div}_\xi(A\nabla_\xi\widehat u).
\]

`piola.diffusion_pullback(geometry, conductivity)` returns this tensor.
Omitting conductivity sets \(K=I\), where \(I\) is the dimensionless identity.
The positive measure makes this tensor positive definite when \(K\) is positive
definite and \(J\) is invertible, including reflected charts. The implementation
does not enforce a material assumption on the supplied conductivity.

`laplace_beltrami(u, xi, geometry)` evaluates
\(v^{-1}\operatorname{Div}_\xi(vg^{-1}\nabla_\xi\widehat u)\).
Its output has shape `(N,)` and units \(U/L^2\).

`scalar_gradient`, `jacobian`, and `divergence` differentiate already evaluated
fields with respect to a `(N, d)` tensor with `requires_grad=True`. Scalar fields
use `(N,)` or `(N, 1)`; vector fields use `(N, m)`; row-wise tensor divergence uses
`(N, m, d)` and returns `(N, m)`. `create_graph=True` is the default. Constants and
affine fields admit repeated derivatives, including exact zeros. Each output
sample must depend only on the corresponding input sample. Training-mode batch
normalization and other batch-coupled functions violate this contract.

For nonlinear charts, construct geometry from the same coordinate tensor with an
intact graph: `ChartGeometry(jacobian(chart(xi), xi))`. Detaching the Jacobian
before evaluating the mapped Laplacian omits derivatives of its coefficients.

## Flux and surface maps

For a physical vector flux \(q\) with units \(Q\), the signed contravariant
pullback has units \(QL^{d-1}\). Its divergence identity is

\[
\widehat q=jJ^{-1}(q\circ\phi),\qquad
\operatorname{Div}_\xi\widehat q=j(\operatorname{div}_x q)\circ\phi.
\]

`contravariant_pullback(q, geometry)` and `contravariant_pushforward` are inverse
maps. `covariant_pullback` uses \(J^Tq\); `covariant_pushforward` uses \(J^{-T}\).
These signed vector conventions are [cited: DefElement](https://defelement.org/finite-elements.html).
For positive-volume weak forms, pass `oriented=False` to the contravariant maps.
This replaces \(j\) with \(v\); the determinant sign must be constant on a chart.

For a second-order tensor flux \(T\) with component units \(S\), the row-wise
pullback is \(\widehat T=jTJ^{-T}\), with units \(SL^{d-1}\).
`tensor_pullback` and `tensor_pushforward` leave the leading Cartesian component
index unchanged. They do not apply a double Piola transformation. A material
deformation gradient and a chart Jacobian describe different maps; passing one
in place of the other changes the configuration of the stress.

For a dimensionless unit reference normal \(N\), the physical outward unit normal
\(n\) and positive area factor \(s\), with units \(L^{d-1}\), satisfy

\[
a=vJ^{-T}N,\qquad s=\|a\|,\qquad n=a/s,
\qquad (q\cdot n)s=(vJ^{-1}q)\cdot N.
\]

Here \(a\) is the area vector with units \(L^{d-1}\).
`surface_transform(N, geometry)` returns `(n, s)` and requires unit input normals.
`cofactor(geometry)` returns the signed second-order tensor \(jJ^{-T}\).
The absolute determinant in the outward-normal formula is deliberate for
orientation-reversing charts.

## Verification boundary

Empirical tests are in `tests/`; commands and the executed environment are recorded
in `CODEX.md`. The affine mapped FEM patch checks displacement gradients and
interior forces at round-off. Polynomial manufactured fields check the
continuous mapped operators. No mesh-convergence rate is inferred from these
checks; this refactor does not introduce a new discretization or change quadrature.

## Open issues

Samplewise invertibility does not certify global injectivity or conditioning.
The polar torus map degenerates at its centerline and identifies its angular seam.
Existing Schwarz inversion can fall back to a pseudoinverse without a final residual gate.
Example 5 retains its pre-existing empty-mesh fallback to an unmapped cube.
Full benchmark convergence, constitutive validity, and inverse identifiability remain outside these tests.
