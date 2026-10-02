"""Differential operators for full-dimensional Euclidean coordinate charts.

Convention: ``J[..., i, A] = dx_i/dxi_A``; vectors store components on the
last axis. Coordinates are ``(N, d)`` and fields must be pointwise in the
batch: output row n depends only on input row n. Batch-coupled networks
(e.g. training-mode BatchNorm) do not satisfy this contract.

The identities below follow the multivariable chain rule and change of
variables; the module implements volume charts, not embedded surfaces.
For dimensionless xi and physical x in length units L, J has units L,
g has units L^2 and volume has units L^d. Field derivatives inherit the
field's units. No coordinate or parameter graph is detached internally.
"""

import torch


def _check_coordinates(xi):
    if xi.ndim != 2 or not xi.is_floating_point() or not xi.requires_grad:
        raise ValueError("coordinates must be a floating (N, d) tensor requiring gradients")


def _derivative(values, xi, create_graph):
    # autograd omits the graph for constants and some affine derivatives.
    # Keep exact zeros connected so repeated differentiation remains defined.
    zero = xi * 0 + values.unsqueeze(-1) * 0
    if not values.requires_grad:
        return zero if create_graph else zero.detach()
    derivative = torch.autograd.grad(
        values, xi, grad_outputs=torch.ones_like(values),
        create_graph=create_graph, retain_graph=True, allow_unused=True,
    )[0]
    if derivative is None:
        return zero if create_graph else zero.detach()
    return derivative + zero if create_graph else derivative


def scalar_gradient(u, xi, create_graph=True):
    """Return du/dxi, shape (N,d), by the pointwise chain rule.

    ``u`` has shape (N,) or (N,1), evaluated on ``xi`` with autograd enabled.
    Constant fields and repeated derivatives of affine fields are supported.
    """
    _check_coordinates(xi)
    if u.ndim == 2 and u.shape[-1] == 1:
        u = u.squeeze(-1)
    if u.shape != xi.shape[:1]:
        raise ValueError("scalar field must have shape (N,) or (N,1)")
    return _derivative(u, xi, create_graph)


def jacobian(v, xi, create_graph=True):
    """Return J[n,i,A]=dv[n,i]/dxi[n,A] by the pointwise chain rule."""
    _check_coordinates(xi)
    if v.ndim != 2 or v.shape[0] != xi.shape[0] or v.shape[1] == 0:
        raise ValueError("vector field must have shape (N,m), m > 0")
    return torch.stack([
        _derivative(v[:, i], xi, create_graph) for i in range(v.shape[1])
    ], dim=1)


def divergence(v, xi, create_graph=True):
    """Return sum_A dv_A/dxi_A, or its row-wise tensor counterpart.

    Source: Cartesian divergence definition. Vector input (N,d) returns
    (N,); tensor input (N,m,d) returns (N,m). No geometry factor is implicit.
    """
    _check_coordinates(xi)
    if v.ndim not in (2, 3) or v.shape[0] != xi.shape[0] or v.shape[-1] != xi.shape[1]:
        raise ValueError("field must have shape (N,d) or (N,m,d)")
    if v.ndim == 3:
        return torch.stack([
            divergence(v[:, i, :], xi, create_graph) for i in range(v.shape[1])
        ], dim=1)
    return sum(_derivative(v[:, A], xi, create_graph)[:, A] for A in range(xi.shape[1]))


class ChartGeometry:
    """Geometry of an invertible square Jacobian, from change of variables.

    ``jacobian`` has shape (...,d,d), with physical rows and reference
    columns. ``determinant`` is signed; ``volume`` is its magnitude.
    Singular or nonfinite samples raise ValueError. No clipping or
    pseudoinverse is used. Invertibility at samples does not establish
    global injectivity or exclude folds between samples.
    """

    def __init__(self, jacobian):
        if (jacobian.ndim < 2 or jacobian.shape[-1] == 0
                or jacobian.shape[-2] != jacobian.shape[-1]
                or not jacobian.is_floating_point()):
            raise ValueError("Jacobian must be a floating tensor of shape (...,d,d)")
        if not bool(torch.isfinite(jacobian).all()):
            raise ValueError("Jacobian contains nonfinite entries")
        determinant = torch.linalg.det(jacobian)
        if not bool((torch.isfinite(determinant) & (determinant != 0)).all()):
            raise ValueError("Jacobian is singular or has a nonfinite determinant")
        try:
            inverse = torch.linalg.inv(jacobian)
        except torch.linalg.LinAlgError as error:
            raise ValueError("Jacobian is singular") from error
        if not bool(torch.isfinite(inverse).all()):
            raise ValueError("Jacobian inverse contains nonfinite entries")
        self.jacobian = jacobian
        self.inverse = inverse
        self.determinant = determinant
        self.volume = determinant.abs()

    @property
    def metric(self):
        """Return g_AB=sum_i J_iA J_iB (Euclidean metric pullback)."""
        return self.jacobian.transpose(-2, -1) @ self.jacobian

    @property
    def inverse_metric(self):
        """Return g^{-1}=J^{-1}J^{-T} (inverse metric identity)."""
        return self.inverse @ self.inverse.transpose(-2, -1)

    def scalar_gradient(self, reference_gradient):
        """Return grad_x u = J^{-T} grad_xi u (chain rule)."""
        return (self.inverse.transpose(-2, -1) @ reference_gradient.unsqueeze(-1)).squeeze(-1)

    def vector_gradient(self, reference_gradient):
        """Return (grad_x v)_ij=(grad_xi v)_iA (J^{-1})_Aj (chain rule)."""
        return reference_gradient @ self.inverse


def laplace_beltrami(u, xi, geometry, create_graph=True):
    """Return Delta_g u = |det J|^{-1} Div_xi(|det J| g^{-1} grad_xi u).

    Source: Euclidean Laplacian under change of variables. For nonlinear
    charts, geometry must be formed from the same ``xi`` with an intact
    derivative graph (e.g. ``ChartGeometry(jacobian(x, xi))``).
    """
    gradient = scalar_gradient(u, xi, create_graph=True)
    flux = geometry.volume.unsqueeze(-1) * (
        geometry.inverse_metric @ gradient.unsqueeze(-1)
    ).squeeze(-1)
    return divergence(flux, xi, create_graph=create_graph) / geometry.volume
