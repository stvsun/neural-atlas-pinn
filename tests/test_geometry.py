"""Independent polynomial and finite-difference checks for chart calculus."""

import math

import pytest
import torch

from mapped_sphere.geometry import (
    ChartGeometry,
    divergence,
    jacobian,
    laplace_beltrami,
    scalar_gradient,
)
from mapped_sphere.maps import TorusChart, torus_boundary_normals, torus_from_angles


def sample_points(dtype=torch.float64):
    return torch.tensor([[0.2, -0.3, 0.1], [-0.4, 0.1, 0.5],
                         [0.1, 0.4, -0.2]], dtype=dtype, requires_grad=True)


def nonlinear_map(xi):
    a, b, c = xi.unbind(-1)
    return torch.stack((a + 0.1 * b.square(), 1.2 * b + 0.05 * c.square(),
                        0.9 * c + 0.08 * a.square()), dim=-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nonsymmetric_affine_chain_rule_and_metric(dtype):
    xi = sample_points(dtype)
    J = torch.tensor([[1.3, 0.2, -0.1], [0.4, 0.9, 0.3],
                      [0.0, -0.2, 1.1]], dtype=dtype)
    x = xi @ J.T
    geom = ChartGeometry(J)
    expected_J = J.expand(len(xi), -1, -1)
    torch.testing.assert_close(jacobian(x, xi), expected_J)
    torch.testing.assert_close(geom.jacobian, J)
    torch.testing.assert_close(geom.metric, J.T @ J)
    torch.testing.assert_close(geom.inverse_metric @ geom.metric, torch.eye(3, dtype=dtype))
    grad_ref = scalar_gradient(x.square().sum(-1), xi)
    torch.testing.assert_close(geom.scalar_gradient(grad_ref), 2.0 * x)
    A = torch.tensor([[0.2, -0.4, 0.3], [0.6, 0.1, 0.5]], dtype=dtype)
    u = x @ A.T
    torch.testing.assert_close(geom.vector_gradient(jacobian(u, xi)),
                               A.expand(len(xi), -1, -1))


@pytest.mark.parametrize("dtype, atol", [(torch.float32, 2e-5), (torch.float64, 2e-12)])
def test_nonlinear_laplace_beltrami_matches_physical_polynomial(dtype, atol):
    xi = sample_points(dtype)
    x = nonlinear_map(xi)
    geom = ChartGeometry(jacobian(x, xi))
    u = x[:, 0] ** 4 + x[:, 1] ** 2 + x[:, 2] ** 2
    expected_gradient = torch.stack((4 * x[:, 0] ** 3, 2 * x[:, 1], 2 * x[:, 2]), -1)
    torch.testing.assert_close(geom.scalar_gradient(scalar_gradient(u, xi)),
                               expected_gradient, atol=atol, rtol=atol)
    lap = laplace_beltrami(u, xi, geom)
    torch.testing.assert_close(lap, 12 * x[:, 0] ** 2 + 4, atol=atol, rtol=atol)
    physical_lap_grad = geom.scalar_gradient(scalar_gradient(lap, xi))
    expected = torch.stack((24 * x[:, 0], torch.zeros_like(x[:, 0]),
                            torch.zeros_like(x[:, 0])), -1)
    torch.testing.assert_close(physical_lap_grad, expected, atol=atol, rtol=atol)


def test_scalar_column_and_constant_fields_allow_higher_derivatives():
    xi = sample_points()
    constant = torch.full((len(xi), 1), 7.0, dtype=xi.dtype)
    grad = scalar_gradient(constant, xi)
    torch.testing.assert_close(grad, torch.zeros_like(xi))
    torch.testing.assert_close(divergence(grad, xi), torch.zeros(len(xi), dtype=xi.dtype))
    affine = 2 * xi[:, :1] - 3 * xi[:, 1:2]
    g = scalar_gradient(affine, xi)
    torch.testing.assert_close(g, torch.tensor([2., -3., 0.], dtype=xi.dtype).expand_as(xi))
    h = jacobian(g, xi)
    torch.testing.assert_close(h, torch.zeros(len(xi), 3, 3, dtype=xi.dtype))
    torch.testing.assert_close(divergence(h, xi), torch.zeros_like(xi))
    geom = ChartGeometry(torch.eye(3, dtype=xi.dtype))
    torch.testing.assert_close(laplace_beltrami(constant, xi, geom),
                               torch.zeros(len(xi), dtype=xi.dtype))


def test_vector_and_tensor_divergence_independent_components():
    xi = sample_points()
    a, b, c = xi.unbind(-1)
    v = torch.stack((a.square(), a * b, c ** 3), -1)
    torch.testing.assert_close(divergence(v, xi), 3 * a + 3 * c.square())
    P = torch.stack((v, torch.stack((b, b.square(), -c), -1)), -2)
    torch.testing.assert_close(divergence(P, xi),
                               torch.stack((3 * a + 3 * c.square(), 2 * b - 1), -1))


@pytest.mark.parametrize("d", [1, 2, 4])
def test_geometry_dimension_and_broadcast_batches(d):
    diagonal = torch.arange(1, d + 1, dtype=torch.float64)
    J = torch.diag(diagonal).expand(2, 1, d, d)
    geom = ChartGeometry(J)
    grad = torch.ones(2, 5, d, dtype=torch.float64)
    expected = (1 / diagonal).expand(2, 5, d)
    torch.testing.assert_close(geom.scalar_gradient(grad), expected)
    torch.testing.assert_close(geom.volume, diagonal.prod().expand(2, 1))


@pytest.mark.parametrize("bad", [torch.zeros(3, 3),
                                torch.tensor([[1., 0.], [0., float("nan")]]),
                                torch.tensor([[1., 0.], [0., float("inf")]])])
def test_invalid_chart_geometry_is_rejected(bad):
    with pytest.raises(ValueError):
        ChartGeometry(bad)


def test_gradient_requires_coordinate_grad_tracking():
    xi = sample_points().detach()
    with pytest.raises(ValueError):
        scalar_gradient(xi[:, 0], xi)


def test_chart_geometry_preserves_parameter_gradients_and_second_derivatives():
    base = torch.tensor([[1.2, 0.1], [0.2, 1.1]], dtype=torch.float64, requires_grad=True)

    def response(J):
        geom = ChartGeometry(J)
        return torch.cat((geom.inverse.reshape(-1), geom.volume.reshape(-1),
                          geom.inverse_metric.reshape(-1)))

    assert torch.autograd.gradcheck(response, (base,), eps=1e-6, atol=1e-6, rtol=1e-5)
    assert torch.autograd.gradgradcheck(response, (base,), eps=1e-6, atol=1e-6, rtol=1e-5)


def test_laplace_residual_backpropagates_to_geometry_and_field_parameters():
    xi = sample_points()
    theta = torch.tensor([0.2, -0.3, 0.1], dtype=xi.dtype, requires_grad=True)
    amplitude = torch.tensor(0.7, dtype=xi.dtype, requires_grad=True)
    x = xi * theta.exp()
    geom = ChartGeometry(jacobian(x, xi))
    u = amplitude * xi.square().sum(-1)
    lap = laplace_beltrami(u, xi, geom)
    inv_sq = torch.exp(-2 * theta.detach())
    expected_lap = 2 * amplitude.detach() * inv_sq.sum()
    torch.testing.assert_close(lap, expected_lap.expand_as(lap), atol=2e-14, rtol=0)
    residual = lap - 1.7
    loss = 0.5 * residual.square().mean()
    grad_theta, grad_amplitude = torch.autograd.grad(loss, (theta, amplitude))
    expected_residual = expected_lap - 1.7
    torch.testing.assert_close(grad_theta, -4 * expected_residual * amplitude.detach() * inv_sq,
                               atol=2e-13, rtol=0)
    torch.testing.assert_close(grad_amplitude, 2 * expected_residual * inv_sq.sum(),
                               atol=2e-13, rtol=0)


@pytest.mark.parametrize("dtype, step, atol", [(torch.float32, 1e-3, 8e-5),
                                             (torch.float64, 1e-6, 2e-10)])
def test_torus_jacobian_matches_central_differences(dtype, step, atol):
    chart = TorusChart(R=1.2, r=0.3, phi_center=0.5, phi_halfwidth=0.7)
    xi = sample_points(dtype)
    fd = []
    for j in range(3):
        perturb = torch.zeros_like(xi)
        perturb[:, j] = step
        fd.append((chart(xi + perturb) - chart(xi - perturb)) / (2 * step))
    J = chart.jacobian(xi)
    torch.testing.assert_close(J, torch.stack(fd, -1), atol=atol, rtol=atol)
    torch.testing.assert_close(J, jacobian(chart(xi), xi))
    geom = ChartGeometry(J)
    lap = laplace_beltrami(chart(xi).square().sum(-1), xi, geom)
    torch.testing.assert_close(lap, torch.full((len(xi),), 6., dtype=dtype),
                               atol=50 * atol, rtol=50 * atol)


def test_torus_centerline_is_explicitly_singular():
    xi = torch.tensor([[0., 0.3, -1.]], dtype=torch.float64)
    with pytest.raises(ValueError):
        ChartGeometry(TorusChart().jacobian(xi))


def test_torus_boundary_normals_match_implicit_surface_gradient():
    phi = torch.tensor([0.2, 1.3, -0.5], dtype=torch.float64)
    theta = torch.tensor([0.4, -0.7, math.pi / 2], dtype=torch.float64)
    rho = torch.full_like(phi, 0.3)
    x = torus_from_angles(phi, theta, rho, 1.2).requires_grad_(True)
    tube_distance = torch.sqrt((torch.linalg.vector_norm(x[:, :2], dim=-1) - 1.2) ** 2
                               + x[:, 2] ** 2)
    expected = torch.autograd.grad(tube_distance.sum(), x)[0]
    normals = torus_boundary_normals(phi, theta)
    torch.testing.assert_close(normals, expected)
    torch.testing.assert_close(torch.linalg.vector_norm(normals, dim=-1), torch.ones_like(phi))
