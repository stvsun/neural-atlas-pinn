"""Piola identities, anisotropic diffusion, and Nanson flux checks."""

import pytest
import torch

from mapped_sphere.geometry import ChartGeometry, divergence, jacobian
from mapped_sphere.piola import (
    cofactor,
    contravariant_pullback,
    contravariant_pushforward,
    covariant_pullback,
    covariant_pushforward,
    diffusion_pullback,
    surface_transform,
    tensor_pullback,
    tensor_pushforward,
)


def affine_matrix(dtype, reflected):
    J = torch.tensor([[1.3, 0.2, -0.1], [0.4, 0.9, 0.3],
                      [0.0, -0.2, 1.1]], dtype=dtype)
    if reflected:
        J[:, 0] *= -1
    return J


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("reflected", [False, True])
@pytest.mark.parametrize("oriented", [False, True])
def test_piola_roundtrips_and_independent_affine_formula(dtype, reflected, oriented):
    J = affine_matrix(dtype, reflected)
    geom = ChartGeometry(J)
    q = torch.tensor([[0.4, -0.2, 0.7], [1.0, 0.5, -0.3]], dtype=dtype)
    P = torch.tensor([[[0.1, 0.3, -0.2], [0.4, -0.5, 0.6]],
                      [[0.7, -0.8, 0.9], [-0.2, 0.1, 0.5]]], dtype=dtype)
    measure = torch.det(J) if oriented else torch.det(J).abs()
    q_ref = contravariant_pullback(q, geom, oriented=oriented)
    expected_q_ref = measure * torch.linalg.solve(J, q.T).T
    torch.testing.assert_close(q_ref, expected_q_ref)
    torch.testing.assert_close(contravariant_pushforward(q_ref, geom, oriented=oriented), q)
    cov_ref = covariant_pullback(q, geom)
    torch.testing.assert_close(cov_ref, q @ J)
    torch.testing.assert_close(covariant_pushforward(cov_ref, geom), q)
    P_ref = tensor_pullback(P, geom, oriented=oriented)
    expected_P_ref = measure * torch.linalg.solve(J, P.transpose(-1, -2)).transpose(-1, -2)
    torch.testing.assert_close(P_ref, expected_P_ref)
    torch.testing.assert_close(tensor_pushforward(P_ref, geom, oriented=oriented), P)


@pytest.mark.parametrize("reflected", [False, True])
def test_nanson_outward_normal_flux_and_tensor_traction(reflected):
    J = affine_matrix(torch.float64, reflected)
    geom = ChartGeometry(J)
    n_ref = torch.tensor([0., 0., 1.], dtype=J.dtype)
    normal, area = surface_transform(n_ref, geom)
    t0, t1 = J[:, 0], J[:, 1]
    expected_area = torch.linalg.vector_norm(torch.linalg.cross(t0, t1))
    torch.testing.assert_close(area, expected_area)
    torch.testing.assert_close(normal @ t0, torch.tensor(0., dtype=J.dtype), atol=1e-14, rtol=0)
    torch.testing.assert_close(normal @ t1, torch.tensor(0., dtype=J.dtype), atol=1e-14, rtol=0)
    assert normal @ J[:, 2] > 0  # outward on the xi_2 = 1 face, even under reflection
    q = torch.tensor([0.7, -0.2, 0.5], dtype=J.dtype)
    q_ref = contravariant_pullback(q, geom, oriented=False)
    torch.testing.assert_close(q_ref @ n_ref, (q @ normal) * area)
    P = torch.tensor([[0.1, 0.2, -0.3], [0.4, -0.2, 0.5]], dtype=J.dtype)
    P_ref = tensor_pullback(P, geom, oriented=False)
    torch.testing.assert_close(P_ref @ n_ref, (P @ normal) * area)
    torch.testing.assert_close(cofactor(geom) @ n_ref,
                               torch.sign(geom.determinant) * normal * area)


@pytest.mark.parametrize("reflected", [False, True])
def test_nonlinear_piola_divergence_and_cofactor_identity(reflected):
    xi = torch.tensor([[0.2, -0.3, 0.1], [-0.4, 0.1, 0.5]],
                      dtype=torch.float64, requires_grad=True)
    a, b, c = xi.unbind(-1)
    x = torch.stack((a + 0.1 * b.square(), 1.2 * b + 0.05 * c.square(),
                     0.9 * c + 0.08 * a.square()), -1)
    if reflected:
        x = x * torch.tensor([-1., 1., 1.], dtype=x.dtype)
    geom = ChartGeometry(jacobian(x, xi))
    torch.testing.assert_close(divergence(cofactor(geom), xi), torch.zeros_like(x),
                               atol=2e-13, rtol=0)
    q = torch.stack((x[:, 0] ** 2, x[:, 0] * x[:, 1], x[:, 2] ** 3), -1)
    expected_div = 3 * x[:, 0] + 3 * x[:, 2] ** 2
    torch.testing.assert_close(divergence(contravariant_pullback(q, geom), xi),
                               geom.determinant * expected_div, atol=2e-13, rtol=2e-13)
    P = torch.stack((q, 2 * q), -2)
    torch.testing.assert_close(divergence(tensor_pullback(P, geom), xi),
                               geom.determinant[:, None] * torch.stack((expected_div, 2 * expected_div), -1),
                               atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize("reflected", [False, True])
def test_anisotropic_diffusion_preserves_energy_and_positive_measure(reflected):
    J = affine_matrix(torch.float64, reflected)
    geom = ChartGeometry(J)
    K = torch.tensor([[2., 0.3, 0.2], [0.3, 1.5, -0.1],
                      [0.2, -0.1, 0.8]], dtype=J.dtype)
    g_ref = torch.tensor([0.3, -0.5, 0.7], dtype=J.dtype)
    g_phys = torch.linalg.solve(J.T, g_ref)
    A = diffusion_pullback(geom, K)
    torch.testing.assert_close(g_ref @ A @ g_ref, J.det().abs() * (g_phys @ K @ g_phys))
    assert torch.linalg.eigvalsh(A).min() > 0
    torch.testing.assert_close(diffusion_pullback(geom), diffusion_pullback(geom, torch.eye(3, dtype=J.dtype)))


def test_piola_parameter_gradcheck_and_gradgradcheck():
    J = affine_matrix(torch.float64, False).requires_grad_(True)
    q = torch.tensor([0.2, -0.3, 0.5], dtype=J.dtype, requires_grad=True)

    def response(matrix, vector):
        geom = ChartGeometry(matrix)
        return torch.cat((contravariant_pullback(vector, geom),
                          diffusion_pullback(geom).reshape(-1),
                          cofactor(geom).reshape(-1)))

    assert torch.autograd.gradcheck(response, (J, q), eps=1e-6, atol=1e-6, rtol=1e-5)
    assert torch.autograd.gradgradcheck(response, (J, q), eps=1e-6, atol=1e-6, rtol=1e-5)
