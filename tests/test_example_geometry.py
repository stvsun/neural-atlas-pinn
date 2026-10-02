"""Small CPU integration checks; these do not validate benchmark accuracy."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from mapped_sphere.elastoplastic.chart_vector_fem import ChartVectorFEMSolver
from mapped_sphere.geometry import ChartGeometry, jacobian, laplace_beltrami
from mapped_sphere.maps import TorusChart


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(autouse=True)
def restore_default_dtype():
    original = torch.get_default_dtype()
    try:
        yield
    finally:
        torch.set_default_dtype(original)


def load_example(relative_path, monkeypatch):
    path = EXAMPLES / relative_path
    monkeypatch.syspath_prepend(str(path.parent))
    name = "_example_test_" + path.parent.name + "_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_example1_exact_ellipsoid_residual(monkeypatch):
    module = load_example("example1_ellipsoid_laplace/run_ellipsoid_laplace.py", monkeypatch)
    xi = torch.tensor([[0.1, 0.2, -0.3], [0.3, -0.2, 0.1]], dtype=torch.float64)
    axes = torch.tensor([1.2, 0.8, 0.5], dtype=xi.dtype)
    source = module.source_term_mapped_constant(*axes.tolist(), "cpu", xi.dtype)
    residual = module.mapped_poisson_residual(module.exact_solution_mapped, xi,
                                              axes.reciprocal().square(), source)
    torch.testing.assert_close(residual, torch.zeros(len(xi), 1, dtype=xi.dtype),
                               atol=2e-14, rtol=0)


@pytest.mark.parametrize("n_charts", [8, 12])
def test_example2_saved_neural_chart_physical_laplacian(monkeypatch, n_charts):
    module = load_example("example2_bunny_poisson/atlas.py", monkeypatch)
    atlas = module.load_atlas(n_charts)
    xi = torch.tensor([[0.01, -0.02, 0.03], [-0.01, 0.02, -0.03]],
                      dtype=torch.float64, requires_grad=True)
    for i in range(atlas.n_charts):
        x = atlas.decode(xi, i)
        geom = ChartGeometry(jacobian(x, xi))
        lap = laplace_beltrami(x.square().sum(-1), xi, geom)
        torch.testing.assert_close(lap, torch.full((len(xi),), 6., dtype=xi.dtype),
                                   atol=5e-12, rtol=5e-12)


@pytest.mark.parametrize("filename", ["run_pinn.py", "run_fem.py"])
def test_example2_entrypoint_import(monkeypatch, filename):
    # Import only: no mesh download or training.
    module = load_example("example2_bunny_poisson/" + filename, monkeypatch)
    assert callable(module.main)


@pytest.mark.parametrize("relative_path", [
    "example3_torus_forward_elastoplastic/run_forward_bvp.py",
    "example5_torus_inverse_elastoplastic/run_inverse_elastoplastic.py",
])
def test_examples3_and5_share_torus_chart(monkeypatch, relative_path):
    module = load_example(relative_path, monkeypatch)
    chart = module.TorusChartDecoder(1.1, 0.3, phi_center=0.2, phi_halfwidth=0.8)
    shared = TorusChart(R=1.1, r=0.3, phi_center=0.2, phi_halfwidth=0.8)
    xi = torch.tensor([[0.1, 0.3, -0.2], [-0.4, 0.1, 0.5]], dtype=torch.float64)
    torch.testing.assert_close(chart(xi), shared(xi))
    torch.testing.assert_close(chart.jacobian(xi), shared.jacobian(xi))


@pytest.mark.parametrize("filename", ["run_global_atlas.py", "run_schwarz_dual.py"])
def test_example4_gradient_tensor_analytic_polynomial(monkeypatch, filename):
    module = load_example("example4_torus_inverse_neohookean/" + filename, monkeypatch)
    x = torch.tensor([[0.1, 0.3, -0.2], [-0.4, 0.1, 0.5]],
                     dtype=torch.float64, requires_grad=True)
    v = torch.stack((x[:, 0] ** 2, x[:, 1] * x[:, 2], 3 * x[:, 0] - x[:, 2]), -1)
    expected = torch.zeros(len(x), 3, 3, dtype=x.dtype)
    expected[:, 0, 0] = 2 * x[:, 0]
    expected[:, 1, 1] = x[:, 2]
    expected[:, 1, 2] = x[:, 1]
    expected[:, 2, 0] = 3
    expected[:, 2, 2] = -1
    torch.testing.assert_close(module.gradient_tensor(v, x, create_graph=False), expected)


@pytest.mark.parametrize("reflected", [False, True])
def test_mapped_vector_fem_affine_patch_to_roundoff(reflected):
    J = torch.tensor([[1.3, 0.2, -0.1], [0.4, 0.9, 0.3],
                      [0., -0.2, 1.1]], dtype=torch.float64)
    if reflected:
        J[:, 0] *= -1

    class AffineChart(torch.nn.Module):
        def forward(self, xi):
            return xi @ J.T + torch.tensor([0.3, -0.1, 0.2], dtype=xi.dtype)

    solver = ChartVectorFEMSolver(n_cells=2, chart_decoder=AffineChart())
    H = torch.tensor([[0.02, 0.03, -0.01], [0.01, -0.01, 0.02],
                      [0.00, 0.01, 0.02]], dtype=J.dtype)
    u = solver.nodes_phys @ H.T
    gradient = solver.compute_grad_u_phys(u)
    torch.testing.assert_close(gradient, H.expand_as(gradient), atol=1e-15, rtol=0)
    stress, _ = solver.make_neo_hookean(mu=1., K=10.)
    forces = solver.internal_forces(u, stress)
    interior_forces = forces[~solver.boundary_mask]
    assert interior_forces.numel() > 0
    torch.testing.assert_close(interior_forces, torch.zeros_like(interior_forces),
                               atol=1e-14, rtol=0)
    torch.testing.assert_close(solver.vol.sum(), 8 * J.det().abs(), atol=2e-14, rtol=0)
