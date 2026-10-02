"""Piola maps and integration factors for Euclidean volume charts.

Source for signed vector Piola maps:
https://defelement.org/finite-elements.html
Tensor row maps, diffusion and surface factors follow from the same chain
rule and conservation of flux. Fields are already evaluated at x=phi(xi).
``geometry`` is a :class:`mapped_sphere.geometry.ChartGeometry` instance.

Contravariant/tensor maps use signed det(J) by default (oriented convention).
Pass ``oriented=False`` for positive-volume weak forms using |det(J)|.
Diffusion and outward surface measures always use the positive volume.
On a smooth nonsingular chart the determinant sign is constant.
"""

import torch


def _measure(geometry, oriented):
    return geometry.determinant if oriented else geometry.volume


def contravariant_pullback(q, geometry, oriented=True):
    """Return q_hat = det(J) J^{-1} q (inverse H(div) Piola map)."""
    return _measure(geometry, oriented).unsqueeze(-1) * (
        geometry.inverse @ q.unsqueeze(-1)
    ).squeeze(-1)


def contravariant_pushforward(q_ref, geometry, oriented=True):
    """Return q = J q_hat / det(J) (H(div) Piola map, DefElement)."""
    return (geometry.jacobian @ q_ref.unsqueeze(-1)).squeeze(-1) / _measure(
        geometry, oriented
    ).unsqueeze(-1)


def covariant_pullback(q, geometry):
    """Return q_hat = J^T q (inverse H(curl) Piola map)."""
    return (geometry.jacobian.transpose(-2, -1) @ q.unsqueeze(-1)).squeeze(-1)


def covariant_pushforward(q_ref, geometry):
    """Return q = J^{-T} q_hat (H(curl) Piola map, DefElement)."""
    return geometry.scalar_gradient(q_ref)


def tensor_pullback(tensor, geometry, oriented=True):
    """Return T_hat = det(J) T J^{-T}, applying H(div) to each row.

    The leading component index remains in the physical Cartesian frame.
    This is a row-wise flux map, not a double Piola map for both indices.
    """
    return _measure(geometry, oriented)[..., None, None] * (
        tensor @ geometry.inverse.transpose(-2, -1)
    )


def tensor_pushforward(tensor_ref, geometry, oriented=True):
    """Return T = T_hat J^T / det(J) (inverse row-wise flux map)."""
    return (tensor_ref @ geometry.jacobian.transpose(-2, -1)) / _measure(
        geometry, oriented
    )[..., None, None]


def diffusion_pullback(geometry, conductivity=None):
    """Return A=|det J| J^{-1} K J^{-T} by weak-form change of variables.

    ``conductivity`` is a physical tensor (...,d,d); None means K=I.
    A scalar conductivity k can be supplied as k*I. Symmetry/positivity
    of K is the caller's constitutive assumption, not imposed here.
    """
    inverse = geometry.inverse
    if conductivity is None:
        transformed = geometry.inverse_metric
    else:
        if conductivity.shape[-2:] != inverse.shape[-2:]:
            raise ValueError("conductivity must have shape (...,d,d)")
        transformed = inverse @ conductivity @ inverse.transpose(-2, -1)
    return geometry.volume[..., None, None] * transformed


def cofactor(geometry):
    """Return cof(J)=det(J) J^{-T} (signed cofactor identity)."""
    return geometry.determinant[..., None, None] * geometry.inverse.transpose(-2, -1)


def surface_transform(reference_normal, geometry):
    """Return (outward unit normal, positive area scale) by Nanson's formula.

    For a unit reference normal N, n*dS/dS_ref=|det J| J^{-T} N.
    The absolute determinant retains outward normals for reflected charts;
    use ``cofactor(geometry) @ N`` for signed, oriented area vectors.
    """
    length = torch.linalg.vector_norm(reference_normal, dim=-1)
    if not torch.allclose(length, torch.ones_like(length), rtol=1e-5, atol=1e-7):
        raise ValueError("reference normals must have unit length")
    area_vector = geometry.volume.unsqueeze(-1) * geometry.scalar_gradient(reference_normal)
    area = torch.linalg.vector_norm(area_vector, dim=-1)
    return area_vector / area.unsqueeze(-1), area
