"""Analytic torus coordinates shared by the forward and inverse examples."""

import math

import torch


def torus_from_angles(phi, theta, rho, R):
    """Return x=((R+rho cos(theta)) cos(phi), ..., rho sin(theta)).

    Source: circular torus parameterization; phi and theta are scalar angles
    in radians, rho and major radius R have length units. Inputs broadcast.
    """
    phi, theta, rho = torch.broadcast_tensors(phi, theta, rho)
    radial = R + rho * torch.cos(theta)
    return torch.stack([
        radial * torch.cos(phi), radial * torch.sin(phi), rho * torch.sin(theta)
    ], dim=-1)


def torus_boundary_normals(phi, theta):
    """Return n=(cos(theta)cos(phi),cos(theta)sin(phi),sin(theta)).

    Source: outward radial derivative of the circular torus map, normalized.
    Angles are in radians; the vector is dimensionless and has unit norm.
    """
    phi, theta = torch.broadcast_tensors(phi, theta)
    normal = torch.stack([
        torch.cos(theta) * torch.cos(phi), torch.cos(theta) * torch.sin(phi),
        torch.sin(theta),
    ], dim=-1)
    return normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True)


class TorusChart(torch.nn.Module):
    """Circular torus sector, preserving the examples' polar chart convention.

    phi=phi_center+phi_halfwidth*xi_0, theta=pi*xi_1,
    rho=r*(1+xi_2)/2. R and r have length units, xi is dimensionless.
    This map degenerates at xi_2=-1 and identifies the theta seam; it is a
    local volume chart only away from those sets. No inversion is supplied.
    """

    def __init__(self, R=1.0, r=0.35, phi_center=0.0, phi_halfwidth=math.pi / 4):
        super().__init__()
        self.R, self.r = R, r
        self.phi_center = phi_center
        self.phi_halfwidth = phi_halfwidth

    def forward(self, xi, **kwargs):
        """Evaluate the circular torus parameterization x=phi(xi)."""
        phi = self.phi_center + xi[..., 0] * self.phi_halfwidth
        theta = math.pi * xi[..., 1]
        rho = 0.5 * self.r * (1 + xi[..., 2])
        return torus_from_angles(phi, theta, rho, self.R)

    def jacobian(self, xi, **kwargs):
        """Return J_iA=dx_i/dxi_A by analytic differentiation of forward."""
        phi = self.phi_center + xi[..., 0] * self.phi_halfwidth
        theta = math.pi * xi[..., 1]
        rho = 0.5 * self.r * (1 + xi[..., 2])
        cp, sp = torch.cos(phi), torch.sin(phi)
        ct, st = torch.cos(theta), torch.sin(theta)
        radial = self.R + rho * ct
        zero = torch.zeros_like(phi)
        dphi = self.phi_halfwidth * torch.stack([-radial * sp, radial * cp, zero], dim=-1)
        dtheta = math.pi * torch.stack([-rho * st * cp, -rho * st * sp, rho * ct], dim=-1)
        drho = 0.5 * self.r * torch.stack([ct * cp, ct * sp, st], dim=-1)
        return torch.stack([dphi, dtheta, drho], dim=-1)
