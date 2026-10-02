# Geometry library delivery gates

| Gate | Requirement | State |
|---|---|---|
| Convention | One physical-row/reference-column Jacobian; signed and positive measures explicit | Complete |
| Library | Differentiable calculus, Piola maps, diffusion, normals, torus geometry | Complete |
| Integration | All five examples call the shared implementation | Complete |
| Verification | Nonlinear identities, parameter derivatives, affine FEM patch, example regression | Complete; see CODEX.md |
| Packaging | Build and import the distributable outside the checkout | Complete; see CODEX.md |

This phase ends at library integration. Full paper campaigns and the pre-existing
solver limitations in NOTES.md are separate work, not implied validation gates
for the coordinate-transform extraction.
