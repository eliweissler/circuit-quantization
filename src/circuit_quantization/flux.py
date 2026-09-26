"""External-flux assignment and irrotational gauge selection."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .coordinates import CoordinateSystem
from .matrices import CircuitMatrices


@dataclass(frozen=True)
class FluxAssignment:
    assignment: sp.ImmutableMatrix
    residual_basis: tuple[sp.ImmutableMatrix, ...]
    gauge: str


def _zero_matrix(matrix: sp.MatrixBase, tolerance: float = 1e-12) -> bool:
    for entry in matrix:
        entry = sp.simplify(entry)
        if entry.equals(0):
            continue
        if not entry.free_symbols:
            try:
                if abs(complex(entry.evalf())) <= tolerance:
                    continue
            except (TypeError, ValueError):
                pass
        return False
    return True


def assign_flux(
    matrices: CircuitMatrices,
    coordinates: CoordinateSystem,
    gauge: str = "irrotational",
) -> FluxAssignment:
    """Assign loop fluxes to branches in the projected irrotational gauge."""
    if gauge != "irrotational":
        raise ValueError("Only the 'irrotational' gauge is supported")
    g = sp.Matrix(matrices.loop_matrix)
    branch_count = len(matrices.branches)
    if g.rows == 0:
        return FluxAssignment(
            assignment=sp.ImmutableMatrix(sp.zeros(branch_count, 0)),
            residual_basis=(),
            gauge=gauge,
        )

    omega_internal = sp.Matrix(matrices.omega_internal)
    capacitance_branch = sp.Matrix(matrices.capacitance_branch)
    z = sp.Matrix(coordinates.transformation)
    c_inverse = sp.Matrix(coordinates.inverse_capacitance)

    g_plus = g.T * (g * g.T).inv()
    projector = (
        omega_internal.T
        * z
        * c_inverse
        * z.T
        * omega_internal
        * capacitance_branch
    )
    assignment = sp.simplify((sp.eye(branch_count) - projector) * g_plus)

    loop_error = sp.simplify(g * assignment - sp.eye(g.rows))
    if not _zero_matrix(loop_error):
        raise ValueError(f"Flux assignment violates G B = I: {loop_error}")
    emf_error = sp.simplify(
        z.T * omega_internal * capacitance_branch * assignment
    )
    if not _zero_matrix(emf_error):
        raise ValueError(
            "Could not construct an exact projected irrotational assignment: "
            f"{emf_error}"
        )

    stacked = g.col_join(z.T * omega_internal * capacitance_branch)
    residual = tuple(sp.ImmutableMatrix(vector) for vector in stacked.nullspace())
    return FluxAssignment(
        assignment=sp.ImmutableMatrix(assignment),
        residual_basis=residual,
        gauge=gauge,
    )
