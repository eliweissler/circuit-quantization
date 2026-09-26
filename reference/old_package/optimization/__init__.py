"""Polynomial coordinate optimization for branch-centric circuits.

This public adapter translates :class:`~circuit_quantization.Circuit` objects
to the topology representation consumed by the solver vendored from
``sircuitenum``. Solver details live in the private ``_vendor`` namespace.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ..coordinates import CoordinateSystem, ModeType
from ..matrices import CircuitMatrices, assemble
from ..model import Circuit


class OptimizationDependencyError(ImportError):
    """The optional SageMath/Singular/Z3 optimization stack is unavailable."""


@dataclass(frozen=True)
class LegacyTopology:
    """Topology-only input understood by the vendored polynomial optimizer."""

    circuit: tuple[tuple[str, ...], ...]
    edges: tuple[tuple[int, int], ...]
    ground_nodes: tuple[int, ...]


def _legacy_topology(matrices: CircuitMatrices) -> LegacyTopology:
    if matrices.source_nodes:
        raise ValueError(
            "Canonical polynomial optimization does not support voltage-source "
            "nodes; use coordinate_strategy='structural' for driven circuits"
        )
    node_index = {node: index for index, node in enumerate(matrices.internal_nodes)}
    ground_index = len(node_index)
    node_index[matrices.circuit.ground] = ground_index
    circuit = []
    edges = []
    for branch in matrices.branches:
        if branch.kind == "source":
            continue
        elements = []
        if branch.capacitance != 0:
            elements.append("C")
        if branch.kind == "junction":
            elements.append("J")
        elif branch.kind == "inductor":
            elements.append("L")
        circuit.append(tuple(elements))
        edges.append((node_index[branch.tail], node_index[branch.head]))
    return LegacyTopology(tuple(circuit), tuple(edges), (ground_index,))


def _mode_counts(indices) -> dict[str, int]:
    return {
        "compact": len(indices[ModeType.COMPACT]),
        "extended": len(indices[ModeType.EXTENDED]),
        "harmonic": len(indices[ModeType.HARMONIC]),
        "free": len(indices[ModeType.FREE]),
        "frozen": len(indices[ModeType.FROZEN]),
        "sigma": len(indices[ModeType.CYCLIC]),
    }


def _coordinate_result(
    matrices: CircuitMatrices,
    structural: CoordinateSystem,
    full_transformation: sp.MatrixBase,
) -> CoordinateSystem:
    full = sp.Matrix(full_transformation)
    dynamical_count = structural.dynamical_count
    dynamical = full[:, :dynamical_count]
    nondynamical = full[:, dynamical_count:]
    capacitance = sp.simplify(
        dynamical.T * matrices.capacitance_internal * dynamical
    )
    susceptance = sp.simplify(
        dynamical.T * matrices.susceptance_internal * dynamical
    )
    for name, form in (
        ("capacitance", matrices.capacitance_internal),
        ("susceptance", matrices.susceptance_internal),
    ):
        cross = sp.simplify(dynamical.T * form * nondynamical)
        if cross != sp.zeros(dynamical.cols, nondynamical.cols):
            raise ValueError(
                f"Polynomial optimizer left a nonzero {name} dynamical/"
                f"non-dynamical block: {cross}"
            )
    return CoordinateSystem(
        transformation=sp.ImmutableMatrix(dynamical),
        full_transformation=sp.ImmutableMatrix(full),
        mode_indices=structural.mode_indices,
        full_mode_indices=structural.full_mode_indices,
        capacitance=sp.ImmutableMatrix(capacitance),
        inverse_capacitance=sp.ImmutableMatrix(sp.simplify(capacitance.inv())),
        susceptance=sp.ImmutableMatrix(susceptance),
        junction_incidence_transformed=sp.ImmutableMatrix(
            sp.simplify(matrices.omega_junction_internal.T * dynamical)
        ),
        inductor_incidence_transformed=sp.ImmutableMatrix(
            sp.simplify(matrices.omega_inductor_internal.T * dynamical)
        ),
        strategy="canonical",
    )


def _ground_transformation(
    transformation: sp.MatrixBase,
    ground_row: int,
    sigma_indices: list[int],
) -> sp.Matrix:
    """Gauge-fix the optimizer's ungrounded result without touching its solver."""
    rows = [row for row in range(transformation.rows) if row != ground_row]
    # Grounding node fluxes is phi_i -> phi_i - phi_ground. Merely deleting
    # the ground row changes branch phases and can mix a harmonic coordinate
    # into the Josephson potential.
    grounded = sp.Matrix.vstack(
        *(transformation[row, :] - transformation[ground_row, :] for row in rows)
    )
    for sigma_index in sigma_indices:
        columns = [
            column for column in range(transformation.cols) if column != sigma_index
        ]
        reduced = grounded[:, columns]
        if (
            grounded[:, sigma_index].is_zero_matrix
            and reduced.rows == reduced.cols
            and reduced.rank() == reduced.rows
        ):
            return reduced
    raise ValueError("Could not remove the optimizer's global sigma gauge mode")


def canonicalize(
    matrices: CircuitMatrices,
    structural: CoordinateSystem,
    **solver_options,
) -> CoordinateSystem:
    """Run the complete sircuitenum polynomial optimizer on current inputs."""
    topology = _legacy_topology(matrices)
    try:
        from ._vendor.quantize import choose_Z
    except (ImportError, ValueError) as exc:
        raise OptimizationDependencyError(
            "Canonical optimization requires SageMath, Singular, Z3, NumPy, "
            "pandas, tqdm, and func-timeout. The cqed environment provides them."
        ) from exc
    transformations, variable_types, _ = choose_Z(
        list(topology.circuit),
        list(topology.edges),
        ground_node=[],
        return_instance=True,
        **solver_options,
    )
    if not transformations:
        raise ValueError("Polynomial optimizer returned no coordinate transformation")
    expected = _mode_counts(structural.full_mode_indices)
    actual = {name: len(variable_types.get(name, [])) for name in expected}
    grounded_actual = {**actual, "sigma": actual["sigma"] - 1}
    if grounded_actual != expected:
        raise ValueError(
            "Vendored optimizer and structural classifier disagree on mode counts: "
            f"grounded optimizer={grounded_actual}, structural={expected}"
        )
    grounded = _ground_transformation(
        transformations[0], topology.ground_nodes[0], variable_types["sigma"]
    )
    result = _coordinate_result(matrices, structural, grounded)
    harmonic = list(result.mode_indices[ModeType.HARMONIC])
    if harmonic and result.junction_incidence_transformed[:, harmonic] != sp.zeros(
        result.junction_incidence_transformed.rows, len(harmonic)
    ):
        raise ValueError(
            "Canonical optimization mixed a harmonic coordinate into the "
            "Josephson potential"
        )
    return result


def optimize_coordinates(circuit: Circuit, **solver_options) -> CoordinateSystem:
    """Optimize coordinates directly from the public branch-centric input model."""
    from ..coordinates import choose_coordinates

    matrices = assemble(circuit)
    structural = choose_coordinates(matrices, strategy="structural")
    return canonicalize(matrices, structural, **solver_options)


__all__ = [
    "LegacyTopology",
    "OptimizationDependencyError",
    "canonicalize",
    "optimize_coordinates",
]
