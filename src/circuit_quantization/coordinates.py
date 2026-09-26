"""Mode classification and structure-preserving coordinate construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterable

import sympy as sp

from .matrices import CircuitMatrices


class ModeType(str, Enum):
    COMPACT = "compact"
    EXTENDED = "extended"
    HARMONIC = "harmonic"
    FREE = "free"
    FROZEN = "frozen"
    CYCLIC = "cyclic"


@dataclass(frozen=True)
class CoordinateSystem:
    """A retained dynamical basis and its auditable full construction."""

    transformation: sp.ImmutableMatrix
    full_transformation: sp.ImmutableMatrix
    mode_indices: object
    full_mode_indices: object
    capacitance: sp.ImmutableMatrix
    inverse_capacitance: sp.ImmutableMatrix
    susceptance: sp.ImmutableMatrix
    junction_incidence_transformed: sp.ImmutableMatrix
    inductor_incidence_transformed: sp.ImmutableMatrix
    strategy: str

    @property
    def dynamical_count(self) -> int:
        return self.transformation.cols


def _columns(matrix: sp.MatrixBase) -> list[sp.Matrix]:
    return [sp.Matrix(matrix[:, index]) for index in range(matrix.cols)]


def _matrix_from_columns(columns: Iterable[sp.Matrix], rows: int) -> sp.Matrix:
    columns = list(columns)
    return sp.Matrix.hstack(*columns) if columns else sp.zeros(rows, 0)


def _independent_columns(columns: Iterable[sp.Matrix]) -> list[sp.Matrix]:
    result: list[sp.Matrix] = []
    for column in columns:
        column = sp.simplify(column)
        if column.is_zero_matrix:
            continue
        before = _matrix_from_columns(result, column.rows).rank()
        after = _matrix_from_columns(result + [column], column.rows).rank()
        if after > before:
            result.append(column)
    return result


def _intersection(
    left: Iterable[sp.Matrix], right: Iterable[sp.Matrix], rows: int
) -> list[sp.Matrix]:
    left = _independent_columns(left)
    right = _independent_columns(right)
    if not left or not right:
        return []
    left_matrix = _matrix_from_columns(left, rows)
    right_matrix = _matrix_from_columns(right, rows)
    null = sp.Matrix.hstack(left_matrix, -right_matrix).nullspace()
    vectors = [sp.simplify(left_matrix * vector[: len(left), :]) for vector in null]
    return _independent_columns(vectors)


def _independent_from(
    candidates: Iterable[sp.Matrix], support: Iterable[sp.Matrix], rows: int
) -> list[sp.Matrix]:
    current = _independent_columns(support)
    result: list[sp.Matrix] = []
    rank = _matrix_from_columns(current, rows).rank()
    for candidate in _independent_columns(candidates):
        trial = _matrix_from_columns(current + result + [candidate], rows)
        if trial.rank() > rank + len(result):
            result.append(candidate)
    return result


def _normalize_compact(
    compact: list[sp.Matrix], omega_junction_internal: sp.MatrixBase, rows: int
) -> list[sp.Matrix]:
    if not compact:
        return []
    compact_matrix = _matrix_from_columns(compact, rows)
    junction_lattice = sp.Matrix(omega_junction_internal.T * compact_matrix)
    independent_rows = junction_lattice.T.rref()[1]
    if len(independent_rows) < len(compact):
        raise ValueError("Compact junction lattice does not have full column rank")
    square = junction_lattice[list(independent_rows[: len(compact)]), :]
    normalized = sp.simplify(compact_matrix * square.inv())
    selected = sp.simplify(omega_junction_internal.T * normalized)[
        list(independent_rows[: len(compact)]), :
    ]
    if selected != sp.eye(len(compact)):
        raise AssertionError("Failed to normalize the compact lattice")
    return _columns(normalized)


def _decoupling_transformation(matrix: sp.Matrix, dynamical_count: int) -> sp.Matrix:
    """One rank-aware lower-triangular decoupling step.

    This is the practical construction used by sircuitenum: only ND directions
    with support in the current trailing block are included in X22.
    """
    coupled = []
    for row in range(dynamical_count, matrix.rows):
        if any(sp.simplify(matrix[row, col]) != 0 for col in range(dynamical_count, matrix.cols)):
            coupled.append(row)
    if not coupled:
        return sp.eye(matrix.rows)

    x22 = matrix.extract(coupled, coupled)
    x21 = matrix.extract(coupled, range(dynamical_count))
    try:
        correction = -x22.inv() * x21
    except Exception as exc:
        raise ValueError(
            "Coupled non-dynamical subblock remains singular during sequential "
            f"decoupling; indices={coupled}, block={x22}"
        ) from exc
    transform = sp.eye(matrix.rows)
    for local_row, row in enumerate(coupled):
        for column in range(dynamical_count):
            transform[row, column] = correction[local_row, column]
    return transform


def decouple_column(
    column: sp.Matrix, nondynamical: sp.Matrix, bilinear_form: sp.MatrixBase
) -> sp.Matrix:
    """Decouple one column using the reverse one-ND-mode-at-a-time sweep."""
    if nondynamical.cols == 0:
        return sp.Matrix(column)
    basis = sp.Matrix.hstack(column, nondynamical)
    for dynamical_count in range(nondynamical.cols, 0, -1):
        transformed = sp.simplify(basis.T * bilinear_form * basis)
        basis = sp.simplify(
            basis * _decoupling_transformation(transformed, dynamical_count)
        )
    return sp.simplify(basis[:, 0])


def _classify(matrices: CircuitMatrices):
    n = len(matrices.internal_nodes)
    null_j = [sp.Matrix(v) for v in sp.Matrix(matrices.omega_junction_internal.T).nullspace()]
    null_l = [sp.Matrix(v) for v in sp.Matrix(matrices.susceptance_internal).nullspace()]
    null_c = [sp.Matrix(v) for v in sp.Matrix(matrices.capacitance_internal).nullspace()]

    cyclic = _intersection(null_l, _intersection(null_j, null_c, n), n)
    free = _independent_from(_intersection(null_j, null_l, n), cyclic, n)
    frozen = _independent_from(_intersection(null_j, null_c, n), cyclic, n)
    nd_support = cyclic + free + frozen

    harmonic = _independent_from(null_j, nd_support, n)
    compact = _independent_from(null_l, nd_support, n)
    compact = _normalize_compact(compact, matrices.omega_junction_internal, n)
    extended = _independent_from(
        _columns(sp.eye(n)), nd_support + harmonic + compact, n
    )

    all_columns = compact + extended + harmonic + free + frozen + cyclic
    if len(all_columns) != n or _matrix_from_columns(all_columns, n).rank() != n:
        raise ValueError("Mode classification did not produce a complete basis")
    return {
        ModeType.COMPACT: compact,
        ModeType.EXTENDED: extended,
        ModeType.HARMONIC: harmonic,
        ModeType.FREE: free,
        ModeType.FROZEN: frozen,
        ModeType.CYCLIC: cyclic,
    }


def _indices(mode_columns, included):
    result = {}
    offset = 0
    for mode in included:
        size = len(mode_columns[mode])
        result[mode] = tuple(range(offset, offset + size))
        offset += size
    return MappingProxyType(result)


def choose_coordinates(
    matrices: CircuitMatrices, strategy: str = "structural"
) -> CoordinateSystem:
    """Construct parameter-independent circuit coordinates.

    ``structural`` is always available.  ``canonical`` is an optional layer
    over the structural result and is imported lazily.
    """
    if strategy not in {"structural", "canonical"}:
        raise ValueError("strategy must be 'structural' or 'canonical'")
    mode_columns = _classify(matrices)
    n = len(matrices.internal_nodes)
    nd_modes = (ModeType.FREE, ModeType.FROZEN, ModeType.CYCLIC)
    dyn_modes = (ModeType.COMPACT, ModeType.EXTENDED, ModeType.HARMONIC)
    nondynamical = _matrix_from_columns(
        [column for mode in nd_modes for column in mode_columns[mode]], n
    )

    corrected = {mode: [] for mode in dyn_modes}
    for mode in dyn_modes:
        for column in mode_columns[mode]:
            column = decouple_column(
                column, nondynamical, matrices.capacitance_internal
            )
            if mode in (ModeType.EXTENDED, ModeType.HARMONIC):
                column = decouple_column(
                    column, nondynamical, matrices.susceptance_internal
                )
            corrected[mode].append(sp.simplify(column))

    dynamical = _matrix_from_columns(
        [column for mode in dyn_modes for column in corrected[mode]], n
    )
    if dynamical.cols == 0:
        raise ValueError("Circuit contains no dynamical modes")
    full = sp.Matrix.hstack(dynamical, nondynamical)

    for form_name, form in (
        ("capacitance", matrices.capacitance_internal),
        ("susceptance", matrices.susceptance_internal),
    ):
        cross = sp.simplify(dynamical.T * form * nondynamical)
        if cross != sp.zeros(dynamical.cols, nondynamical.cols):
            raise ValueError(
                f"Sequential decoupling left a nonzero {form_name} cross block: {cross}"
            )
    junction_nd = sp.simplify(matrices.omega_junction_internal.T * nondynamical)
    if junction_nd != sp.zeros(junction_nd.rows, junction_nd.cols):
        raise ValueError("Junction potential depends on a non-dynamical coordinate")

    capacitance = sp.simplify(
        dynamical.T * matrices.capacitance_internal * dynamical
    )
    if capacitance.det().equals(0):
        raise ValueError("Retained dynamical capacitance matrix is singular")
    inverse_capacitance = sp.simplify(capacitance.inv())
    susceptance = sp.simplify(
        dynamical.T * matrices.susceptance_internal * dynamical
    )

    mode_indices = _indices(corrected, dyn_modes)
    full_columns = {
        **corrected,
        ModeType.FREE: mode_columns[ModeType.FREE],
        ModeType.FROZEN: mode_columns[ModeType.FROZEN],
        ModeType.CYCLIC: mode_columns[ModeType.CYCLIC],
    }
    full_mode_indices = _indices(full_columns, dyn_modes + nd_modes)

    result = CoordinateSystem(
        transformation=sp.ImmutableMatrix(dynamical),
        full_transformation=sp.ImmutableMatrix(full),
        mode_indices=mode_indices,
        full_mode_indices=full_mode_indices,
        capacitance=sp.ImmutableMatrix(capacitance),
        inverse_capacitance=sp.ImmutableMatrix(inverse_capacitance),
        susceptance=sp.ImmutableMatrix(susceptance),
        junction_incidence_transformed=sp.ImmutableMatrix(
            sp.simplify(matrices.omega_junction_internal.T * dynamical)
        ),
        inductor_incidence_transformed=sp.ImmutableMatrix(
            sp.simplify(matrices.omega_inductor_internal.T * dynamical)
        ),
        strategy="structural",
    )
    if strategy == "canonical":
        from ._canonical import canonicalize

        return canonicalize(matrices, result)
    return result


def coordinates_from_transformation(
    matrices: CircuitMatrices,
    transformation: sp.MatrixBase,
    mode_types: Iterable[ModeType | str],
) -> CoordinateSystem:
    """Construct coordinates from an explicit square node-flux transformation.

    ``transformation`` implements ``phi_node = Z phi_new`` in the internal-node
    order reported by :class:`CircuitMatrices`. Explicit coordinates currently
    describe a fully dynamical circuit, so ``Z`` must be square and invertible.
    """
    z = sp.Matrix(transformation)
    node_count = len(matrices.internal_nodes)
    if z.shape != (node_count, node_count):
        raise ValueError(
            "An explicit transformation must be square with one row per "
            f"internal node; expected {(node_count, node_count)}, got {z.shape}"
        )
    if z.det().equals(0):
        raise ValueError("Explicit coordinate transformation is singular")

    parsed_types = tuple(ModeType(mode_type) for mode_type in mode_types)
    if len(parsed_types) != node_count:
        raise ValueError("mode_types must contain one entry per explicit coordinate")
    dynamical_modes = (ModeType.COMPACT, ModeType.EXTENDED, ModeType.HARMONIC)
    invalid = [mode for mode in parsed_types if mode not in dynamical_modes]
    if invalid:
        raise ValueError(
            "Explicit coordinates may only be compact, extended, or harmonic; "
            f"got {invalid}"
        )
    indices = MappingProxyType(
        {
            mode: tuple(
                index for index, coordinate_type in enumerate(parsed_types)
                if coordinate_type == mode
            )
            for mode in dynamical_modes
        }
    )

    capacitance = sp.simplify(z.T * matrices.capacitance_internal * z)
    if capacitance.det().equals(0):
        raise ValueError("Explicit coordinates have a singular capacitance matrix")
    susceptance = sp.simplify(z.T * matrices.susceptance_internal * z)
    junction = sp.simplify(matrices.omega_junction_internal.T * z)
    inductor = sp.simplify(matrices.omega_inductor_internal.T * z)

    compact = list(indices[ModeType.COMPACT])
    if compact and inductor[:, compact] != sp.zeros(inductor.rows, len(compact)):
        raise ValueError("A coordinate declared compact enters an inductive branch")
    harmonic = list(indices[ModeType.HARMONIC])
    if harmonic and junction[:, harmonic] != sp.zeros(junction.rows, len(harmonic)):
        raise ValueError("A coordinate declared harmonic enters a Josephson branch")

    return CoordinateSystem(
        transformation=sp.ImmutableMatrix(z),
        full_transformation=sp.ImmutableMatrix(z),
        mode_indices=indices,
        full_mode_indices=indices,
        capacitance=sp.ImmutableMatrix(capacitance),
        inverse_capacitance=sp.ImmutableMatrix(sp.simplify(capacitance.inv())),
        susceptance=sp.ImmutableMatrix(susceptance),
        junction_incidence_transformed=sp.ImmutableMatrix(junction),
        inductor_incidence_transformed=sp.ImmutableMatrix(inductor),
        strategy="explicit",
    )
