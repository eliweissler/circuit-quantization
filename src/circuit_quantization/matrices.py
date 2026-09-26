"""Construction of graph, loop, and element matrices."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import networkx as nx
import sympy as sp

from .model import Branch, Circuit


class IncompleteLoopBasisWarning(UserWarning):
    """A supplied physical-loop list was completed with zero-flux cycles."""


@dataclass(frozen=True)
class CircuitMatrices:
    circuit: Circuit
    branches: tuple[Branch, ...]
    nodes: tuple[object, ...]
    internal_nodes: tuple[object, ...]
    source_nodes: tuple[object, ...]
    omega: sp.ImmutableMatrix
    omega_internal: sp.ImmutableMatrix
    omega_ground: sp.ImmutableMatrix
    omega_source: sp.ImmutableMatrix
    capacitor_indices: tuple[int, ...]
    junction_indices: tuple[int, ...]
    inductor_indices: tuple[int, ...]
    source_indices: tuple[int, ...]
    capacitance_branch: sp.ImmutableMatrix
    capacitance_full: sp.ImmutableMatrix
    capacitance_internal: sp.ImmutableMatrix
    capacitance_internal_source: sp.ImmutableMatrix
    susceptance_full: sp.ImmutableMatrix
    susceptance_internal: sp.ImmutableMatrix
    omega_junction_internal: sp.ImmutableMatrix
    omega_inductor_internal: sp.ImmutableMatrix
    inductance_branch_inverse: sp.ImmutableMatrix
    source_voltages: sp.ImmutableMatrix
    loop_matrix: sp.ImmutableMatrix
    loop_names: tuple[str, ...]
    external_fluxes: sp.ImmutableMatrix

    @property
    def cycle_rank(self) -> int:
        return self.loop_matrix.rows


def _sorted_nodes(nodes):
    return tuple(sorted(nodes, key=lambda item: (type(item).__name__, repr(item))))


def _column_subset(matrix: sp.MatrixBase, indices: tuple[int, ...]) -> sp.Matrix:
    if not indices:
        return sp.zeros(matrix.rows, 0)
    return matrix[:, list(indices)]


def _fundamental_cycle_rows(
    branches: tuple[Branch, ...], nodes: tuple[object, ...]
) -> list[sp.Matrix]:
    """Return a deterministic fundamental-cycle basis as row vectors."""
    parent = {node: node for node in nodes}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return False
        parent[right_root] = left_root
        return True

    tree = nx.Graph()
    tree.add_nodes_from(nodes)
    tree_edges: dict[frozenset, int] = {}
    closure_indices: list[int] = []
    for index, branch in enumerate(branches):
        if union(branch.tail, branch.head):
            tree.add_edge(branch.tail, branch.head)
            tree_edges[frozenset((branch.tail, branch.head))] = index
        else:
            closure_indices.append(index)

    rows: list[sp.Matrix] = []
    for closure_index in closure_indices:
        closure = branches[closure_index]
        row = sp.zeros(1, len(branches))
        row[0, closure_index] = 1
        path = nx.shortest_path(tree, closure.head, closure.tail)
        for start, stop in zip(path, path[1:]):
            edge_index = tree_edges[frozenset((start, stop))]
            edge = branches[edge_index]
            row[0, edge_index] = 1 if (edge.tail, edge.head) == (start, stop) else -1
        rows.append(row)
    return rows


def _loop_basis(circuit: Circuit, omega: sp.Matrix, nodes: tuple[object, ...]):
    branches = circuit.branches
    branch_index = {branch.name: index for index, branch in enumerate(branches)}
    graph = nx.MultiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from((branch.tail, branch.head) for branch in branches)
    cycle_rank = len(branches) - len(nodes) + nx.number_connected_components(graph)

    rows: list[sp.Matrix] = []
    names: list[str] = []
    values: list[sp.Expr] = []
    for loop in circuit.loop_fluxes:
        unknown = set(loop.cycle).difference(branch_index)
        if unknown:
            raise ValueError(
                f"Loop {loop.name!r} references unknown branches: {sorted(unknown)}"
            )
        row = sp.zeros(1, len(branches))
        for name, orientation in loop.cycle.items():
            row[0, branch_index[name]] = orientation
        if row * omega.T != sp.zeros(1, omega.rows):
            raise ValueError(f"Loop {loop.name!r} is not an oriented graph cycle")
        candidate = sp.Matrix.vstack(*rows, row) if rows else row
        if candidate.rank() != len(rows) + 1:
            raise ValueError(f"Loop {loop.name!r} is dependent on earlier loops")
        rows.append(row)
        names.append(loop.name)
        values.append(loop.value)

    if len(rows) > cycle_rank:
        raise ValueError("More independent loops were supplied than the graph supports")

    if rows and len(rows) < cycle_rank:
        warnings.warn(
            f"Supplied {len(rows)} of {cycle_rank} independent loops; "
            "completing the basis with zero-flux cycles",
            IncompleteLoopBasisWarning,
            stacklevel=3,
        )

    for fundamental_index, row in enumerate(
        _fundamental_cycle_rows(branches, nodes), start=1
    ):
        if len(rows) == cycle_rank:
            break
        candidate = sp.Matrix.vstack(*rows, row) if rows else row
        if candidate.rank() == len(rows) + 1:
            rows.append(row)
            names.append(f"cycle_{fundamental_index}")
            values.append(sp.S.Zero)

    if len(rows) != cycle_rank:
        raise ValueError("Could not construct a complete graph-cycle basis")
    loop_matrix = sp.Matrix.vstack(*rows) if rows else sp.zeros(0, len(branches))
    if loop_matrix * omega.T != sp.zeros(cycle_rank, omega.rows):
        raise AssertionError("Internal error: generated loop basis is inconsistent")
    flux_vector = sp.Matrix(values) if values else sp.zeros(0, 1)
    return loop_matrix, tuple(names), flux_vector


def assemble(circuit: Circuit) -> CircuitMatrices:
    """Assemble exact graph and SI element matrices for ``circuit``."""
    branches = circuit.branches
    source_nodes_by_branch = []
    source_voltage_by_node = {}
    for branch in branches:
        if branch.kind != "source":
            continue
        source_node = branch.head if branch.tail == circuit.ground else branch.tail
        source_nodes_by_branch.append(source_node)
        source_voltage_by_node[source_node] = branch.voltage
    source_nodes = _sorted_nodes(source_nodes_by_branch)

    all_nodes = {circuit.ground}
    for branch in branches:
        all_nodes.update((branch.tail, branch.head))
    internal_nodes = _sorted_nodes(all_nodes.difference({circuit.ground}, source_nodes))
    if not internal_nodes:
        raise ValueError("Circuit has no internal nodes to quantize")
    nodes = internal_nodes + (circuit.ground,) + source_nodes
    node_index = {node: index for index, node in enumerate(nodes)}

    omega = sp.zeros(len(nodes), len(branches))
    for column, branch in enumerate(branches):
        omega[node_index[branch.tail], column] = 1
        omega[node_index[branch.head], column] = -1

    n_internal = len(internal_nodes)
    ground_index = n_internal
    omega_internal = omega[:n_internal, :]
    omega_ground = omega[ground_index : ground_index + 1, :]
    omega_source = omega[ground_index + 1 :, :]

    capacitor_indices = tuple(i for i, branch in enumerate(branches) if branch.kind == "capacitor")
    junction_indices = tuple(i for i, branch in enumerate(branches) if branch.kind == "junction")
    inductor_indices = tuple(i for i, branch in enumerate(branches) if branch.kind == "inductor")
    source_indices = tuple(i for i, branch in enumerate(branches) if branch.kind == "source")

    capacitance_branch = sp.diag(*(branch.capacitance for branch in branches))
    capacitance_full = omega * capacitance_branch * omega.T
    capacitance_internal = capacitance_full[:n_internal, :n_internal]
    # The note writes the full off-diagonal block as -C_int,src.
    capacitance_internal_source = -capacitance_full[
        :n_internal, ground_index + 1 :
    ]

    omega_inductor = _column_subset(omega, inductor_indices)
    inductance_branch_inverse = sp.diag(
        *(1 / branches[index].inductance for index in inductor_indices)
    ) if inductor_indices else sp.zeros(0, 0)
    susceptance_full = (
        omega_inductor * inductance_branch_inverse * omega_inductor.T
        if inductor_indices
        else sp.zeros(len(nodes), len(nodes))
    )
    susceptance_internal = susceptance_full[:n_internal, :n_internal]

    omega_junction_internal = _column_subset(omega_internal, junction_indices)
    omega_inductor_internal = _column_subset(omega_internal, inductor_indices)
    source_voltages = (
        sp.Matrix([source_voltage_by_node[node] for node in source_nodes])
        if source_nodes
        else sp.zeros(0, 1)
    )
    loop_matrix, loop_names, external_fluxes = _loop_basis(circuit, omega, nodes)

    freeze = sp.ImmutableMatrix
    return CircuitMatrices(
        circuit=circuit,
        branches=branches,
        nodes=nodes,
        internal_nodes=internal_nodes,
        source_nodes=source_nodes,
        omega=freeze(omega),
        omega_internal=freeze(omega_internal),
        omega_ground=freeze(omega_ground),
        omega_source=freeze(omega_source),
        capacitor_indices=capacitor_indices,
        junction_indices=junction_indices,
        inductor_indices=inductor_indices,
        source_indices=source_indices,
        capacitance_branch=freeze(capacitance_branch),
        capacitance_full=freeze(capacitance_full),
        capacitance_internal=freeze(capacitance_internal),
        capacitance_internal_source=freeze(capacitance_internal_source),
        susceptance_full=freeze(susceptance_full),
        susceptance_internal=freeze(susceptance_internal),
        omega_junction_internal=freeze(omega_junction_internal),
        omega_inductor_internal=freeze(omega_inductor_internal),
        inductance_branch_inverse=freeze(inductance_branch_inverse),
        source_voltages=freeze(source_voltages),
        loop_matrix=freeze(loop_matrix),
        loop_names=loop_names,
        external_fluxes=freeze(external_fluxes),
    )
