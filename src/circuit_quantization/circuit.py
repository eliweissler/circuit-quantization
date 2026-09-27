"""Circuit description, incidence matrix, and cycle basis (Sec I)."""

from dataclasses import dataclass, field

import networkx as nx
import sympy as sym

# Branch order of eq:Omega_columns.
TYPES = ("C", "J", "L", "V")


@dataclass(frozen=True)
class Branch:
    """One oriented branch, tail -> head (Sec I.A, tab:elements).

    C [F] is an attribute any branch may carry.
    At most one of L [H], EJ [GHz], or V [V] makes it an inductor, junction, or source.
    V is the source node's potential relative to ground.
    Floats are converted to exact rationals, so identities the notes prove
    (G B = I, vanishing cross blocks) hold exactly.
    """

    name: str
    tail: object
    head: object
    C: object = 0
    L: object = None
    EJ: object = None
    V: object = None

    def __post_init__(self):
        for attr in ("C", "L", "EJ", "V"):
            value = getattr(self, attr)
            if value is not None:
                object.__setattr__(self, attr, sym.nsimplify(sym.sympify(value), rational=True))
        if self.tail == self.head:
            raise ValueError(f"Branch {self.name!r} is a self-loop")
        if sum(x is not None for x in (self.L, self.EJ, self.V)) > 1:
            raise ValueError(f"Branch {self.name!r} may carry at most one of L, EJ, V")
        if not (self.C == 0 or self.C.is_positive):
            raise ValueError(f"Branch {self.name!r}: C must be 0 or provably positive")
        if self.type in ("C", "J") and self.C == 0:
            raise ValueError(f"Branch {self.name!r}: a {self.type} branch needs C > 0")
        if self.type == "V" and self.C != 0:
            raise ValueError(f"Source branch {self.name!r} must have C = 0")
        for attr in ("L", "EJ"):
            value = getattr(self, attr)
            if value is not None and value.is_positive is False:
                raise ValueError(f"Branch {self.name!r}: {attr} must be positive")

    @property
    def type(self):
        if self.EJ is not None:
            return "J"
        if self.L is not None:
            return "L"
        if self.V is not None:
            return "V"
        return "C"


def _ordered(nodes):
    nodes = list(dict.fromkeys(nodes))
    try:
        return sorted(nodes)
    except TypeError:
        return nodes


@dataclass(frozen=True)
class Circuit:
    """Branches plus a ground node, with nodes ordered int | gnd | src (Sec I.B)."""

    branches: tuple
    ground: object
    t: sym.Symbol = field(default=sym.Symbol("t", real=True))

    def __post_init__(self):
        branches = tuple(self.branches)
        names = [b.name for b in branches]
        if len(set(names)) != len(names):
            raise ValueError("Branch names must be unique")
        sources = []
        for b in branches:
            if b.type == "V":
                if self.ground not in (b.tail, b.head):
                    raise ValueError(f"Source {b.name!r} must connect to ground")
                sources.append(b.head if b.tail == self.ground else b.tail)
        if len(set(sources)) != len(sources):
            raise ValueError("Each source node carries exactly one source")
        for b in branches:
            if b.type in ("J", "L") and {b.tail, b.head} & set(sources):
                raise ValueError(f"Only capacitors may end on a source node: {b.name!r}")
        # Sort by type C | J | L | V, keeping user order within a type (eq:Omega_columns).
        branches = tuple(sorted(branches, key=lambda b: TYPES.index(b.type)))
        object.__setattr__(self, "branches", branches)
        object.__setattr__(self, "source_nodes", tuple(_ordered(sources)))
        others = [n for b in branches for n in (b.tail, b.head)]
        internal = [n for n in _ordered(others) if n != self.ground and n not in sources]
        if not internal:
            raise ValueError("Circuit has no internal nodes")
        object.__setattr__(self, "internal_nodes", tuple(internal))

    @property
    def nodes(self):
        return self.internal_nodes + (self.ground,) + self.source_nodes

    @property
    def branch_names(self):
        return tuple(b.name for b in self.branches)


def incidence(circuit):
    """Omega: +1 at each branch's tail and -1 at its head (Sec I.B)."""
    row = {node: i for i, node in enumerate(circuit.nodes)}
    Omega = sym.zeros(len(circuit.nodes), len(circuit.branches))
    for k, b in enumerate(circuit.branches):
        Omega[row[b.tail], k] = 1
        Omega[row[b.head], k] = -1
    return sym.ImmutableMatrix(Omega)


def _graph(circuit):
    graph = nx.MultiGraph()
    graph.add_nodes_from(circuit.nodes)
    graph.add_edges_from((b.tail, b.head) for b in circuit.branches)
    return graph


def default_tree(circuit):
    """A spanning forest grown over S_ind (J, L) first, then S_cap (C, V).

    Growing S_ind first makes the closure rows inside S_ind span that
    subgraph's cycle space, so G comes out ordered as in eq:G_blocks.
    """
    forest = nx.utils.UnionFind(circuit.nodes)
    tree = []
    for b in sorted(circuit.branches, key=lambda b: b.type not in ("J", "L")):
        if forest[b.tail] != forest[b.head]:
            forest.union(b.tail, b.head)
            tree.append(b.name)
    return tree


def fundamental_cycles(circuit, tree=None):
    """Fundamental cycles of a spanning tree, as signed branch dicts (Appendix A).

    Returns {closure branch name: {branch name: +1 or -1}}, superconducting
    cycles first (eq:G_blocks).
    """
    names = circuit.branch_names
    tree = default_tree(circuit) if tree is None else list(tree)
    if set(tree) - set(names):
        raise ValueError(f"Unknown tree branches: {sorted(set(tree) - set(names))}")
    # Delete one root row per connected component, the ground in its own.
    roots = set()
    for component in nx.connected_components(_graph(circuit)):
        roots.add(circuit.ground if circuit.ground in component
                  else next(n for n in circuit.nodes if n in component))
    keep = [i for i, n in enumerate(circuit.nodes) if n not in roots]
    tree_cols = [names.index(name) for name in tree]
    closure_cols = sorted(
        (k for k in range(len(names)) if names[k] not in tree),
        key=lambda k: circuit.branches[k].type not in ("J", "L"),
    )
    Omega = incidence(circuit)
    Omega_t = Omega.extract(keep, tree_cols)
    Omega_c = Omega.extract(keep, closure_cols)
    if Omega_t.rows != Omega_t.cols or Omega_t.det() == 0:
        raise ValueError("tree is not a spanning forest of the circuit")
    G_t = -Omega_c.T * Omega_t.T.inv()  # eq:tree_B, Appendix A
    cycles = {}
    for row, k in enumerate(closure_cols):
        cycle = {names[j]: int(G_t[row, col]) for col, j in enumerate(tree_cols) if G_t[row, col] != 0}
        cycle[names[k]] = 1
        cycles[names[k]] = cycle
    return cycles


class Loops:
    """A complete cycle basis G and the external flux through each cycle (Sec I.C).

    Every cycle not given a value in ``fluxes`` gets a visible symbol Phi_<name> [Wb].
    Pass 0 explicitly for a cycle that encloses no flux.
    """

    def __init__(self, circuit, cycles, fluxes=None):
        fluxes = dict(fluxes or {})
        names = circuit.branch_names
        if set(fluxes) - set(cycles):
            raise ValueError(f"Fluxes for unknown cycles: {sorted(set(fluxes) - set(cycles))}")
        rows = []
        for loop, cycle in cycles.items():
            if set(cycle) - set(names):
                raise ValueError(f"Cycle {loop!r} uses unknown branches: {sorted(set(cycle) - set(names))}")
            rows.append([cycle.get(name, 0) for name in names])
        G = sym.Matrix(rows) if rows else sym.zeros(0, len(names))
        Omega = incidence(circuit)
        for loop, row in zip(cycles, rows):
            if any(sym.Matrix([row]) * Omega.T):
                raise ValueError(f"Cycle {loop!r} is not a graph cycle (eq:G_OmegaT)")
        if G.rank() < G.rows:
            raise ValueError("Cycles are linearly dependent")
        N_ell = len(names) - Omega.rank()
        if G.rows < N_ell:
            raise ValueError(f"{N_ell - G.rows} cycle(s) missing: a complete basis has {N_ell}")
        self.circuit = circuit
        self.names = tuple(cycles)
        self.G = sym.ImmutableMatrix(G)
        self.Phi_ext = sym.ImmutableMatrix(
            len(self.names), 1,
            [fluxes.get(loop, sym.Symbol(f"Phi_{loop}", real=True)) for loop in self.names],
        )

    def rebase(self, G_new, names=None):
        """Re-express the same fluxes in another cycle basis G_new = T G (eq:G_basis_change)."""
        G_new = sym.Matrix(G_new)
        G = sym.Matrix(self.G)
        T = G_new * G.T * (G * G.T).inv()
        if T * G != G_new:
            raise ValueError("G_new rows are not in the cycle space")
        if any(not x.is_integer for x in T) or abs(T.det()) != 1:
            raise ValueError("G_new = T G needs T integer with det T = +-1")
        names = names or tuple(f"loop_{i + 1}" for i in range(G_new.rows))
        branch_names = self.circuit.branch_names
        cycles = {
            loop: {branch_names[k]: int(G_new[i, k]) for k in range(G_new.cols) if G_new[i, k] != 0}
            for i, loop in enumerate(names)
        }
        return Loops(self.circuit, cycles, dict(zip(names, T * self.Phi_ext)))
