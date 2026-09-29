"""Circuit description, incidence matrix, and cycle basis (Sec I)."""

import re
from dataclasses import dataclass, field
from itertools import combinations

import networkx as nx
import sympy as sym

from .units import Phi0, e, ghz_to_farad, ghz_to_henry, h, hbar, h_GHz

# Branch order of eq:Omega_columns.
TYPES = ("C", "J", "L", "V")


def _variables(values):
    """Unique free symbols in value order, excluding physical constants."""
    constants = {e, h, hbar, Phi0, h_GHz}
    return tuple(dict.fromkeys(
        symbol for value in values if value is not None
        for symbol in sorted(sym.sympify(value).free_symbols - constants, key=sym.default_sort_key)
    ))


@dataclass(frozen=True)
class Branch:
    """One oriented branch, tail -> head (Sec I.A, tab:elements).

    C [F] is an attribute any branch may carry.
    At most one of L [H], EJ [GHz], or V [V] makes it an inductor, junction, or source.
    V is the source node's potential relative to ground.
    Floats are converted to exact rationals, so identities the notes prove
    (G B = I, vanishing cross blocks) hold exactly.

    With no values supplied, infer C/J/L/V from the name's first letter and
    generate symbols: C1 -> C_1, L1 -> L_1, J1 -> (C_J1, E_J1), V1 -> V_1.
    Omitted junction C is generated even when EJ is supplied. Inductor C defaults
    to zero. energy_symbols=True generates E_C1, E_CJ1, E_L1 instead, converting
    these GHz energies to SI internally. Explicit values always retain the
    units above. All generated symbols are positive except real-valued voltages.
    """

    name: str
    tail: object
    head: object
    C: object = None
    L: object = None
    EJ: object = None
    V: object = None
    energy_symbols: bool = field(default=False, kw_only=True)

    def __post_init__(self):
        name = self.name[0] + self.name[1:].lstrip("_") if self.name else ""

        def parameter(kind):
            label = ("C" + name) if kind == "C" and name[:1] != "C" else name
            energy = kind == "EJ" or (self.energy_symbols and kind in ("C", "L"))
            label = f"E_{label}" if energy else label[:1] + ("_" + label[1:] if label[1:] else "")
            symbol = sym.Symbol(label, **({"real": True} if kind == "V" else {"positive": True}))
            if self.energy_symbols and kind in ("C", "L"):
                return (ghz_to_farad if kind == "C" else ghz_to_henry)(symbol)
            return symbol

        if all(value is None for value in (self.C, self.L, self.EJ, self.V)):
            if name[:1] not in TYPES:
                raise ValueError("A branch without values needs a name starting with C, J, L, or V")
            attr = {"C": "C", "J": "EJ", "L": "L", "V": "V"}[name[0]]
            object.__setattr__(self, attr, parameter(attr))
        if self.C is None:
            object.__setattr__(self, "C", parameter("C") if self.type in ("C", "J") else 0)
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
    def variables(self):
        """Parameter symbols in C, L, EJ, V order; numeric values contribute none."""
        return _variables((self.C, self.L, self.EJ, self.V))

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
    """Distinct nodes in natural order of their string labels: 2 < 10, "q2" < "q10"."""

    def key(node):
        return [(0, int(part), "") if part.isdigit() else (1, 0, part)
                for part in re.split(r"(\d+)", str(node)) if part]

    return sorted(dict.fromkeys(nodes), key=key)


@dataclass(frozen=True)
class Circuit:
    """Connected branches plus a reference ground, which may be isolated for a floating circuit.

    Nodes are ordered int | gnd | src (Sec I.B).
    """

    branches: tuple
    ground: object
    t: sym.Symbol = field(default=sym.Symbol("t", real=True))

    def __post_init__(self):
        branches = tuple(self.branches)
        if not branches or not nx.is_connected(nx.Graph((b.tail, b.head) for b in branches)):
            raise ValueError("Circuit branches must form a nonempty connected graph")
        names = [b.name for b in branches]
        if len(set(names)) != len(names):
            raise ValueError("Branch names must be unique")
        labels = {self.ground} | {n for b in branches for n in (b.tail, b.head)}
        if len({str(n) for n in labels}) != len(labels):
            raise ValueError("Two different nodes have the same string label, e.g. 1 and '1'")
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

    @property
    def variables(self):
        """Unique parameter symbols in branch order, then each branch's value order."""
        return tuple(dict.fromkeys(v for b in self.branches for v in b.variables))


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


def spanning_trees(circuit):
    """Yield tuples of branch names, in deterministic circuit order.

    For floating circuits, include the isolated reference ground as a singleton
    component of the spanning forest. Parallel branches remain distinct choices.
    Exhaustive subset enumeration is intended for small circuits.
    """
    size = len(circuit.nodes) - nx.number_connected_components(_graph(circuit))
    for branches in combinations(circuit.branches, size):
        forest = nx.utils.UnionFind(circuit.nodes)
        for b in branches:
            if forest[b.tail] == forest[b.head]:
                break
            forest.union(b.tail, b.head)
        else:
            yield tuple(b.name for b in branches)


def cycle_bases(circuit, *, primitive=True):
    """Planned enumeration of bases made from simple loops; not implemented.

    Will yield cycle dictionaries accepted by Loops, with canonical row order
    and orientation. Parallel branch identities and two-branch loops must be
    preserved. primitive=True will require an integer cycle-lattice basis;
    False will require only a full-rank real cycle-space basis.
    Non-fundamental bases have no associated tree or closure-branch labels.
    Use spanning_trees and fundamental_cycles for closure-labelled candidates.
    """
    raise NotImplementedError("Enumeration of simple-loop cycle bases is not implemented")


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
            if any(value not in (-1, 0, 1) for value in cycle.values()):
                # G in {0, +1, -1}: a flux is only meaningful through a physical loop (Sec I.C).
                raise ValueError(f"Cycle {loop!r} is not a physical loop: coefficients must be 0, +1, or -1")
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

    @property
    def variables(self):
        """Unique external-flux symbols in loop order, including explicit expressions."""
        return _variables(self.Phi_ext)

    def rebase(self, G_new, names=None):
        """Re-express the same fluxes in another cycle basis G_new = T G (eq:G_basis_change)."""
        G_new = sym.Matrix(G_new)
        G = sym.Matrix(self.G)
        T = G_new * G.T * (G * G.T).inv()
        if T * G != G_new:
            raise ValueError("G_new rows are not in the cycle space")
        if any(not x.is_integer for x in T) or abs(T.det()) != 1:
            raise ValueError("G_new = T G needs T integer with det T = +-1")
        names = tuple(names) if names is not None else tuple(f"loop_{i + 1}" for i in range(G_new.rows))
        if len(names) != G_new.rows or len(set(names)) != len(names):
            raise ValueError(f"names needs one name per row of G_new ({G_new.rows}), all distinct")
        branch_names = self.circuit.branch_names
        cycles = {
            loop: {branch_names[k]: G_new[i, k] for k in range(G_new.cols) if G_new[i, k] != 0}
            for i, loop in enumerate(names)
        }
        return Loops(self.circuit, cycles, dict(zip(names, T * self.Phi_ext)))
