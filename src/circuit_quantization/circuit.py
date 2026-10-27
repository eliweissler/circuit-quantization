"""Circuit description and incidence matrix (Sec I.A, I.B)."""

import re
from dataclasses import dataclass, field

import networkx as nx
import sympy as sym

from .units import Phi0, e, ghz_to_farad, ghz_to_henry, h, hbar, h_GHz

# Branch order of eq:Omega_columns.
TYPES = ("C", "J", "L", "V")
INDUCTIVE = ("J", "L")


def _variables(values):
    """Unique free symbols in value order, excluding physical constants."""
    constants = {e, h, hbar, Phi0, h_GHz}
    return tuple(dict.fromkeys(
        symbol for value in values if value is not None
        for symbol in sorted(sym.sympify(value).free_symbols - constants, key=sym.default_sort_key)
    ))

# Generate symbols for branch parameters
def parameter(name, energy_symbols=False):
    """Symbol for an omitted value; the first letter of name selects C, J (for E_J), L, or V.

    C1 -> C_1, CJ1 -> C_J1, J1 -> E_J1, L1 -> L_1, V1 -> V_1.
    energy_symbols turns C and L into GHz energies E_C1, E_CJ1, E_L1, converted to SI.
    """
    kind = name[0]
    if kind == "J" or (energy_symbols and kind in ("C", "L")):
        symbol = sym.Symbol(f"E_{name}", positive=True)
    else:
        label = f"{kind}_{name[1:]}" if name[1:] else kind
        symbol = sym.Symbol(label, **({"real": True} if kind == "V" else {"positive": True}))
    if energy_symbols and kind == "C":
        return ghz_to_farad(symbol)
    if energy_symbols and kind == "L":
        return ghz_to_henry(symbol)
    return symbol

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

    :param str name: Unique branch label. With all values omitted, its first letter
        selects C, J, L, or V and its suffix names the generated symbols.
    :param tail: Hashable label of the positive-incidence endpoint.
    :param head: Hashable label of the negative-incidence endpoint.
    :param C: Capacitance in farads. Omitted junction capacitance is generated;
        omitted inductor/source capacitance is zero.
    :param L: Inductance in henries, or None if not an inductor.
    :param EJ: Josephson energy E/h in GHz, or None if not a junction.
    :param V: Source-node voltage relative to ground, in volts, or None.
    :param bool energy_symbols: Generate capacitance/inductance energy symbols in
        GHz and convert internally to SI. Does not change explicit input units.
    :raises ValueError: Invalid element values, incompatible types, or a self-loop.
    """

    name: str
    tail: object
    head: object
    C: object = None
    L: object = None
    EJ: object = None
    V: object = None
    energy_symbols: bool = field(default=False, kw_only=True)

    # Make any necessary symbols and verify valid branches
    def __post_init__(self):
        self._set_default_values()
        self._check_branch()

    def _set_default_values(self):
        """Generate symbols for omitted values, then convert floats to exact rationals."""
        name = self.name[0] + self.name[1:].lstrip("_") if self.name else ""  # J_1 -> J1
        if self.C is None and self.L is None and self.EJ is None and self.V is None:
            if name[:1] not in TYPES:
                raise ValueError("A branch without values needs a name starting with C, J, L, or V")
            attr = {"C": "C", "J": "EJ", "L": "L", "V": "V"}[name[0]]
            object.__setattr__(self, attr, parameter(name, self.energy_symbols))
        # Omitted C is generated for a junction and zero for an inductor or source.
        if self.C is None:
            object.__setattr__(self, "C", parameter("C" + name, self.energy_symbols) if self.type == "J" else 0)
        for attr in ("C", "L", "EJ", "V"):
            value = getattr(self, attr)
            if value is not None:
                object.__setattr__(self, attr, sym.nsimplify(sym.sympify(value), rational=True))

    def _check_branch(self):
        # No self loops
        if self.tail == self.head:
            raise ValueError(f"Branch {self.name!r} is a self-loop")
        # Voltage sources only have V
        if self.V is not None and (self.C != 0 or self.L is not None or self.EJ is not None):
            raise ValueError(f"Source branch {self.name!r} may carry only V")
        # At most one of L, EJ
        if self.L is not None and self.EJ is not None:
            raise ValueError(f"Branch {self.name!r} may carry at most one of L, EJ")
        # Valid capacitance, inductance, and Josephson energy values; only explicit numbers are
        # checked, so symbolic values such as C_J*(1 + epsilon) are accepted as given.
        if self.C.is_number and self.C.is_negative:
            raise ValueError(f"Branch {self.name!r}: C must be 0 or positive")
        if self.type in ("C", "J") and self.C == 0:
            raise ValueError(f"Branch {self.name!r}: a {self.type} branch needs C > 0")
        for attr in ("L", "EJ"):
            value = getattr(self, attr)
            if value is not None and value.is_number and not value.is_positive:
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


def _graph(circuit):
    """Undirected multigraph keyed by branch name, so parallel branches stay distinct."""
    G = nx.MultiGraph()
    for i,b in enumerate(circuit.branches):
        G.add_edge(b.tail, b.head, key=b.name, inductive=b.type in INDUCTIVE, source=b.type == "V", has_capacitance =b.C != 0,
                   rank=i + len(circuit.branches) * (b.type not in INDUCTIVE) + len(circuit.branches) * (b.type == "V"))  # J, L before C then V then branch order as tiebreaker
    return G


@dataclass(frozen=True)
class Circuit:
    """Connected branches plus an optional reference ground.

    Nodes are ordered int | gnd | src (Sec I.B); a floating circuit has no gnd row.

    :param branches: Iterable of Branch objects forming one connected graph.
    :param ground: Label of a node on some branch, or None for a floating circuit.
    :param t: SymPy time symbol used for time-dependent fluxes and voltages.
    :raises ValueError: Empty/disconnected graph, duplicate labels, or invalid source connections.
    """

    branches: tuple
    ground: object = None
    t: sym.Symbol = field(default=sym.Symbol("t", real=True))
    graph: nx.MultiGraph = field(init=False, repr=False, compare=False)
    source_nodes: tuple = field(init=False, repr=False)
    internal_nodes: tuple = field(init=False, repr=False)

    # Verify valid circuit and make graph representation
    def __post_init__(self):
        self._sort_nodes_and_branches()
        object.__setattr__(self, "graph", _graph(self))
        self._check_circuit()

    def _sort_nodes_and_branches(self):
        # Sort by type C | J | L | V, keeping user order within a type (eq:Omega_columns).
        # And seperate internal nodes from source nodes, sorting by natural string within each sector.
        branches = tuple(sorted(self.branches, key=lambda b: TYPES.index(b.type)))
        all_nodes = {b.tail for b in branches} | {b.head for b in branches}
        sources = []
        for b in branches:
            if b.type == "V":
                sources.append(b.head if b.tail == self.ground else b.tail)
        internal = [n for n in all_nodes if n not in sources and n != self.ground]
        object.__setattr__(self, "source_nodes", tuple(_ordered(sources)))
        object.__setattr__(self, "internal_nodes", tuple(_ordered(internal)))
        object.__setattr__(self, "branches", branches)

    def _check_circuit(self):
        # Nonempty connected graph, ground on it, at least one internal node
        if not self.branches or not nx.is_connected(self.graph):
            raise ValueError("Circuit branches must form a nonempty connected graph")
        if self.ground is not None and self.ground not in self.graph:
            raise ValueError(f"Ground {self.ground!r} is on no branch; use ground=None for a floating circuit")
        if not self.internal_nodes:
            raise ValueError("Circuit has no internal nodes")
        # Distinct string labels, so the natural node order is well defined
        if len({str(n) for n in self.graph}) != len(self.graph):
            raise ValueError("Two different nodes have the same string label, e.g. 1 and '1'")
        # Branches have unique names
        branches = self.branches
        if len(set(self.branch_names)) != len(branches):
            raise ValueError("Branch names must be unique")
        # Source nodes must connect to ground and carry no other branches
        # Source nodes must only be connected to internal nodes by capacitors
        sources = []
        for b in branches:
            if b.type == "V":
                if self.ground not in (b.tail, b.head):
                    raise ValueError(f"Source {b.name!r} must connect to ground")
                sources.append(b.head if b.tail == self.ground else b.tail)
        if len(set(sources)) != len(sources):
            raise ValueError("Each source node carries exactly one source")
        for b in branches:
            if b.type in INDUCTIVE and {b.tail, b.head} & set(sources):
                raise ValueError(f"Only capacitors may end on a source node: {b.name!r}")

    @property
    def nodes(self):
        ground = () if self.ground is None else (self.ground,)
        return self.internal_nodes + ground + self.source_nodes

    @property
    def branch_names(self):
        return tuple(b.name for b in self.branches)

    @property
    def variables(self):
        """Unique parameter symbols in branch order, then each branch's value order."""
        return tuple(dict.fromkeys(v for b in self.branches for v in b.variables))


def incidence(circuit, *, elem=None):
    """:param circuit: Circuit whose ordered nodes and branches define the matrix.
    :param elem: Optional branch type to select a submatrix - C, J, L, or V (Sec I.B, tab:elements).
    :returns: Immutable integer matrix with node rows and branch columns.
    Omega: +1 at each branch's tail and -1 at its head (Sec I.B).
    """
    row = {node: i for i, node in enumerate(circuit.nodes)}
    branches = [b for b in circuit.branches if elem is None or b.type == elem]
    Omega = sym.zeros(len(circuit.nodes), len(branches))
    for k, b in enumerate(branches):
        Omega[row[b.tail], k] = 1
        Omega[row[b.head], k] = -1
    return sym.ImmutableMatrix(Omega)
