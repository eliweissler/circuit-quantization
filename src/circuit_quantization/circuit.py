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
                object.__setattr__(self, attr, sym.sympify(value))
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
