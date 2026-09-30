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

    :param branches: Iterable of Branch objects forming one connected graph.
    :param ground: Reference-node label; may be absent from branches for a floating circuit.
    :param t: SymPy time symbol used for time-dependent fluxes and voltages.
    :raises ValueError: Empty/disconnected graph, duplicate labels, or invalid source connections.
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
    """Omega: +1 at each branch's tail and -1 at its head (Sec I.B).

    :param circuit: Circuit whose ordered nodes and branches define the matrix.
    :returns: Immutable integer matrix with node rows and branch columns.
    """
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

    :param circuit: Circuit to enumerate. Parallel branches remain distinct.
    :yields: Tuples of branch labels defining spanning trees; isolated ground is retained.
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

    :param circuit: Circuit defining branch orientations and order.
    :param tree: Complete iterable of branch labels forming a spanning tree.
        None selects a deterministic tree, prioritizing J/L over C/V branches.
    :returns: Dict mapping each closure branch to a signed branch dictionary.
        Closure coefficients are +1; superconducting cycles precede broken cycles.
    :raises ValueError: Unknown labels or a selection that is not a spanning tree.
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
    branch_dict maps loop names to tuples of branch labels. Signs are inferred
    from incidence, with the first listed branch positive. Explicit signed
    dictionaries are also accepted, including for rebasing existing fluxes.
    """

    def __init__(self, circuit, branch_dict, fluxes=None):
        """Construct an explicit complete cycle basis and its external fluxes.

        :param circuit: Circuit defining the branch labels and orientations.
        :param branch_dict: Complete mapping of loop names to branch-label
            tuples. Incidence fixes relative signs; the first label is +1.
            Alternatively, supply signed branch dictionaries with values +/-1.
        :param fluxes: Optional mapping from loop names to flux expressions in
            webers. Missing entries generate real Phi_<name> symbols; use 0
            explicitly for zero flux. Keys must belong to branch_dict.
        :raises ValueError: Unknown labels, invalid/incomplete cycles, dependent
            rows, or fluxes for unknown loops.
        """
        fluxes = dict(fluxes or {})
        names = circuit.branch_names
        Omega = incidence(circuit)
        if set(fluxes) - set(branch_dict):
            raise ValueError(f"Fluxes for unknown branch_dict: {sorted(set(fluxes) - set(branch_dict))}")
        rows = []
        for loop, cycle in branch_dict.items():
            if set(cycle) - set(names):
                raise ValueError(f"Cycle {loop!r} uses unknown branches: {sorted(set(cycle) - set(names))}")
            if not isinstance(cycle, dict):
                if not isinstance(cycle, (tuple, list)) or not cycle or len(set(cycle)) != len(cycle):
                    raise ValueError(f"Cycle {loop!r} needs distinct branch labels in a nonempty tuple")
                basis = Omega[:, [names.index(name) for name in cycle]].nullspace()
                if len(basis) != 1 or any(x == 0 for x in basis[0]):
                    raise ValueError(f"Cycle {loop!r} must describe a single closed loop")
                signs = basis[0] / basis[0][0]
                cycle = dict(zip(cycle, signs))
            if any(value not in (-1, 0, 1) for value in cycle.values()):
                # G in {0, +1, -1}: a flux is only meaningful through a physical loop (Sec I.C).
                raise ValueError(f"Cycle {loop!r} is not a physical loop: coefficients must be 0, +1, or -1")
            rows.append([cycle.get(name, 0) for name in names])
        G = sym.Matrix(rows) if rows else sym.zeros(0, len(names))
        for loop, row in zip(branch_dict, rows):
            if any(sym.Matrix([row]) * Omega.T):
                raise ValueError(f"Cycle {loop!r} is not a graph cycle (eq:G_OmegaT)")
        if G.rank() < G.rows:
            raise ValueError("branch_dict are linearly dependent")
        N_ell = len(names) - Omega.rank()
        if G.rows < N_ell:
            raise ValueError(f"{N_ell - G.rows} cycle(s) missing: a complete basis has {N_ell}")
        self.circuit = circuit
        self.names = tuple(branch_dict)
        self.G = sym.ImmutableMatrix(G)
        # Put a positive branch first so this tuple representation preserves G.
        self.branch_dict = {
            loop: (tuple(branch_dict[loop]) if not isinstance(branch_dict[loop], dict) else
                   tuple(names[k] for k in sorted(
                       (k for k, value in enumerate(row) if value), key=lambda k: bool(row[k] < 0)
                   ))) for loop, row in zip(self.names, rows)
        }
        self.Phi_ext = sym.ImmutableMatrix(
            len(self.names), 1,
            [fluxes.get(loop, sym.Symbol(f"Phi_{loop}", real=True)) for loop in self.names],
        )

    @classmethod
    def from_circuit(cls, circuit, *, zero_capacitive=False):
        """Generate and print fundamental cycles from the default spanning tree.

        The deterministic tree prioritizes J/L over C/V branches in circuit
        order. Each loop is named after its closure branch, whose sign is +1.
        Prints signed branches, classification, and flux, one loop per line.
        For custom cycles or flux values, use :class:`Loops` directly; for a
        custom tree, obtain its cycles with :func:`fundamental_cycles` first.

        :param circuit: Circuit whose fundamental cycles are generated.
        :param bool zero_capacitive: If True, set every loop containing a C or V
            branch to zero flux. Otherwise generate a real Phi_<name> symbol
            for every loop. Junction/inductor capacitance does not count as
            a capacitively broken branch. Default False.
        :returns: Loops with generated branch_dict, G, Phi_ext, and variables.
        """
        cycles = fundamental_cycles(circuit)
        branch_dict = {closure: (closure, *(name for name in cycle if name != closure))
                       for closure, cycle in cycles.items()}
        capacitive = {b.name for b in circuit.branches if b.type in ("C", "V")}
        broken = {loop for loop, labels in branch_dict.items() if capacitive.intersection(labels)}
        values = dict.fromkeys(broken, 0) if zero_capacitive else {}
        result = cls(circuit, branch_dict, values)
        for loop, flux in zip(result.names, result.Phi_ext):
            elements = " ".join(f"{'+' if cycles[loop][name] == 1 else '-'}{name}"
                                for name in branch_dict[loop])
            kind = "capacitively broken" if loop in broken else "superconducting"
            print(f"{loop}: {elements} [{kind}]; flux = {flux}")
        if not branch_dict:
            print("No loops.")
        return result

    @property
    def variables(self):
        """Unique external-flux symbols in loop order, including explicit expressions."""
        return _variables(self.Phi_ext)

    def rebase(self, G_new, names=None):
        """Re-express the same fluxes in another cycle basis G_new = T G (eq:G_basis_change).

        :param G_new: Complete cycle matrix in the same branch-column order as G.
            Must be related to G by an integer unimodular transformation.
        :param names: One unique label per new row; None generates loop_1, loop_2, ... .
        :returns: New Loops with transformed fluxes, preserving their physical meaning.
        :raises ValueError: Invalid basis, non-unimodular transformation, or invalid names.
        """
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
