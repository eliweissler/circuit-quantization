"""Cycle bases and external fluxes (Sec I.C, Appendix A)."""

from dataclasses import InitVar, dataclass, field
from itertools import combinations, product

import networkx as nx
import sympy as sym

from .circuit import Circuit, _variables, incidence


def spanning_trees(circuit):
    """Yield each spanning tree as a tuple of branch names in circuit order; parallel branches stay distinct."""
    for tree in nx.SpanningTreeIterator(circuit.graph):
        names = {name for _, _, name in tree.edges(keys=True)}
        yield tuple(name for name in circuit.branch_names if name in names)


def default_tree(circuit):
    """The minimum spanning tree by branch rank: S_ind (J, L) before C before V, then circuit order.

    Growing S_ind first makes the closure rows inside S_ind span that
    subgraph's cycle space, so G comes out ordered as in eq:G_blocks.
    """
    return [name for _, _, name in nx.minimum_spanning_edges(circuit.graph, weight="rank", keys=True, data=False)]


def tree_to_G(circuit, tree):
    """Closure branches and fundamental cycles G = (G_t I) of a spanning tree (Appendix A, eq:A1).

    With the root row deleted, G_t = -Omega_c^T (Omega_t^T)^{-1}; the root is the
    ground, or the first node of a floating circuit. Closure branches come in rank
    order, so superconducting cycles come first (eq:G_blocks).

    :param tree: Branch labels of a spanning tree, e.g. default_tree(circuit).
    :returns: Pair (closure branch names, G) with G columns in circuit.branch_names order.
    """
    names = circuit.branch_names
    rank = {name: r for _, _, name, r in circuit.graph.edges(keys=True, data="rank")}
    closure = tuple(sorted((name for name in names if name not in tree), key=rank.get))
    root = circuit.nodes.index(circuit.ground) if circuit.ground is not None else 0
    Omega = incidence(circuit)
    Omega = Omega[[i for i in range(Omega.rows) if i != root], :]
    columns = [names.index(name) for name in (*tree, *closure)]  # (tree | closure) order
    Omega_t, Omega_c = Omega[:, columns[:len(tree)]], Omega[:, columns[len(tree):]]
    G = sym.Matrix.hstack(-Omega_c.T * Omega_t.T.inv(), sym.eye(len(closure)))  # eq:A1
    return closure, sym.ImmutableMatrix(G[:, [columns.index(k) for k in range(len(names))]])


def branch_dict_to_G(circuit, branch_dict):
    """Cycle matrix G from {loop name: branch labels}, one row per loop in dict order.

    A tuple of labels gets its signs from incidence, with the first label +1;
    a signed dict {label: +1 or -1} is used as given.

    :raises ValueError: Unknown labels, or a tuple that is not a single closed loop.
    """
    names = circuit.branch_names
    Omega = incidence(circuit)
    rows = []
    for loop, cycle in branch_dict.items():
        if set(cycle) - set(names):
            raise ValueError(f"Cycle {loop!r} uses unknown branches: {sorted(set(cycle) - set(names))}")
        if not isinstance(cycle, dict):
            basis = Omega[:, [names.index(name) for name in cycle]].nullspace()
            if len(cycle) != len(set(cycle)) or len(basis) != 1 or any(x == 0 for x in basis[0]):
                raise ValueError(f"Cycle {loop!r} needs a nonempty tuple of distinct branches forming a single closed loop")
            cycle = dict(zip(cycle, basis[0] / basis[0][0]))
        rows.append([cycle.get(name, 0) for name in names])
    return sym.ImmutableMatrix(len(rows), len(names), [x for row in rows for x in row])


def _closes(circuit, G, closure_branches):
    """True when G restricted to closure_branches is the identity, as for a tree basis (eq:A1)."""
    columns = [circuit.branch_names.index(name) for name in closure_branches]
    return G[:, columns] == sym.eye(G.rows)


def check_loops(circuit, G, names, closure_branches=None, fluxes=()):
    """Check that G is a cycle basis (Sec I.C) and that its labels fit.

    G needs N_ell = N_b - rank(Omega) linearly independent rows with entries
    0, +1, -1 that satisfy G Omega^T = 0 (eq:G_OmegaT). names labels each row;
    closure_branches, if given, are the identity columns of G; fluxes name existing loops.
    """
    Omega = incidence(circuit)
    N_ell = Omega.cols - Omega.rank()
    if G.shape != (N_ell, Omega.cols) or G.rank() != N_ell:
        raise ValueError(f"G needs N_ell = N_b - rank(Omega) = {N_ell} independent rows of {Omega.cols} entries")
    if any(x not in (-1, 0, 1) for x in G) or any(sym.simplify(G * Omega.T)):
        raise ValueError("G needs entries 0, +1, -1 (a physical loop) and G Omega^T = 0 (eq:G_OmegaT)")
    if len(names) != G.rows or len(set(names)) != len(names):
        raise ValueError(f"names needs one name per row of G ({G.rows}), all distinct")
    if set(fluxes) - set(names):
        raise ValueError(f"Fluxes for unknown loops: {sorted(set(fluxes) - set(names))}")
    if closure_branches is not None and not _closes(circuit, G, closure_branches):
        raise ValueError("closure_branches must be the identity columns of G")


def cycle_bases(circuit, *, primitive=True):
    """Yield each cycle basis made of simple loops, as a branch_dict for Loops.

    Rows are named loop_1, loop_2, ... in a canonical order; each loop is a tuple
    of branch names in circuit order, which Loops orients with its first branch positive.
    primitive=True keeps only integer cycle-lattice bases (eq:G_basis_change, det T = +-1);
    False keeps every full-rank basis. Tries every subset of simple loops; for small circuits.
    """
    graph, names = circuit.graph, circuit.branch_names
    tail = {b.name: b.tail for b in circuit.branches}
    loops = {}
    for nodes in nx.simple_cycles(graph):
        # Expand the node sequence over parallel branches; a two-node loop is a parallel pair.
        keys = [graph[u][v] for u, v in zip(nodes, nodes[1:] + nodes[:1])]
        for hops in combinations(keys[0], 2) if len(nodes) == 2 else product(*keys):
            signs = {name: 1 if tail[name] == u else -1 for u, name in zip(nodes, hops)}  # +1 along tail -> head
            loops[tuple(name for name in names if name in signs)] = signs
    # Any basis is G = T G_f, and G_f is the identity on closure columns, so T = G[:, closure].
    tree = default_tree(circuit)
    closure = [name for name in names if name not in tree]
    for basis in combinations(sorted(loops), len(closure)):
        det = sym.Matrix([[loops[loop].get(name, 0) for name in closure] for loop in basis]).det()
        if (abs(det) == 1) if primitive else (det != 0):
            yield {f"loop_{i + 1}": loop for i, loop in enumerate(basis)}


@dataclass(frozen=True)
class Loops:
    """A complete cycle basis G and the external flux through each cycle (Sec I.C).

    Every cycle not given a value in ``fluxes`` gets a visible symbol Phi_<name> [Wb].
    Pass 0 explicitly for a cycle that encloses no flux. closure_branches is set
    when G comes from a spanning tree (G restricted to them is the identity);
    exporting the loops to scqubits requires it.

    :param circuit: Circuit defining the branch labels and orientations.
    :param G: Cycle matrix with one row per loop and columns in circuit.branch_names order.
    :param names: One distinct label per row; None generates loop_1, loop_2, ... .
    :param closure_branches: Tree closure branch for each row, or None.
    :param fluxes: Optional {loop name: flux in Wb}; missing loops get real Phi_<name> symbols.
    :raises ValueError: See check_loops.
    """

    circuit: Circuit
    G: sym.ImmutableMatrix
    names: tuple = None
    closure_branches: tuple = None
    fluxes: InitVar[dict] = None
    Phi_ext: sym.ImmutableMatrix = field(init=False)

    def __post_init__(self, fluxes):
        fluxes = dict(fluxes or {})
        G = sym.ImmutableMatrix(self.G)
        names = tuple(self.names) if self.names is not None else tuple(f"loop_{i + 1}" for i in range(G.rows))
        closure = tuple(self.closure_branches) if self.closure_branches is not None else None
        check_loops(self.circuit, G, names, closure, fluxes)
        Phi_ext = [fluxes.get(loop, sym.Symbol(f"Phi_{loop}", real=True)) for loop in names]
        object.__setattr__(self, "G", G)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "closure_branches", closure)
        object.__setattr__(self, "Phi_ext", sym.ImmutableMatrix(len(names), 1, Phi_ext))

    @classmethod
    def from_branch_dict(cls, circuit, branch_dict, fluxes=None):
        """Loops named and ordered as branch_dict; see branch_dict_to_G."""
        return cls(circuit, branch_dict_to_G(circuit, branch_dict), tuple(branch_dict), fluxes=fluxes)

    @classmethod
    def from_tree(cls, circuit, tree, fluxes=None):
        """Fundamental loops of a spanning tree, each named after its closure branch (eq:A1)."""
        closure, G = tree_to_G(circuit, tree)
        return cls(circuit, G, closure, closure, fluxes=fluxes)

    @classmethod
    def from_circuit(cls, circuit, *, zero_capacitive=False):
        """Generate and print fundamental cycles from the default spanning tree.

        The deterministic tree prioritizes J/L over C/V branches in circuit
        order. Each loop is named after its closure branch, whose sign is +1.
        Prints signed branches, classification, and flux, one loop per line.
        For custom cycles or flux values, use :meth:`from_branch_dict` or
        :meth:`from_tree`.

        :param circuit: Circuit whose fundamental cycles are generated.
        :param bool zero_capacitive: If True, set every loop containing a C or V
            branch to zero flux. Otherwise generate a real Phi_<name> symbol
            for every loop. Junction/inductor capacitance does not count as
            a capacitively broken branch. Default False.
        :returns: Loops with G, names, closure_branches, Phi_ext, and variables.
        """
        closure, G = tree_to_G(circuit, default_tree(circuit))
        capacitive = [k for k, b in enumerate(circuit.branches) if b.type in ("C", "V")]
        broken = {loop for i, loop in enumerate(closure) if any(G[i, k] for k in capacitive)}
        result = cls(circuit, G, closure, closure, fluxes=dict.fromkeys(broken, 0) if zero_capacitive else {})
        for loop, flux in zip(result.names, result.Phi_ext):
            elements = " ".join(f"{'+' if sign == 1 else '-'}{name}" for name, sign in result.branch_dict[loop].items())
            kind = "capacitively broken" if loop in broken else "superconducting"
            print(f"{loop}: {elements} [{kind}]; flux = {flux}")
        if not closure:
            print("No loops.")
        return result

    @property
    def branch_dict(self):
        """{loop name: {branch label: +1 or -1}}, closure branch first when known, then circuit order.

        Signed, so from_branch_dict(circuit, loops.branch_dict) rebuilds the same G.
        """
        closure = self.closure_branches or (None,) * len(self.names)
        branch_dict = {}
        for loop, first, row in zip(self.names, closure, self.G.tolist()):
            signs = {first: 1} if first is not None else {}  # a closure branch is +1 in its own row
            for name, sign in zip(self.circuit.branch_names, row):
                if sign and name != first:
                    signs[name] = sign
            branch_dict[loop] = signs
        return branch_dict

    @property
    def variables(self):
        """Unique external-flux symbols in loop order, including explicit expressions."""
        return _variables(self.Phi_ext)

    def rebase(self, G_new, names=None):
        """Re-express the same fluxes in another cycle basis G_new = T G (eq:G_basis_change).

        closure_branches carry over only while G_new is still the identity on them.

        :param G_new: Complete cycle matrix in the same branch-column order as G.
            Must be related to G by an integer unimodular transformation.
        :param names: One unique label per new row; None generates loop_1, loop_2, ... .
        :returns: New Loops with transformed fluxes, preserving their physical meaning.
        :raises ValueError: Invalid basis, non-unimodular transformation, or invalid names.
        """
        G_new = sym.ImmutableMatrix(G_new)
        names = tuple(names) if names is not None else tuple(f"loop_{i + 1}" for i in range(G_new.rows))
        check_loops(self.circuit, G_new, names)  # G_new is a cycle basis, so G_new = T G exactly
        T = G_new * self.G.T * (self.G * self.G.T).inv()
        if any(not x.is_integer for x in T) or abs(T.det()) != 1:
            raise ValueError("G_new = T G needs T integer with det T = +-1")
        closure = self.closure_branches
        if closure is not None and not _closes(self.circuit, G_new, closure):
            closure = None
        return Loops(self.circuit, G_new, names, closure, fluxes=dict(zip(names, T * self.Phi_ext)))
