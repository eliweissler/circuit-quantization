"""Mode classification, decoupling, and residual freedom (Sec III).

Mode types are the superscripts of eq:Z_mode_blocks:
"c" compact, "e" extended, "h" harmonic (dynamical) and
"fr" free, "fz" frozen, "cy" cyclic (non-dynamical).
"""

from dataclasses import dataclass
from functools import reduce
from itertools import combinations
from math import gcd

import networkx as nx
import sympy as sym
import numpy as np
from scipy.linalg import eigh

DYNAMICAL = ("c", "e", "h")
NONDYNAMICAL = ("fr", "fz", "cy")
TYPES = DYNAMICAL + NONDYNAMICAL


@dataclass(frozen=True)
class Coordinates:
    Z_full: sym.ImmutableMatrix  # full decoupled basis, including non-dynamical columns
    types_full: tuple
    Z: sym.ImmutableMatrix  # retained embedding: Phi_int = Z Phi_tilde + Z_2 Phi_disc
    R: sym.ImmutableMatrix  # retained extraction: Phi_tilde = R Phi_int, eq:Z_truncated
    types: tuple
    C_tilde: sym.ImmutableMatrix  # eq:Zd_schur
    C_tilde_inv: sym.ImmutableMatrix
    Lstar_tilde: sym.ImmutableMatrix
    omega_J_tilde: sym.ImmutableMatrix  # Z^T omega_J_int
    omega_L_tilde: sym.ImmutableMatrix  # Z^T omega_L_int


def _cancel(M):
    """Cancel each entry; the result is immutable."""
    return sym.ImmutableMatrix(M).applyfunc(sym.cancel)


def _quick_inv(M):
    """Inverse by cofactors for 2x2 to 4x4, where it beats elimination; M.inv() otherwise.

    3x3 and 4x4 use sym.inv_quick, whose determinants expand over permutations, so
    larger matrices fall back. 2x2 is written out: in sympy 1.14 inv_quick returns
    zeros for a 2x2 with any numeric entry, such as a 0 off the diagonal.
    """
    if M.rows == 2:
        (a, b), (c, d) = M.tolist()
        return sym.Matrix([[d, -b], [-c, a]]) / (a * d - b * c)
    return sym.inv_quick(M) if 3 <= M.rows <= 4 else M.inv()


# --- subspaces, each a matrix of basis columns ------------------------------


def _columns(vectors, n):
    return sym.Matrix.hstack(*vectors) if vectors else sym.zeros(n, 0)


def _null(A, n):
    return _columns(A.nullspace(), n)


def _perp(S):
    return _null(S.T, S.rows)


def _sum(A, B):
    return _columns(sym.Matrix.hstack(A, B).columnspace(), A.rows)


def _cap(A, B):
    return _perp(_sum(_perp(A), _perp(B)))


def _contains(S, v):
    return sym.Matrix.hstack(S, v).rank() == S.rank()


def subspaces(lag):
    """Column bases of the six mode subspaces of tab:var_basis.

    x^T C_int x = sum_k C_k (Omega_k^T x)^2 with C_k > 0, so N(C_int) is the
    nullspace of the capacitive columns of Omega_int, and likewise for L*_int.
    All three nullspaces therefore come from integer incidence matrices.
    """
    n = lag.Omega_int.rows
    capacitive = [k for k in range(lag.C_b.rows) if lag.C_b[k, k] != 0]
    J = _null(lag.omega_J_int.T, n)
    L = _null(lag.omega_L_int.T, n)
    C = _null(lag.Omega_int[:, capacitive].T, n)
    JL, CL = _cap(J, L), _cap(C, L)
    return {
        "c": _cap(L, _perp(JL)),
        "e": _perp(_sum(J, L)),
        "h": _cap(J, _perp(_sum(C, JL))),
        "fr": _cap(JL, _perp(CL)),
        "fz": _cap(C, _perp(CL)),
        "cy": CL,
        "J": J, "L": L, "C": C, "JL": JL,
    }


def _normalize_compact(Z_c, omega_J_int):
    """Give each compact coordinate period Phi0: S_cc = (omega_{J,K}^T)^{-1}, eq:Scc_selection."""
    if Z_c.cols == 0:
        return Z_c
    W = omega_J_int.T * Z_c
    K = list(W.T.rref()[1])  # independent rows of omega_{J,c}^T
    return Z_c * W.extract(K, list(range(W.cols))).inv()


def mode_basis(lag):
    """Square Z = (Z^c Z^e Z^h Z^fr Z^fz Z^cy) of eq:Z_mode_blocks and its column types.

    :param lag: Node Lagrangian matrices.
    :returns: Pair (Z_full, types), ordered c, e, h, fr, fz, cy.
    """
    sub = subspaces(lag)
    sub["c"] = _normalize_compact(sub["c"], lag.omega_J_int)
    Z = sym.Matrix.hstack(*(sub[t] for t in TYPES))
    types = tuple(t for t in TYPES for _ in range(sub[t].cols))
    return sym.ImmutableMatrix(Z), types


# --- validation ------------------------------------------------------------


def _is_integer(x):
    return sym.simplify(x).is_integer  # exact: nsimplify would round 1 + 10**-20 to 1


def mode_dimensions(lag):
    """Dimensions of the six subspaces of tab:var_basis, which are circuit invariants.

    These differ from the column types of a particular Z: the compact and
    non-dynamical counts are fixed, but the harmonic count is only a maximum,
    since a harmonic direction may sit inside extended columns (eq:S1_structure).

    :param lag: Node Lagrangian matrices.
    :returns: Dict of intrinsic dimensions keyed by c, e, h, fr, fz, cy.
    """
    sub = subspaces(lag)
    return {t: sub[t].cols for t in TYPES}


def check_coordinates(lag, Z_full, types):
    """Check a decoupled full basis against tab:var_basis and eq:compact_period_lattice.

    Decoupling changes neither the junction content of a column nor the
    non-dynamical columns, so passing here also validates the basis given to decouple.
    Every declared type must match mode_types, and the compact and non-dynamical
    counts must match mode_dimensions; together with invertibility this leaves
    at most dim h harmonic columns and a positive-definite C_tilde.
    """
    n = lag.Omega_int.rows
    if Z_full.shape != (n, n) or len(types) != n:
        raise ValueError(f"Z_full must be {n}x{n} with {n} types, including non-dynamical columns")
    if sym.simplify(Z_full.det()) == 0:
        raise ValueError("Z_full is singular")
    for j, (declared, inferred) in enumerate(zip(types, mode_types(lag, Z_full))):
        if declared != inferred:
            raise ValueError(f"Column {j} is not a valid {declared!r} direction (tab:var_basis)")
    dims = mode_dimensions(lag)
    for t in ("c", *NONDYNAMICAL):
        if types.count(t) != dims[t]:
            raise ValueError(f"Circuit has {dims[t]} {t!r} modes, Z_full declares {types.count(t)}")
    compact = [j for j, t in enumerate(types) if t == "c"]
    if compact:
        W = _cancel(lag.omega_J_int.T * Z_full[:, compact])  # omega_{J,c}^T
        if not all(_is_integer(x) for x in W):
            raise ValueError("omega_J,c^T must be integer (Sec IV.B.1)")
        minors = [sym.simplify(W.extract(list(rows), list(range(W.cols))).det()) for rows in combinations(range(W.rows), W.cols)]
        if reduce(gcd, (int(m) for m in minors), 0) != 1:
            raise ValueError("Compact periods are not primitive (eq:compact_period_lattice)")


def _check_residual(types, S):
    """S_1 is invertible, mixes only as eq:S1_structure allows, and S_cc lies in GL(N_c, Z)."""
    if S.shape != (len(types), len(types)) or sym.simplify(S.det()) == 0:
        raise ValueError("S1 must be square, invertible, and match the retained modes")
    for i in range(S.rows):
        for j in range(S.cols):
            if S[i, j] != 0 and (types[i], types[j]) not in _S1b:
                raise ValueError(f"S1[{i}, {j}] mixes {types[i]!r} into {types[j]!r} (eq:S1_structure)")
    compact = [j for j, t in enumerate(types) if t == "c"]
    S_cc = _cancel(S.extract(compact, compact))
    if compact and (not all(_is_integer(x) for x in S_cc) or abs(S_cc.det()) != 1):
        raise ValueError("S_cc must lie in GL(N_c, Z) (eq:Scc_unimodular)")


# --- decoupling and truncation ----------------------------------------------


def mode_types(lag, Z):
    """Classify each column by capacitive, junction, and inductor participation.

    :param lag: Node Lagrangian matrices.
    :param Z: Node-space columns, full or retained, in internal-node row order.
    :returns: Tuple of c, e, h, fr, fz, or cy labels in column order.
    """
    J, L, C = [_cancel(Z.T * mat).tolist() for mat in (lag.omega_J_int, lag.omega_L_int, lag.C_int)]
    return tuple(
        ("e" if any(l) else "c") if any(j) else
        ("h" if any(c) else "fz") if any(l) else
        ("fr" if any(c) else "cy")
        for j, l, c in zip(J, L, C)
    )


def coordinates(lag, Z_full, types=None, *, C_tilde=None, Lstar_tilde=None, R=None):
    """Build retained objects from a full decoupled basis (eq:Z_truncated).

    No validation; decouple runs check_coordinates first.

    :param lag: Node Lagrangian matrices.
    :param Z_full: Square decoupled node-to-coordinate matrix, including all
        non-dynamical columns. Use decouple first if the sectors are coupled.
    :param types: Optional labels for all columns; otherwise infer with mode_types.
    :param C_tilde: Optional Z^T C_int Z, when already known (e.g. diagonal by construction).
    :param Lstar_tilde: Optional Z^T L*_int Z, likewise.
    :param R: Optional retained rows of Z_full^{-1}, likewise.
    :returns: Coordinates retaining Z_full and types_full, with dynamical columns
        Z and corresponding rows R of the full inverse. Column order is preserved.
    :raises ValueError: No dynamical modes.
    """
    Z_full = sym.ImmutableMatrix(Z_full)
    types = mode_types(lag, Z_full) if types is None else tuple(types)
    retained = [i for i, t in enumerate(types) if t in DYNAMICAL]
    if not retained:
        raise ValueError("Circuit has no dynamical modes")
    Z = Z_full[:, retained]
    C_tilde = _cancel(Z.T * lag.C_int * Z if C_tilde is None else C_tilde)
    Lstar_tilde = _cancel(Z.T * lag.Lstar_int * Z if Lstar_tilde is None else Lstar_tilde)
    return Coordinates(
        Z_full=Z_full, types_full=types,
        Z=Z, types=tuple(types[i] for i in retained),
        R=_cancel(_quick_inv(Z_full)[retained, :] if R is None else R),  # eq:Z_truncated
        C_tilde=C_tilde,
        C_tilde_inv=_cancel(_quick_inv(C_tilde)),
        Lstar_tilde=Lstar_tilde,
        omega_J_tilde=Z.T * lag.omega_J_int, omega_L_tilde=Z.T * lag.omega_L_int,
    )


def _decouple_block(V, n_to_decouple, form):
    """Make every column of V form-orthogonal to the final n columns.

    Eliminate one n_to_decouple column at a time, in reverse order, updating all
    earlier columns, including the earlier n_to_decouple ones. A degenerate
    sector therefore needs no inverse.
    """
    k = V.cols
    Z = sym.Matrix.hstack(V, n_to_decouple)
    for i in range(Z.cols - 1, k - 1, -1):
        w = Z[:, i]
        pivot = sym.cancel((w.T * form * w)[0])
        if pivot != 0:  # A null direction of this positive-semidefinite form needs no correction.
            Z[:, :i] = (Z[:, :i] - w * (w.T * form * Z[:, :i]) / pivot).applyfunc(sym.cancel)
    return Z[:, :k]


def decouple(lag, Z_full, types, *, check=True):
    """Decouple the dynamical block, check, then truncate (eq:ZY_blocks, Sec III.B).

    Eliminate free directions in C and frozen directions in L* serially
    (eq:decouple_final, eq:decouple_final_L), updating the remaining columns.
    Cyclic directions are discarded; no discarded-sector matrix is inverted.

    :param lag: Node Lagrangian matrices.
    :param Z_full: Full square node-to-coordinate basis.
    :param types: Mode label for each basis column.
    :param bool check: Run check_coordinates on the decoupled basis. Skip only
        for a basis that is valid by construction, such as mode_basis.
    :returns: Coordinates containing the retained columns of Z_dc, corresponding
        rows of its full inverse R = P Z_dc^{-1}, and transformed matrices.
    :raises ValueError: An invalid basis or types, if check is True.
    """
    Z = sym.Matrix(Z_full)
    types = tuple(types)
    dynamical = [j for j, t in enumerate(types) if t in DYNAMICAL]
    Z_d = Z[:, dynamical]
    for kind, form in (("fr", lag.C_int), ("fz", lag.Lstar_int)):
        Z_d = _decouple_block(Z_d, Z[:, [j for j, t in enumerate(types) if t == kind]], form)
    for col, j in enumerate(dynamical):
        Z[:, j] = Z_d[:, col]
    Z = _cancel(Z)
    if check:
        check_coordinates(lag, Z, types)
    return coordinates(lag, Z, types)


# --- residual freedom -------------------------------------------------------

# (old type, new type) pairs a column may be mixed along, eq:S1_structure.
_S1a = {("c", "c"), ("c", "e"), ("e", "e"), ("h", "e"), ("h", "h")}
# Without preserving harmonic columns, any noncompact column may take extended or
# compact components; a harmonic column that takes a compact one becomes extended.
# Compact columns still take only compact components, so the lattice is kept.
_S1b = _S1a | {("e", "h"), ("c", "h")}


def residual_freedom(coords, keep_harmonic=True):
    """Symbolic S_1 with free entries where eq:S1_structure allows them.

    keep_harmonic=True gives S_{1,a}; False gives S_{1,b}, which lets harmonic
    columns acquire extended or compact components (making them extended).
    S_cc must also lie in GL(N_c, Z) (eq:Scc_unimodular); transform() checks
    that once values are chosen.

    :param coords: Retained Coordinates defining the current mode order.
    :param bool keep_harmonic: Preserve harmonic columns when True; allow
        extended/harmonic mixing when False.
    :returns: Symbolic matrix of allowed residual mixing parameters.
    """
    allowed = _S1a if keep_harmonic else _S1b
    types = coords.types
    return sym.Matrix(
        len(types), len(types),
        lambda i, j: sym.Symbol(f"s_{i + 1}_{j + 1}") if (types[i], types[j]) in allowed else 0,
    )


def junction_inversions(coords, types=("c", "e"), *, clean=True):
    """S_1 that make chosen junctions single-mode coordinates, one per choice of junctions.

    Compact ("c"): every set K of N_c junctions whose compact rows of omega~_J^T are
    independent gives S_cc = (omega~_{J,K}^T)^{-1} (eq:Scc_selection), kept when S_cc and
    omega~_{J,c}^T S_cc are integer (eq:Scc_unimodular). With clean, S_ce = -S_cc B, B the
    extended columns of those rows, also removes their extended components, so each
    junction in K acts on one compact mode.
    Extended ("e"): after the compact step, let r be the rank of the junctions' extended
    rows. Every set E of r junctions with independent extended rows A gives
    S_ee = (A^T (A A^T)^{-1}, N), N a basis of the extended directions no junction acts on,
    so each junction in E acts on one extended mode (keeping its compact components,
    which no allowed S_1 removes) and the N_e - r columns of N become harmonic.
    When r = N_e this is S_ee = A^{-1}.
    Requesting both yields every compact choice combined with every extended choice.

    :param coords: Retained Coordinates, in any column order.
    :param types: The blocks to invert: ("c",), ("e",), or ("c", "e").
    :param bool clean: Remove the extended components of the compact-aligned junctions.
    :returns: Generator of distinct S_1 in coords.types order, valid for transform.
    """
    if not types or set(types) - {"c", "e"}:
        raise ValueError(f"types must be a nonempty subset of ('c', 'e'), not {types!r}")
    n = len(coords.types)
    c = [j for j, t in enumerate(coords.types) if t == "c"]
    e = [j for j, t in enumerate(coords.types) if t == "e"]
    W = _cancel(coords.omega_J_tilde.T)  # one row per junction
    rows = range(W.rows)

    def compact_steps():
        if "c" not in types or not c:
            yield sym.eye(n)
            return
        W_c = W.extract(list(rows), c)
        for K in combinations(rows, len(c)):
            A = W_c.extract(list(K), list(range(len(c))))
            if A.det() == 0:
                continue
            S_cc = _quick_inv(A)
            if not all(_is_integer(x) for x in list(S_cc) + list(W_c * S_cc)):
                continue
            S = sym.eye(n)
            for a, i in enumerate(c):
                for b, j in enumerate(c):
                    S[i, j] = S_cc[a, b]
            if clean and e:
                S_ce = -S_cc * W.extract(list(K), e)
                for a, i in enumerate(c):
                    for b, j in enumerate(e):
                        S[i, j] = S_ce[a, b]
            yield S

    def extended_steps(S_c):
        if "e" not in types or not e:
            yield sym.eye(n)
            return
        W_e = _cancel(W * S_c).extract(list(rows), e)
        r = W_e.rank(simplify=True)
        N = _null(W_e, len(e))
        for E in combinations(rows, r):
            A = W_e.extract(list(E), list(range(len(e))))
            gram = _cancel(A * A.T)
            if sym.cancel(gram.det()) == 0:
                continue
            yield _embed(_cancel(sym.Matrix.hstack(A.T * _quick_inv(gram), N)), e, n)

    seen = set()
    for S_c in compact_steps():
        for S_e in extended_steps(S_c):
            S = _cancel(S_c * S_e)
            if S not in seen:
                seen.add(S)
                yield S


def _clusters(W, n):
    """Connected components of n modes, linked whenever one row of W acts on both."""
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for k in range(W.rows):
        acted = [j for j in range(n) if sym.cancel(W[k, j]) != 0]
        graph.add_edges_from(zip(acted, acted[1:]))
    return sorted(tuple(sorted(cluster)) for cluster in nx.connected_components(graph))


def junction_clusters(coords):
    """Modes grouped by junction coupling: connected components of the graph linking any
    two modes that share a junction, i.e. the nonzero pattern of the junction matrix.

    :param coords: Retained Coordinates.
    :returns: Sorted list of tuples of mode indices, in coords.types order.
    """
    return _clusters(coords.omega_J_tilde.T, len(coords.types))


CLUSTER_DIMS = {"c": 40, "e": 100, "h": 20}


def cluster_cost(coords, dims=None):
    """Hilbert-space dimension of each junction cluster, largest first; compare lexicographically.

    A cluster's dimension is the product of its modes' basis sizes, a proxy for the cost of
    diagonalizing it on its own, as in hierarchical diagonalization.

    :param coords: Retained Coordinates.
    :param dims: Basis size per mode type; defaults to CLUSTER_DIMS (c: 40 charge states,
        e: 100 phase-grid points, h: 20 oscillator states).
    :returns: Tuple of cluster dimensions in decreasing order.
    """
    return _cluster_cost(junction_clusters(coords), coords.types, dims)


def _cluster_cost(clusters, types, dims=None):
    dims = CLUSTER_DIMS | (dims or {})
    return tuple(sorted((reduce(lambda a, j: a * dims[types[j]], cluster, 1) for cluster in clusters), reverse=True))


def hierarchical_decoupling(coords, types=("c", "e"), dims=None, *, clean=True):
    """The junction inversion with the smallest junction clusters, for hierarchical diagonalization.

    Ranks the identity and every S_1 of junction_inversions by cluster_cost; the first
    of the cheapest wins. The cost is read from omega~_J^T S_1 without rebuilding the
    coordinates: a noncompact column no junction acts on is harmonic, as in mode_types,
    so an extended inversion can turn an extended column of a mixed basis harmonic.

    :param coords: Retained Coordinates, in any column order.
    :param types: Blocks to invert, as in junction_inversions.
    :param dims: Basis size per mode type, as in cluster_cost.
    :param bool clean: As in junction_inversions.
    :returns: (S_1, cost); apply S_1 with transform or Result.transform.
    """
    W = coords.omega_J_tilde.T
    n = len(coords.types)
    best = None
    for S in [sym.eye(n), *junction_inversions(coords, types, clean=clean)]:
        W_S = _cancel(W * S)
        new_types = tuple(t if t == "c" or any(W_S.col(j)) else "h" for j, t in enumerate(coords.types))
        cost = _cluster_cost(_clusters(W_S, n), new_types, dims)
        if best is None or cost < best[1]:
            best = (S, cost)
    return best


def compact_charge_decoupling(coords):
    """S_1 that removes the charge coupling between compact and noncompact modes exactly.

    Makes each noncompact column C~-orthogonal to the compact columns, the same
    elimination decouple applies to free modes. In (c, n) blocks this is
    S = [[I, -X], [0, I]] with C_cc X = C_cn, so S^T C~ S = diag(C_cc, M), with
    M = C_nn - C_nc X = ((C~^{-1})_nn)^{-1}, and S^T L~* S = L~*, since compact
    columns carry no inductance. Compact columns are unchanged, keeping their periods;
    noncompact columns gain compact components, so junction participation can spread
    and a harmonic column becomes extended. Rational in the circuit parameters.

    :param coords: Retained Coordinates, in any column order.
    :returns: S_1 in coords.types order; apply it with transform or Result.transform.
    """
    columns = sym.eye(len(coords.types))
    compact = [j for j, t in enumerate(coords.types) if t == "c"]
    other = [j for j, t in enumerate(coords.types) if t != "c"]
    S = sym.Matrix(columns)
    if compact and other:
        sheared = _decouple_block(columns[:, other], columns[:, compact], coords.C_tilde)
        for k, j in enumerate(other):
            S[:, j] = sheared[:, k]
    return _cancel(S)


def _gram_schmidt(vectors, form):
    """Make the columns form-orthogonal, keeping their order (no normalization)."""
    orthogonal = []
    for v in vectors:
        for u in orthogonal:
            v = v - ((u.T * form * v)[0] / (u.T * form * u)[0]) * u
        orthogonal.append(v.applyfunc(sym.cancel))
    return orthogonal


def normal_mode_decoupling(lag, coords, values=None):
    """S_1 that removes every quadratic coupling: compact charge shear, then normal modes.

    After compact_charge_decoupling, the noncompact block of C~ is
    C = ((C~^{-1})_nn)^{-1} and that of L~* is unchanged, so the noncompact columns
    become the solutions of L* x = lambda C x, i.e. eigenvectors of (C~^{-1})_nn L*.
    Distinct eigenvalues are already C-orthogonal; degenerate eigenspaces are
    C-orthogonalized by Gram-Schmidt. Extended and harmonic columns mix, so harmonic
    modes generally become extended.

    Symbolically (values=None) the eigenvalues are roots of a degree-N_n polynomial,
    so this is only practical for two or three noncompact modes. With values, the
    circuit parameters are substituted first and scipy.linalg.eigh(L*, C, type=1)
    gives C-orthonormal columns, so the noncompact block of S_1 is floating point.

    The new C~ and L~* are known by construction: the compact block of the shear and
    diag(x_i^T C x_i), diag(x_i^T L* x_i) on the noncompact block. They are returned so
    transform can skip recomputing them, which is slow when S_1 carries square roots.

    :param lag: Node Lagrangian for the same circuit as coords.
    :param coords: Retained Coordinates, in any column order.
    :param values: Optional {symbol: number} substituted before diagonalizing.
    :returns: (S_1, C_tilde, Lstar_tilde), S_1 in coords.types order; apply it with
        transform(lag, coords, S_1, C_tilde=C_tilde, Lstar_tilde=Lstar_tilde).
        C_tilde and Lstar_tilde are None when there is nothing to diagonalize.
    """
    # Count the noncompact modes, which are the last N_n columns of S_1.
    eh = [j for j, t in enumerate(coords.types) if t != "c"]
    n_non_compact = len(eh)
    if n_non_compact == 0:
        return sym.eye(len(coords.types)), None, None
    
    # Compact-charge decoupling
    S_a = compact_charge_decoupling(coords)
    if values is not None:
        S_a = S_a.subs(values)
    if n_non_compact == 1:
        return S_a, None, None
    coords_a = transform(lag, coords, S_a)
    C_tilde, Lstar_tilde = coords_a.C_tilde.as_mutable(), coords_a.Lstar_tilde.as_mutable()
    if values is not None:
        C_tilde, Lstar_tilde = C_tilde.subs(values), Lstar_tilde.subs(values)
    C_a = C_tilde.extract(eh, eh)
    L_a = Lstar_tilde.extract(eh, eh)

    # Normal-mode decoupling of the noncompact block
    S_b = _normal_mode_block(C_a, L_a, values)

    # Concatenate S_b with identity on the compact block
    S_n = _embed(S_b, eh, len(coords.types))

    # The noncompact blocks of C~ and L~* become diagonal
    for a, i in enumerate(eh):
        v = S_b[:, a]
        for j in eh:
            C_tilde[i, j] = Lstar_tilde[i, j] = 0
        C_tilde[i, i] = _radical_cancel((v.T * C_a * v)[0])
        Lstar_tilde[i, i] = _radical_cancel((v.T * L_a * v)[0])

    # Return the product of the two transformations
    return S_a * S_n, C_tilde, Lstar_tilde


def _normal_mode_block(C_a, L_a, values=None):
    """Columns x_i solving L_a x = lambda C_a x, C_a-orthogonal; C_a-orthonormal when numerical."""
    # Numerical path
    if values is not None:
        _, X = eigh(np.array(L_a, dtype=float), np.array(C_a, dtype=float), type=1)
        return sym.Matrix(X.tolist())
    # Symbolic path
    columns = []
    C_a_inv = _quick_inv(C_a)
    for eval, multiplicity, vectors in _cancel(C_a_inv*L_a).eigenvects(simplify=False):
        columns += _gram_schmidt(vectors, C_a)
    return sym.Matrix.hstack(*columns)


def _embed(B, indices, n):
    """n x n identity with B in the rows and columns indices."""
    M = sym.eye(n)
    for a, i in enumerate(indices):
        for b, j in enumerate(indices):
            M[i, j] = B[a, b]
    return M


def _radical_cancel(x):
    """cancel, with each square root sqrt(D) a symbol r reduced mod r^2 - D and rationalized away below.

    Normal modes of two noncompact modes carry one sqrt of the discriminant, which
    plain cancel expands as an independent generator; reducing mod r^2 - D lets
    the entries that vanish by construction cancel to 0.
    """
    x = sym.sympify(x)
    roots = {p.base: sym.Dummy("r") for p in x.atoms(sym.Pow) if p.exp.is_Rational and p.exp.q == 2}
    if not roots:
        return sym.cancel(x)
    x = x.replace(lambda e: e.is_Pow and e.base in roots and e.exp.is_Rational and e.exp.q == 2,
                  lambda e: roots[e.base] ** e.exp.p)
    num, den = sym.fraction(sym.cancel(x))
    for D, r in roots.items():
        num, den = sym.rem(num, r**2 - D, r), sym.rem(den, r**2 - D, r)
        if den.has(r):  # den = a + b r; multiply through by a - b r
            conjugate = den.subs(r, -r)
            num, den = sym.rem(sym.expand(num * conjugate), r**2 - D, r), sym.rem(sym.expand(den * conjugate), r**2 - D, r)
    return sym.cancel(num / den).xreplace({r: sym.sqrt(D) for D, r in roots.items()})


def transform(lag, coords, S1, *, C_tilde=None, Lstar_tilde=None):
    """Apply Z -> Z S_1, rebuild from lag, and classify the final columns.

    The allowed mixing follows eq:S1_structure and eq:Scc_unimodular;
    cancellations can restore harmonic axes inside the extended block.
    The new compact columns are Z_c S_cc, so their periods stay primitive.
    Z_full gains blockdiag(S_1, I) on the right, so the new R is S_1^{-1} R and the
    full node matrix is not inverted again.

    :param lag: Node Lagrangian for the same circuit as coords.
    :param coords: Current retained coordinates.
    :param S1: Invertible matrix satisfying residual mixing and compact-period constraints.
    :param C_tilde: Optional S1^T C~ S1, when already known; passed to coordinates.
    :param Lstar_tilde: Optional S1^T L~* S1, likewise.
    :returns: New Coordinates with node matrix coords.Z * S1.
    :raises ValueError: Invalid dimensions, disallowed mixing, or nonprimitive compact periods.
    """
    S = sym.ImmutableMatrix(S1)
    _check_residual(coords.types, S)
    retained = [i for i, t in enumerate(coords.types_full) if t in DYNAMICAL]
    Z_full = coords.Z_full.as_mutable()
    Z = coords.Z * S
    for j, index in enumerate(retained):
        Z_full[:, index] = Z[:, j]
    return coordinates(lag, Z_full, C_tilde=C_tilde, Lstar_tilde=Lstar_tilde, R=_quick_inv(S) * coords.R)
