"""Mode classification, decoupling, and residual freedom (Sec III).

Mode types are the superscripts of eq:Z_mode_blocks:
"c" compact, "e" extended, "h" harmonic (dynamical) and
"fr" free, "fz" frozen, "cy" cyclic (non-dynamical).
"""

from dataclasses import dataclass
from functools import reduce
from itertools import combinations
from math import gcd

import sympy as sym

DYNAMICAL = ("c", "e", "h")
NONDYNAMICAL = ("fr", "fz", "cy")
TYPES = DYNAMICAL + NONDYNAMICAL


@dataclass(frozen=True)
class Coordinates:
    Z: sym.ImmutableMatrix  # retained block, Phi_int = Z Phi_tilde
    types: tuple
    C_tilde: sym.ImmutableMatrix  # eq:Zd_schur
    C_tilde_inv: sym.ImmutableMatrix
    Lstar_tilde: sym.ImmutableMatrix
    omega_J_tilde: sym.ImmutableMatrix  # Z^T omega_J_int
    omega_L_tilde: sym.ImmutableMatrix  # Z^T omega_L_int


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
    return sym.nsimplify(sym.simplify(x)).is_integer


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


def check_basis(lag, Z_full, types):
    """Check a square basis before decoupling (tab:var_basis).

    Z_full must be invertible and its non-dynamical columns must lie in their
    subspaces.  The dynamical columns are checked by check_coordinates after
    decouple, so Z_full need not be decoupled yet.
    """
    Z = sym.Matrix(Z_full)
    n = lag.Omega_int.rows
    types = tuple(types)
    if Z.shape != (n, n) or len(types) != n:
        raise ValueError(f"Z_full must be {n}x{n} with {n} types")
    if set(types) - set(TYPES):
        raise ValueError(f"Unknown mode types: {sorted(set(types) - set(TYPES))}")
    if sym.simplify(Z.det()) == 0:
        raise ValueError("Z_full is singular")
    sub = subspaces(lag)
    for j, t in enumerate(types):
        if t in NONDYNAMICAL and not _contains(sub[{"fr": "JL", "fz": "C", "cy": "cy"}[t]], Z[:, j]):
            raise ValueError(f"Column {j} is not a valid {t!r} direction (tab:var_basis)")
    for t in NONDYNAMICAL:
        if types.count(t) != sub[t].cols:
            raise ValueError(f"Circuit has {sub[t].cols} {t!r} modes, Z_full declares {types.count(t)}")


def check_coordinates(lag, coords):
    """Check the decoupled dynamical columns against tab:var_basis and eq:compact_period_lattice."""
    sub = subspaces(lag)
    actual = _mode_types(coords.omega_J_tilde, coords.omega_L_tilde)
    for j, (declared, inferred) in enumerate(zip(coords.types, actual)):
        if declared != inferred:
            raise ValueError(f"Retained column {j} is not a valid {declared!r} direction (tab:var_basis)")
    types = coords.types
    if types.count("c") != sub["c"].cols:
        raise ValueError(f"Circuit has {sub['c'].cols} 'c' modes, coordinates declare {types.count('c')}")
    if types.count("h") > sub["h"].cols:
        raise ValueError(f"Circuit has at most {sub['h'].cols} 'h' modes, coordinates declare {types.count('h')}")
    compact = [j for j, t in enumerate(types) if t == "c"]
    if compact:
        W = (lag.omega_J_int.T * coords.Z[:, compact]).applyfunc(sym.simplify)  # omega_{J,c}^T
        if not all(_is_integer(x) for x in W):
            raise ValueError("omega_J,c^T must be integer (Sec IV.B.1)")
        minors = [sym.simplify(W.extract(list(rows), list(range(W.cols))).det()) for rows in combinations(range(W.rows), W.cols)]
        if reduce(gcd, (int(m) for m in minors), 0) != 1:
            raise ValueError("Compact periods are not primitive (eq:compact_period_lattice)")


# --- decoupling and truncation ----------------------------------------------


def _mode_types(omega_J, omega_L):
    """Classify retained columns by junction and inductor participation (tab:var_basis)."""
    return tuple(
        "e" if any(j) and any(l) else "c" if any(j) else "h" if any(l) else "fr"
        for j, l in zip(omega_J.applyfunc(sym.simplify).tolist(),
                        omega_L.applyfunc(sym.simplify).tolist())
    )


def coordinates(lag, Z, types=None):
    """Transformed objects of the retained block Z (eq:Zd_schur)."""
    Z = sym.Matrix(Z)
    C_tilde = (Z.T * lag.C_int * Z).applyfunc(sym.cancel)
    if sym.cancel(C_tilde.det()) == 0:
        raise ValueError("Retained capacitance matrix is singular")
    omega_J, omega_L = Z.T * lag.omega_J_int, Z.T * lag.omega_L_int
    freeze = sym.ImmutableMatrix
    return Coordinates(
        Z=freeze(Z), types=_mode_types(omega_J, omega_L) if types is None else tuple(types),
        C_tilde=freeze(C_tilde),
        C_tilde_inv=freeze(C_tilde.inv().applyfunc(sym.cancel)),
        Lstar_tilde=freeze((Z.T * lag.Lstar_int * Z).applyfunc(sym.cancel)),
        omega_J_tilde=freeze(omega_J), omega_L_tilde=freeze(omega_L),
    )


def _decouple_column(v, nondynamical, form):
    """Eliminate discarded columns in reverse order, updating all earlier columns."""
    Z = sym.Matrix.hstack(v, nondynamical)
    for i in range(Z.cols - 1, 0, -1):
        w = Z[:, i]
        pivot = sym.cancel((w.T * form * w)[0])
        if pivot != 0:  # A null direction of this positive-semidefinite form needs no correction.
            Z[:, :i] = (Z[:, :i] - w * (w.T * form * Z[:, :i]) / pivot).applyfunc(sym.cancel)
    return Z[:, 0]


def decouple(lag, Z_full, types):
    """Decouple one column at a time, then truncate (eq:ZY_blocks, Sec III.B).

    Eliminate free directions in C and frozen directions in L* serially
    (eq:decouple_final, eq:decouple_final_L), updating the remaining columns.
    Cyclic directions are discarded; no discarded-sector matrix is inverted.

    :param lag: Node Lagrangian matrices.
    :param Z_full: Full square node-to-coordinate basis.
    :param types: Mode label for each basis column.
    :returns: Coordinates containing the retained dynamical columns and transformed matrices.
    """
    Z = sym.Matrix(Z_full)
    sectors = [
        (form, Z[:, [j for j, t in enumerate(types) if t == kind]])
        for kind, form in (("fr", lag.C_int), ("fz", lag.Lstar_int))
    ]
    retained, retained_types = [], []
    for j, t in enumerate(types):
        if t not in DYNAMICAL:
            continue
        z = Z[:, j]
        for form, nondynamical in sectors:
            z = _decouple_column(z, nondynamical, form)
        retained.append(z.applyfunc(sym.cancel))
        retained_types.append(t)
    if not retained:
        raise ValueError("Circuit has no dynamical modes")
    return coordinates(lag, sym.Matrix.hstack(*retained), retained_types)


# --- residual freedom -------------------------------------------------------

# (old type, new type) pairs a column may be mixed along, eq:S1_structure.
_S1a = {("c", "c"), ("c", "e"), ("e", "e"), ("h", "e"), ("h", "h")}
_S1b = _S1a | {("e", "h")}


def residual_freedom(coords, keep_harmonic=True):
    """Symbolic S_1 with free entries where eq:S1_structure allows them.

    keep_harmonic=True gives S_{1,a}; False gives S_{1,b}, which lets harmonic
    columns acquire extended components. S_cc must also lie in GL(N_c, Z)
    (eq:Scc_unimodular); transform() checks that once values are chosen.

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


def transform(lag, coords, S1):
    """Apply Z -> Z S_1, rebuild from lag, and classify the final columns.

    The allowed mixing follows eq:S1_structure and eq:Scc_unimodular;
    cancellations can restore harmonic axes inside the extended block.

    :param lag: Node Lagrangian for the same circuit as coords.
    :param coords: Current retained coordinates.
    :param S1: Invertible matrix satisfying residual mixing and compact-period constraints.
    :returns: New Coordinates with node matrix coords.Z * S1.
    :raises ValueError: Invalid dimensions, disallowed mixing, or nonprimitive compact periods.
    """
    S = sym.Matrix(S1)
    types = coords.types
    if S.shape != (len(types), len(types)) or sym.simplify(S.det()) == 0:
        raise ValueError("S1 must be square, invertible, and match the retained modes")
    for i in range(S.rows):
        for j in range(S.cols):
            if S[i, j] != 0 and (types[i], types[j]) not in _S1b:
                raise ValueError(f"S1[{i}, {j}] mixes {types[i]!r} into {types[j]!r} (eq:S1_structure)")
    compact = [j for j, t in enumerate(types) if t == "c"]
    S_cc = S.extract(compact, compact).applyfunc(sym.simplify)
    if compact and (not all(_is_integer(x) for x in S_cc) or abs(S_cc.det()) != 1):
        raise ValueError("S_cc must lie in GL(N_c, Z) (eq:Scc_unimodular)")
    new = coordinates(lag, coords.Z * S)
    check_coordinates(lag, new)
    return new
