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


def _primitive(v):
    """Scale a rational vector to coprime integers, for readability."""
    denominators = [sym.fraction(sym.nsimplify(x))[1] for x in v]
    v = v * reduce(sym.ilcm, denominators, 1)
    divisor = reduce(gcd, (int(x) for x in v), 0)
    return v / divisor if divisor else v


def _columns(vectors, n):
    return sym.Matrix.hstack(*vectors) if vectors else sym.zeros(n, 0)


def _null(A, n):
    return _columns([_primitive(v) for v in A.nullspace()], n)


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
    """Square Z = (Z^c Z^e Z^h Z^fr Z^fz Z^cy) of eq:Z_mode_blocks and its column types."""
    sub = subspaces(lag)
    sub["c"] = _normalize_compact(sub["c"], lag.omega_J_int)
    Z = sym.Matrix.hstack(*(sub[t] for t in TYPES))
    types = tuple(t for t in TYPES for _ in range(sub[t].cols))
    return sym.ImmutableMatrix(Z), types


# --- validation ------------------------------------------------------------


def _is_integer(x):
    return sym.nsimplify(sym.simplify(x)).is_integer


def check_basis(lag, Z_full, types):
    """Check a square basis against tab:var_basis and eq:compact_period_lattice."""
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
    required = {"c": "L", "h": "J", "fr": "JL", "fz": "C", "cy": "cy"}
    excluded = {"e": ("J", "L"), "h": ("L",)}  # extended sees both terms, harmonic the quadratic one
    for j, t in enumerate(types):
        inside = t in required and not _contains(sub[required[t]], Z[:, j])
        outside = any(_contains(sub[s], Z[:, j]) for s in excluded.get(t, ()))
        if inside or outside:
            raise ValueError(f"Column {j} is not a valid {t!r} direction (tab:var_basis)")
    for t in ("c",) + NONDYNAMICAL:
        if types.count(t) != sub[t].cols:
            raise ValueError(f"Circuit has {sub[t].cols} {t!r} modes, Z_full declares {types.count(t)}")
    compact = [j for j, t in enumerate(types) if t == "c"]
    if compact:
        W = (lag.omega_J_int.T * Z[:, compact]).applyfunc(sym.simplify)  # omega_{J,c}^T
        if not all(_is_integer(x) for x in W):
            raise ValueError("omega_J,c^T must be integer (Sec IV.B.1)")
        minors = [sym.simplify(W.extract(list(rows), list(range(W.cols))).det()) for rows in combinations(range(W.rows), W.cols)]
        if reduce(gcd, (int(m) for m in minors), 0) != 1:
            raise ValueError("Compact periods are not primitive (eq:compact_period_lattice)")


# --- decoupling and truncation ----------------------------------------------


def _orthogonalize(vectors, form):
    """Gram-Schmidt in the bilinear form, so each step divides by a positive scalar."""
    result = []
    for v in vectors:
        for u in result:
            v = v - (u.T * form * v)[0] / (u.T * form * u)[0] * u
        result.append(v.applyfunc(sym.cancel))
    return result


def coordinates(lag, Z, types):
    """Transformed objects of the retained block Z (eq:Zd_schur)."""
    Z = sym.Matrix(Z)
    C_tilde = (Z.T * lag.C_int * Z).applyfunc(sym.cancel)
    if sym.cancel(C_tilde.det()) == 0:
        raise ValueError("Retained capacitance matrix is singular")
    freeze = sym.ImmutableMatrix
    return Coordinates(
        Z=freeze(Z), types=tuple(types), C_tilde=freeze(C_tilde),
        C_tilde_inv=freeze(C_tilde.inv().applyfunc(sym.cancel)),
        Lstar_tilde=freeze((Z.T * lag.Lstar_int * Z).applyfunc(sym.cancel)),
        omega_J_tilde=freeze(Z.T * lag.omega_J_int),
        omega_L_tilde=freeze(Z.T * lag.omega_L_int),
    )


def decouple(lag, Z_full, types):
    """Retained block Z_1 + Z_2 M of eq:ZY_blocks, then truncation (Sec III.B).

    M_C = -C~_22^{-1} C~_21 (eq:decouple_final) acts on the free block and
    M_L = -L~*_22^{-1} L~*_21 (eq:decouple_final_L) on the frozen block.
    Each block is first orthogonalized in its own form, which makes C~_22 and
    L~*_22 diagonal, so M is applied one column and one mode at a time.
    """
    Z = sym.Matrix(Z_full)
    columns = [Z[:, j] for j in range(Z.cols)]
    free = _orthogonalize([z for z, t in zip(columns, types) if t == "fr"], lag.C_int)
    frozen = _orthogonalize([z for z, t in zip(columns, types) if t == "fz"], lag.Lstar_int)
    retained, retained_types = [], []
    for z, t in zip(columns, types):
        if t not in DYNAMICAL:
            continue
        for v in free:
            z = z - (v.T * lag.C_int * z)[0] / (v.T * lag.C_int * v)[0] * v
        for w in frozen:
            z = z - (w.T * lag.Lstar_int * z)[0] / (w.T * lag.Lstar_int * w)[0] * w
        retained.append(z.applyfunc(sym.cancel))
        retained_types.append(t)
    if not retained:
        raise ValueError("Circuit has no dynamical modes")
    Z_1 = sym.Matrix.hstack(*retained)
    Z_2 = _columns([z for z, t in zip(columns, types) if t in NONDYNAMICAL], Z.rows)
    for name, form in (("capacitance", lag.C_int), ("susceptance", lag.Lstar_int)):
        if (Z_1.T * form * Z_2).applyfunc(sym.cancel) != sym.zeros(Z_1.cols, Z_2.cols):
            raise AssertionError(f"Nonzero {name} cross block after decoupling (eq:ZY_blocks)")
    return coordinates(lag, Z_1, retained_types)


# --- residual freedom -------------------------------------------------------

# (old type, new type) pairs a column may be mixed along, eq:S1_structure.
_S1a = {("c", "c"), ("c", "e"), ("e", "e"), ("h", "e"), ("h", "h")}
_S1b = _S1a | {("e", "h")}


def residual_freedom(coords, keep_harmonic=True):
    """Symbolic S_1 with free entries where eq:S1_structure allows them.

    keep_harmonic=True gives S_{1,a}; False gives S_{1,b}, which lets harmonic
    columns acquire extended components. S_cc must also lie in GL(N_c, Z)
    (eq:Scc_unimodular); transform() checks that once values are chosen.
    """
    allowed = _S1a if keep_harmonic else _S1b
    types = coords.types
    return sym.Matrix(
        len(types), len(types),
        lambda i, j: sym.Symbol(f"s_{i + 1}_{j + 1}") if (types[i], types[j]) in allowed else 0,
    )


def transform(coords, S1):
    """Apply Z -> Z S_1 to the retained coordinates (eq:S1_structure, eq:Scc_unimodular)."""
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
    new_types = tuple(
        "e" if t == "h" and any(S[i, j] != 0 for i, u in enumerate(types) if u == "e") else t
        for j, t in enumerate(types)
    )
    freeze = sym.ImmutableMatrix
    C_tilde = (S.T * coords.C_tilde * S).applyfunc(sym.cancel)
    return Coordinates(
        Z=freeze(coords.Z * S), types=new_types, C_tilde=freeze(C_tilde),
        C_tilde_inv=freeze(C_tilde.inv().applyfunc(sym.cancel)),
        Lstar_tilde=freeze((S.T * coords.Lstar_tilde * S).applyfunc(sym.cancel)),
        omega_J_tilde=freeze(S.T * coords.omega_J_tilde),
        omega_L_tilde=freeze(S.T * coords.omega_L_tilde),
    )
