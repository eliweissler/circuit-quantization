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
    for j, t in enumerate(types):
        if t in required and not _contains(sub[required[t]], Z[:, j]):
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
