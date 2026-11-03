"""Numerical search over the residual transformation ``S_1`` (Sec. III.B).

The parameterization follows ``eq:S_param`` in the paper: its real invertible
blocks are written in polar form as ``exp(A) exp(B)``, with ``A`` antisymmetric
and ``B`` symmetric.  The first implemented application is the normal-mode
cost of ``eq:opt_problem``.
"""

from dataclasses import dataclass, field

import numpy as np
import sympy as sym
from scipy.linalg import expm
from scipy.optimize import OptimizeResult, minimize

from .quantize import numerify


def _indices(types, kind):
    return tuple(i for i, value in enumerate(types) if value == kind)


def _skew_symmetric(values, n):
    matrix = np.zeros((n, n))
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = values[k]
            matrix[j, i] = -values[k]
            k += 1
    return matrix


def _symmetric(values, n):
    matrix = np.zeros((n, n))
    k = 0
    for i in range(n):
        for j in range(i, n):
            matrix[i, j] = matrix[j, i] = values[k]
            k += 1
    return matrix


def _polar_block(values, n):
    n_skew = n * (n - 1) // 2
    A = _skew_symmetric(values[:n_skew], n)
    B = _symmetric(values[n_skew:], n)
    return expm(A) @ expm(B)


@dataclass(frozen=True)
class Parameterization:
    """``S(x)`` of ``eq:S_param`` in ``coords.types`` order.

    ``S_cc`` is fixed.  With ``relaxed=False`` the extended and harmonic
    diagonal blocks have independent polar parameterizations and ``S_ce`` and
    ``S_he`` are free.  With ``relaxed=True`` all noncompact columns form one
    polar block and their compact-row block is free.
    """

    types: tuple
    S_cc: np.ndarray
    relaxed: bool
    c: tuple = field(init=False, repr=False)
    e: tuple = field(init=False, repr=False)
    h: tuple = field(init=False, repr=False)
    n: tuple = field(init=False, repr=False)

    def __post_init__(self):
        types = tuple(self.types)
        if set(types) - {"c", "e", "h"}:
            raise ValueError("types must contain only retained modes 'c', 'e', and 'h'")
        c, e, h = (_indices(types, kind) for kind in ("c", "e", "h"))
        S_cc = np.asarray(self.S_cc)
        if S_cc.shape != (len(c), len(c)):
            raise ValueError(f"S_cc must have shape {(len(c), len(c))}")
        if not np.all(np.equal(S_cc, np.rint(S_cc))):
            raise ValueError("S_cc must have integer entries")
        if c and round(abs(np.linalg.det(np.asarray(S_cc, dtype=float)))) != 1:
            raise ValueError("S_cc must be unimodular")
        object.__setattr__(self, "types", types)
        object.__setattr__(self, "S_cc", np.asarray(np.rint(S_cc), dtype=int))
        object.__setattr__(self, "c", c)
        object.__setattr__(self, "e", e)
        object.__setattr__(self, "h", h)
        object.__setattr__(self, "n", e + h)

    @property
    def size(self):
        nc, ne, nh, nn = len(self.c), len(self.e), len(self.h), len(self.n)
        if self.relaxed:
            return nc * nn + nn * nn
        return nc * ne + nh * ne + ne * ne + nh * nh

    def x0(self):
        """The origin, which gives ``blockdiag(S_cc, I)``."""
        return np.zeros(self.size)

    def S(self, x):
        """Assemble the residual transformation represented by ``x``."""
        x = np.asarray(x, dtype=float)
        if x.shape != (self.size,):
            raise ValueError(f"x must have shape {(self.size,)}")
        S = np.eye(len(self.types))
        if self.c:
            S[np.ix_(self.c, self.c)] = self.S_cc
        k = 0
        if self.relaxed:
            free = len(self.c) * len(self.n)
            if free:
                S[np.ix_(self.c, self.n)] = x[k:k + free].reshape(len(self.c), len(self.n))
            k += free
            if self.n:
                S[np.ix_(self.n, self.n)] = _polar_block(x[k:], len(self.n))
            return S

        free = len(self.c) * len(self.e)
        if free:
            S[np.ix_(self.c, self.e)] = x[k:k + free].reshape(len(self.c), len(self.e))
        k += free
        free = len(self.h) * len(self.e)
        if free:
            S[np.ix_(self.h, self.e)] = x[k:k + free].reshape(len(self.h), len(self.e))
        k += free
        if self.e:
            count = len(self.e) ** 2
            S[np.ix_(self.e, self.e)] = _polar_block(x[k:k + count], len(self.e))
            k += count
        if self.h:
            S[np.ix_(self.h, self.h)] = _polar_block(x[k:], len(self.h))
        return S


def parameterization(types, *, S_cc=None, relaxed=False):
    """Construct the ``eq:S_param`` parameterization for retained mode types."""
    types = tuple(types)
    nc = types.count("c")
    if S_cc is None:
        S_cc = np.eye(nc, dtype=int)
    return Parameterization(types, S_cc, relaxed)


def _kappa(M, indices):
    """Normalized sub-block, with a finite-value check for optimizer safety."""
    indices = tuple(indices)
    if not indices:
        return np.zeros((0, 0))
    block = np.asarray(M)[np.ix_(indices, indices)]
    diagonal = np.diag(block)
    if np.any(diagonal <= 0) or not np.all(np.isfinite(diagonal)):
        return np.full(block.shape, np.nan)
    return block / np.sqrt(np.outer(diagonal, diagonal))


def _quadratic_forms(S, C, L):
    Cp = S.T @ C @ S
    Lp = S.T @ L @ S
    try:
        Cp_inv = np.linalg.solve(Cp, np.eye(Cp.shape[0]))
    except np.linalg.LinAlgError:
        return Cp, None, Lp
    return Cp, Cp_inv, Lp


def _masked_upper_square(K, mask=None):
    if not np.all(np.isfinite(K)):
        return np.inf
    upper = np.triu(np.ones(K.shape, dtype=bool), 1)
    if mask is not None:
        upper &= np.asarray(mask, dtype=bool)
    return float(np.sum(K[upper] ** 2))


def cost_normal_modes(C, L, types):
    """Return the scale-free normal-mode objective ``f(S)`` of Sec. III.B."""
    C, L = np.asarray(C, dtype=float), np.asarray(L, dtype=float)
    all_indices = tuple(range(len(types)))
    noncompact = tuple(i for i, kind in enumerate(types) if kind != "c")

    def objective(S):
        _, Cp_inv, Lp = _quadratic_forms(np.asarray(S), C, L)
        if Cp_inv is None:
            return np.inf
        return (_masked_upper_square(_kappa(Cp_inv, all_indices))
                + _masked_upper_square(_kappa(Lp, noncompact)))

    return objective


def cost_targeted(C, L, types, mask):
    """Return a normal-mode objective restricted by coupling masks.

    ``mask`` may be one square array, applied to both forms, or a pair
    ``(mask_C_inv, mask_L)``.  The latter may use either full-size masks or a
    mask already restricted to the noncompact block.
    """
    C, L = np.asarray(C, dtype=float), np.asarray(L, dtype=float)
    all_indices = tuple(range(len(types)))
    noncompact = tuple(i for i, kind in enumerate(types) if kind != "c")
    if isinstance(mask, tuple) and len(mask) == 2:
        mask_C, mask_L = map(np.asarray, mask)
    else:
        mask_C = mask_L = np.asarray(mask)
    if mask_C.shape != C.shape:
        raise ValueError("the C^-1 mask must have the same shape as C")
    if mask_L.shape == L.shape:
        mask_L = mask_L[np.ix_(noncompact, noncompact)]
    if mask_L.shape != (len(noncompact), len(noncompact)):
        raise ValueError("the L mask must be full-size or match the noncompact block")

    def objective(S):
        _, Cp_inv, Lp = _quadratic_forms(np.asarray(S), C, L)
        if Cp_inv is None:
            return np.inf
        return (_masked_upper_square(_kappa(Cp_inv, all_indices), mask_C)
                + _masked_upper_square(_kappa(Lp, noncompact), mask_L))

    return objective


def cost_localization(C, L, wJ, types):
    """Return the scale-fixed junction-localization stage-one objective.

    This is the continuous l1 alternative in Sec. III.B.  Discrete junction
    selections and the closed-form shear of ``eq:shears`` remain separate
    staging operations; this constructor intentionally represents only the
    objective accepted by :func:`numerical_S1`.
    """
    C, wJ = np.asarray(C, dtype=float), np.asarray(wJ, dtype=float)
    del L  # The lexicographic quadratic stage follows after localization.
    active = tuple(i for i, kind in enumerate(types) if kind != "h")

    def objective(S):
        Cp = S.T @ C @ S
        diagonal = np.diag(Cp)
        if np.any(diagonal <= 0) or not np.all(np.isfinite(diagonal)):
            return np.inf
        # Gauge-fix to (S.T C S)_jj = C_jj without actually changing S.
        normalized = (wJ @ S) * np.sqrt(np.diag(C) / diagonal)[None, :]
        return float(np.sum(np.abs(normalized[:, active])))

    return objective


@dataclass(frozen=True)
class NumericalInfo:
    """Diagnostics returned by :func:`numerical_S1`."""

    result: OptimizeResult
    cost: float
    C_tilde: np.ndarray
    C_tilde_inv: np.ndarray
    Lstar_tilde: np.ndarray
    kappa_C_inv: np.ndarray
    kappa_L: np.ndarray


def _normalized(S, C, noncompact, mode):
    if mode is None:
        return S
    if mode not in {"C", "keep"}:
        raise ValueError("normalize must be 'C', 'keep', or None")
    S = S.copy()
    Cp = S.T @ C @ S
    for j in noncompact:
        target = 1.0 if mode == "C" else C[j, j]
        S[:, j] *= np.sqrt(target / Cp[j, j])
    return S


def _sympy_matrix(S, parameterization):
    result = sym.zeros(*S.shape)
    compact_positions = {(i, j) for i in parameterization.c for j in parameterization.c}
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if (i, j) in compact_positions:
                result[i, j] = sym.Integer(int(parameterization.S_cc[
                    parameterization.c.index(i), parameterization.c.index(j)]))
            elif S[i, j] != 0:
                result[i, j] = sym.Float(float(S[i, j]), 17)
    return sym.ImmutableMatrix(result)


def numerical_S1(lag, coords, values, cost, *, S_cc=None, relaxed=False,
                 x0=None, restarts=3, seed=0, tol=1e-12, normalize="C"):
    """Minimize a Sec. III.B cost over ``S(x)``.

    ``cost`` is a constructor with signature ``cost(C, L, types) -> f(S)``;
    pass :func:`cost_normal_modes` directly.  The return value is ``(S1,
    info)``, with exact compact integers in ``S1`` and numerical diagnostics in
    ``info``.
    """
    del lag  # Kept in the public signature alongside normal_mode_decoupling.
    numeric = numerify(coords, values or {})
    C = np.asarray(numeric.C_tilde, dtype=float)
    L = np.asarray(numeric.Lstar_tilde, dtype=float)
    if C.shape != L.shape or C.shape != (len(coords.types), len(coords.types)):
        raise ValueError("coordinate matrices and types have incompatible shapes")
    if not np.all(np.isfinite(C)) or not np.all(np.isfinite(L)):
        raise ValueError("values must numerically specify every entry of C and L")
    p = parameterization(coords.types, S_cc=S_cc, relaxed=relaxed)
    objective_S = cost(C, L, coords.types)
    objective_x = lambda x: objective_S(p.S(x))

    if restarts < 0:
        raise ValueError("restarts must be nonnegative")

    starts = []
    if x0 is not None:
        x0 = np.asarray(x0, dtype=float)
        if x0.shape != (p.size,):
            raise ValueError(f"x0 must have shape {(p.size,)}")
        starts.append(x0)
    else:
        starts.append(p.x0())
    rng = np.random.default_rng(seed)
    starts.extend(0.3 * rng.normal(size=p.size) for _ in range(restarts))
    if p.size:
        results = [minimize(objective_x, start, method="BFGS",
                            options={"gtol": tol, "maxiter": 5000}) for start in starts]
    else:
        value = objective_x(p.x0())
        results = [OptimizeResult(x=p.x0(), fun=value, success=True,
                                  message="the parameterization has no free variables")]
    finite = [result for result in results if np.isfinite(result.fun)]
    if not finite:
        raise RuntimeError("every numerical-search restart returned a non-finite cost")
    best = min(finite, key=lambda result: result.fun)

    noncompact = tuple(i for i, kind in enumerate(coords.types) if kind != "c")
    S = _normalized(p.S(best.x), C, noncompact, normalize)
    Cp, Cp_inv, Lp = _quadratic_forms(S, C, L)
    info = NumericalInfo(
        result=best,
        cost=float(objective_S(S)),
        C_tilde=Cp,
        C_tilde_inv=Cp_inv,
        Lstar_tilde=Lp,
        kappa_C_inv=_kappa(Cp_inv, range(len(coords.types))),
        kappa_L=_kappa(Lp, noncompact),
    )
    return _sympy_matrix(S, p), info
