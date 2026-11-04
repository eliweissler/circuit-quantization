"""Product-state coordinates: the mean-field cost of Sec. III.B.

The product ansatz ``|Psi> = (x)_i |psi_i>`` is optimized self-consistently
for the Hamiltonian written in the transformed coordinates ``phi = S phi'``,
and its energy is the cost minimized over ``S(x)`` of ``eq:S_numerical``.
This is the optimized-coordinate vibrational self-consistent field of Yagi,
Keceli, and Hirata, J. Chem. Phys. 137, 204118 (2012), with compact modes
solved in the charge basis.  In the harmonic limit the optimum is the normal
modes, so the cost contains ``eq:numerical_normal_cost`` as a special case.

Hamiltonian convention (docs/spec-format.md)::

    H/h [GHz] = 4 (n - n_g)^T E_C (n - n_g) + 1/2 phi^T E_L phi + phi^T b
                - sum_k E_k cos(omega_k^T phi + delta_k) + c

Under ``phi = S phi'`` and ``n' = S^T n`` the coefficients transform as::

    E_C' = S^-1 E_C S^-T,  n_g' = S^T n_g,  E_L' = S^T E_L S,
    b' = S^T b,            omega_k' = S^T omega_k.

The mean-field Hamiltonian of mode ``i`` keeps every term of ``H`` with the
operators of the other modes replaced by their expectation values, so every
junction cosine survives as a cosine of ``phi_i`` with a complex weight.
"""

import itertools
from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sympy as sym
from scipy.linalg import eigh
from scipy.optimize import OptimizeResult, minimize

from .numerical import cost_normal_modes, parameterization

# Eighth-order central stencils for d/dphi and d^2/dphi^2 (interior points).
_D1 = np.array([1 / 280, -4 / 105, 1 / 5, -4 / 5, 0.0, 4 / 5, -1 / 5, 4 / 105, -1 / 280])
_D2 = np.array([-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560])


@dataclass(frozen=True)
class NumericHamiltonian:
    """Floating-point Hamiltonian coefficients in the convention above."""

    types: tuple
    E_C: np.ndarray
    E_L: np.ndarray
    flux_linear: np.ndarray
    n_g: np.ndarray
    cosines: tuple  # of (energy, omega, offset)
    constant: float = 0.0

    def __post_init__(self):
        N = len(self.types)
        E_C, E_L = (np.asarray(M, dtype=float) for M in (self.E_C, self.E_L))
        b, n_g = (np.asarray(v, dtype=float).reshape(-1) for v in (self.flux_linear, self.n_g))
        if E_C.shape != (N, N) or E_L.shape != (N, N) or b.shape != (N,) or n_g.shape != (N,):
            raise ValueError("coefficient shapes do not match the number of modes")
        cosines = tuple((float(E), np.asarray(w, dtype=float).reshape(N), float(d))
                        for E, w, d in self.cosines)
        for name, value in (("E_C", E_C), ("E_L", E_L), ("flux_linear", b), ("n_g", n_g)):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "types", tuple(self.types))
        object.__setattr__(self, "cosines", cosines)
        object.__setattr__(self, "constant", float(self.constant))

    @property
    def size(self):
        return len(self.types)

    def transformed(self, S):
        """The coefficients in the coordinates ``phi'`` with ``phi = S phi'``."""
        S = np.asarray(S, dtype=float)
        S_inv = np.linalg.inv(S)
        return replace(
            self,
            E_C=S_inv @ self.E_C @ S_inv.T,
            E_L=S.T @ self.E_L @ S,
            flux_linear=S.T @ self.flux_linear,
            n_g=S.T @ self.n_g,
            cosines=tuple((E, S.T @ w, d) for E, w, d in self.cosines),
        )


def numeric_hamiltonian(spec, values):
    """Evaluate a ``HamiltonianSpec`` at parameter ``values`` (symbols or names)."""
    by_name = {s.name: s for s in spec.parameters}
    subs = {}
    for key, value in values.items():
        symbol = by_name[key] if isinstance(key, str) else by_name[key.name]
        subs[symbol] = value

    def scalar(expr):
        value = sym.sympify(expr).subs(subs)
        if value.free_symbols:
            raise ValueError(f"unassigned parameters {value.free_symbols} in {expr}")
        return float(value)

    def array(M):
        return np.array([[scalar(x) for x in row] for row in sym.Matrix(M).tolist()])

    return NumericHamiltonian(
        types=tuple(kind for _, kind in spec.modes),
        E_C=array(spec.E_C),
        E_L=array(spec.E_L),
        flux_linear=array(spec.flux_linear).reshape(-1),
        n_g=array(spec.n_g).reshape(-1),
        cosines=tuple((scalar(E), [float(w) for w in omega], scalar(d))
                      for E, omega, d in spec.cosines),
        constant=scalar(spec.constant),
    )


# ---------------------------------------------------------------- bases


class ChargeBasis:
    """Charge states ``|m>``, ``m = -cutoff..cutoff``, for a compact mode."""

    compact = True

    def __init__(self, cutoff):
        self.cutoff = int(cutoff)
        self.m = np.arange(-self.cutoff, self.cutoff + 1, dtype=float)

    @property
    def size(self):
        return self.m.size

    def n(self):
        return np.diag(self.m)

    def n2(self):
        return np.diag(self.m ** 2)

    def expiphi(self, k):
        """``exp(i k phi) |m> = |m + k>`` for integer ``k``."""
        if abs(k - round(k)) > 1e-8:
            raise ValueError(f"a compact mode needs an integer cosine multiplier, got {k}")
        k = int(round(k))
        return np.eye(self.size, k=-k)

    def phi(self):
        raise ValueError("a compact mode has no phase operator")

    def edge_weight(self, psi):
        return float(abs(psi[0]) ** 2 + abs(psi[-1]) ** 2)


class PhaseGrid:
    """Uniform grid on ``[-extent, extent]`` with eighth-order stencils."""

    compact = False

    def __init__(self, extent, points):
        self.extent = float(extent)
        self.points = int(points)
        self.grid = np.linspace(-self.extent, self.extent, self.points)
        self.step = self.grid[1] - self.grid[0]

    @property
    def size(self):
        return self.points

    def _stencil(self, coefficients, power):
        return sum(np.eye(self.size, k=offset) * c
                   for offset, c in zip(range(-4, 5), coefficients)) / self.step ** power

    def n(self):
        return -1j * self._stencil(_D1, 1)

    def n2(self):
        return -self._stencil(_D2, 2)

    def expiphi(self, k):
        return np.diag(np.exp(1j * k * self.grid))

    def phi(self):
        return np.diag(self.grid)

    def edge_weight(self, psi):
        return float(abs(psi[0]) ** 2 + abs(psi[-1]) ** 2)


def _touching(ham, i):
    """Junction energy and largest multiplier seen by mode ``i``."""
    energy = sum(abs(E) for E, w, _ in ham.cosines if abs(w[i]) > 1e-12)
    k_max = max((abs(w[i]) for E, w, _ in ham.cosines if abs(w[i]) > 1e-12), default=0.0)
    return energy, k_max


def default_bases(ham, *, charge_cutoff=20, max_points=400, extent_scale=1.0):
    """Bases sized from the diagonal quadratic terms and junction energies."""
    bases = []
    for i, kind in enumerate(ham.types):
        if kind == "c":
            bases.append(ChargeBasis(charge_cutoff))
            continue
        E_C, E_L = ham.E_C[i, i], ham.E_L[i, i]
        if E_C <= 0 or E_L <= 0:
            raise ValueError(f"mode {i} needs positive diagonal E_C and E_L, got {E_C}, {E_L}")
        zpf = (2 * E_C / E_L) ** 0.25
        omega = np.sqrt(8 * E_C * E_L)
        energy, k_max = _touching(ham, i)
        extent = max(6 * zpf, np.sqrt(2 * (2 * energy + 12 * omega + 4 * E_C) / E_L))
        extent *= extent_scale
        step = zpf / 6
        if k_max > 0:
            step = min(step, np.pi / (6 * k_max))
        points = int(np.ceil(2 * extent / step)) + 1
        bases.append(PhaseGrid(extent, min(points, max_points)))
    return bases


# ---------------------------------------------------------------- mean field


@dataclass(frozen=True)
class ProductState:
    """A converged product ansatz and its diagnostics."""

    energy: float
    states: tuple
    bases: tuple
    iterations: int
    converged: bool
    expectations: dict

    def vector(self):
        """The product state as one vector in the kron (mode-order) basis."""
        out = np.ones(1, dtype=complex)
        for psi in self.states:
            out = np.kron(out, psi)
        return out


def _mode_operators(ham, bases):
    ops = []
    for i, basis in enumerate(bases):
        n = basis.n() - ham.n_g[i] * np.eye(basis.size)
        n2 = basis.n2() - 2 * ham.n_g[i] * basis.n() + ham.n_g[i] ** 2 * np.eye(basis.size)
        phi = None if basis.compact else basis.phi()
        X = [basis.expiphi(w[i]) if abs(w[i]) > 1e-12 else None for _, w, _ in ham.cosines]
        ops.append({"n": n, "n2": n2, "phi": phi, "X": X})
    return ops


def _check_compact(ham):
    for i, kind in enumerate(ham.types):
        if kind != "c":
            continue
        if np.any(np.abs(ham.E_L[i]) > 1e-9) or abs(ham.flux_linear[i]) > 1e-9:
            raise ValueError(f"compact mode {i} carries an inductive or linear term")


def _mean_field_energy(ham, ex):
    N = ham.size
    E = ham.constant
    for i in range(N):
        E += 4 * ham.E_C[i, i] * ex["n2"][i] + 0.5 * ham.E_L[i, i] * ex["phi2"][i]
        E += ham.flux_linear[i] * ex["phi"][i]
        for j in range(i + 1, N):
            E += 8 * ham.E_C[i, j] * ex["n"][i] * ex["n"][j]
            E += ham.E_L[i, j] * ex["phi"][i] * ex["phi"][j]
    for k, (E_k, _, delta) in enumerate(ham.cosines):
        E -= E_k * (np.exp(1j * delta) * np.prod(ex["X"][k])).real
    return float(E)


def _mode_hamiltonian(ham, ops, ex, i):
    N = ham.size
    op = ops[i]
    H = 4 * ham.E_C[i, i] * op["n2"] + 0.5 * ham.E_L[i, i] * (op["phi"] ** 2 if op["phi"] is not None else 0)
    if op["phi"] is not None:
        H = H + ham.flux_linear[i] * op["phi"]
    for j in range(N):
        if j == i:
            continue
        H = H + 8 * ham.E_C[i, j] * ex["n"][j] * op["n"]
        if op["phi"] is not None and abs(ham.E_L[i, j]) > 0:
            H = H + ham.E_L[i, j] * ex["phi"][j] * op["phi"]
    for k, (E_k, _, delta) in enumerate(ham.cosines):
        X = op["X"][k]
        if X is None:
            continue
        z = np.exp(1j * delta) * np.prod([ex["X"][k][j] for j in range(N) if j != i])
        H = H - 0.5 * E_k * (z * X + np.conj(z) * X.conj().T)
    return H


def _expectations(ham, ops, states):
    N = ham.size
    ex = {"n": np.zeros(N), "n2": np.zeros(N), "phi": np.zeros(N), "phi2": np.zeros(N),
          "X": [np.ones(N, dtype=complex) for _ in ham.cosines]}
    for i, (op, psi) in enumerate(zip(ops, states)):
        ex["n"][i] = (psi.conj() @ op["n"] @ psi).real
        ex["n2"][i] = (psi.conj() @ op["n2"] @ psi).real
        if op["phi"] is not None:
            ex["phi"][i] = (psi.conj() @ op["phi"] @ psi).real
            ex["phi2"][i] = (psi.conj() @ (op["phi"] ** 2) @ psi).real
        for k, X in enumerate(op["X"]):
            if X is not None:
                ex["X"][k][i] = psi.conj() @ X @ psi
    return ex


def _initial(ham, phi0):
    """Expectations of modes displaced to ``phi0``, or delocalized for ``None``."""
    N = ham.size
    if phi0 is None:
        return {"n": np.zeros(N), "n2": np.zeros(N), "phi": np.zeros(N), "phi2": np.zeros(N),
                "X": [np.zeros(N, dtype=complex) for _ in ham.cosines]}
    phi0 = np.asarray(phi0, dtype=float)
    noncompact = np.array([kind != "c" for kind in ham.types])
    phi = np.where(noncompact, phi0, 0.0)
    return {"n": np.zeros(N), "n2": np.zeros(N), "phi": phi, "phi2": phi ** 2,
            "X": [np.exp(1j * w * phi0) for _, w, _ in ham.cosines]}


def default_starts(N):
    """Displacements of every mode to ``0`` or ``+-pi``, and the delocalized start.

    All ``3^N`` combinations for up to three modes; beyond that only the
    uniform displacements, since the count grows exponentially.
    """
    if N <= 3:
        starts = list(itertools.product((0.0, np.pi, -np.pi), repeat=N))
    else:
        starts = [(0.0,) * N, (np.pi,) * N, (-np.pi,) * N]
    return tuple(starts) + (None,)


def _scf(ham, bases, tol, maxiter, ex):
    ops = _mode_operators(ham, bases)
    N = ham.size
    states = [None] * N
    energy, mixing, rises = np.inf, 1.0, 0
    for iteration in range(1, maxiter + 1):
        for i in range(N):
            H = _mode_hamiltonian(ham, ops, ex, i)
            _, vectors = eigh(H, subset_by_index=(0, 0))
            states[i] = vectors[:, 0]
        new = _expectations(ham, ops, states)
        if mixing < 1:
            for key in ("n", "n2", "phi", "phi2"):
                new[key] = mixing * new[key] + (1 - mixing) * ex[key]
            new["X"] = [mixing * a + (1 - mixing) * b for a, b in zip(new["X"], ex["X"])]
        new_energy = _mean_field_energy(ham, new)
        change = max([np.max(np.abs(new[key] - ex[key])) for key in ("n", "phi")]
                     + [np.max(np.abs(a - b)) for a, b in zip(new["X"], ex["X"])] + [0.0])
        if new_energy > energy + 1e-12:
            rises += 1
            if rises >= 2:
                mixing = 0.5
        done = abs(new_energy - energy) < tol * max(1.0, abs(new_energy)) and change < np.sqrt(tol)
        ex, energy = new, new_energy
        if done:
            return energy, states, ex, iteration, True
    return energy, states, ex, maxiter, False


def product_state(ham, *, bases=None, tol=1e-11, maxiter=300, adapt=True, edge_tol=1e-8,
                  starts=None, warm=None, **basis_kwargs):
    """Self-consistent product ground state of ``ham``.

    The mean-field iteration has several fixed points, and a symmetric one can
    be a saddle of the product energy, so it is run from every displacement in
    ``starts`` (:func:`default_starts` by default) and from the expectations
    ``warm`` of an earlier solve, and the lowest energy is kept.
    With ``adapt=True`` a phase grid whose ground state reaches its edge is
    enlarged and the solve repeated (up to three times).
    """
    _check_compact(ham)
    bases = list(bases) if bases is not None else default_bases(ham, **basis_kwargs)
    if starts is None:
        starts = default_starts(ham.size)
    initial = [_initial(ham, phi0) for phi0 in starts]
    if warm is not None:
        initial.append(warm)
    for _ in range(4):
        runs = [_scf(ham, bases, tol, maxiter, ex0) for ex0 in initial]
        energy, states, ex, iterations, converged = min(runs, key=lambda run: run[0])
        if not adapt:
            break
        grown = False
        for i, basis in enumerate(bases):
            if not basis.compact and basis.edge_weight(states[i]) > edge_tol:
                bases[i] = PhaseGrid(1.5 * basis.extent, int(np.ceil(1.5 * basis.points)))
                grown = True
        if not grown:
            break
    return ProductState(energy, tuple(states), tuple(bases), iterations, converged, ex)


# ---------------------------------------------------------------- optimization


def cost_product_state(ham, *, regularization=0.0, warm_start=True, **kwargs):
    """Return ``f(S)``: the product-state energy in the coordinates ``S``.

    ``regularization`` adds that multiple of the scale-free normal-mode cost
    ``eq:numerical_normal_cost`` as a tie-break.  The product energy alone is
    degenerate in the harmonic limit, where the ground state factorizes in
    every frame that diagonalizes its Gaussian, not only in the normal modes.
    With ``warm_start`` each solve also starts from the previous solution.
    """
    quadratic = cost_normal_modes(np.linalg.inv(ham.E_C), ham.E_L, ham.types)
    cache = {"warm": None}

    def objective(S):
        S = np.asarray(S, dtype=float)
        if not np.all(np.isfinite(S)) or abs(np.linalg.det(S)) < 1e-12:
            return np.inf
        try:
            state = product_state(ham.transformed(S), warm=cache["warm"], **kwargs)
        except (ValueError, np.linalg.LinAlgError):
            return np.inf
        if warm_start:
            cache["warm"] = state.expectations
        value = state.energy
        if regularization:
            value += regularization * quadratic(S)
        return value

    return objective


@dataclass(frozen=True)
class ProductStateInfo:
    result: OptimizeResult
    energy: float
    S: np.ndarray
    state: ProductState


def product_state_S1(ham, *, S_cc=None, relaxed=True, x0=None, restarts=0, seed=0,
                     gtol=1e-6, eps=1e-6, normalize="C", regularization=0.0, **kwargs):
    """Minimize the product-state energy over ``S(x)`` of ``eq:S_numerical``.

    Returns ``(S, info)`` with the noncompact columns of ``S`` normalized so
    that ``(S^T C S)_jj = 1`` in units where ``C = E_C^{-1}`` (``normalize="C"``),
    left as found (``None``), or scaled to the reference diagonal (``"keep"``).
    ``info.energy`` is the product-state energy at ``S`` without the
    ``regularization`` tie-break.
    """
    p = parameterization(ham.types, S_cc=S_cc, relaxed=relaxed)
    objective_S = cost_product_state(ham, regularization=regularization, **kwargs)
    objective_x = lambda x: objective_S(p.S(x))
    starts = [np.asarray(x0, dtype=float) if x0 is not None else p.x0()]
    rng = np.random.default_rng(seed)
    starts.extend(0.3 * rng.normal(size=p.size) for _ in range(restarts))
    if p.size:
        results = [minimize(objective_x, start, method="BFGS",
                            options={"gtol": gtol, "eps": eps, "maxiter": 500})
                   for start in starts]
    else:
        results = [OptimizeResult(x=p.x0(), fun=objective_x(p.x0()), success=True,
                                  message="the parameterization has no free variables")]
    finite = [r for r in results if np.isfinite(r.fun)]
    if not finite:
        raise RuntimeError("every restart returned a non-finite product-state energy")
    best = min(finite, key=lambda r: r.fun)
    S = p.S(best.x)
    if normalize is not None:
        C = np.linalg.inv(ham.E_C)
        Cp = S.T @ C @ S
        for j, kind in enumerate(ham.types):
            if kind != "c":
                target = 1.0 if normalize == "C" else C[j, j]
                S[:, j] *= np.sqrt(target / Cp[j, j])
    state = product_state(ham.transformed(S), **kwargs)
    return S, ProductStateInfo(best, state.energy, S, state)


def normal_mode_transformation(ham):
    """The relaxed-family normal modes of Sec. III.B for ``ham`` (``C = E_C^{-1}``)."""
    N = ham.size
    c = [i for i, kind in enumerate(ham.types) if kind == "c"]
    n = [i for i, kind in enumerate(ham.types) if kind != "c"]
    C = np.linalg.inv(ham.E_C)
    S = np.eye(N)
    if not n:
        return S
    C_bar = C[np.ix_(n, n)]
    if c:
        C_cc_inv = np.linalg.inv(C[np.ix_(c, c)])
        C_bar = C_bar - C[np.ix_(n, c)] @ C_cc_inv @ C[np.ix_(c, n)]
    _, V = eigh(ham.E_L[np.ix_(n, n)], C_bar)
    S[np.ix_(n, n)] = V
    if c:
        S[np.ix_(c, n)] = -C_cc_inv @ C[np.ix_(c, n)] @ V
    return S


# ---------------------------------------------------------------- exact reference


def _sparse_ops(basis):
    ops = {"n": sp.csr_matrix(basis.n()), "n2": sp.csr_matrix(basis.n2())}
    ops["phi"] = None if basis.compact else sp.csr_matrix(basis.phi())
    return ops


def exact_hamiltonian(ham, bases, max_dim=200_000):
    """The full Hamiltonian on the kron product of ``bases`` as a sparse matrix."""
    _check_compact(ham)
    N = ham.size
    dims = [b.size for b in bases]
    if int(np.prod(dims)) > max_dim:
        raise ValueError(f"product dimension {np.prod(dims)} exceeds max_dim={max_dim}")

    def embed(factors):
        out = sp.identity(1, format="csr")
        for i in range(N):
            out = sp.kron(out, factors.get(i, sp.identity(dims[i], format="csr")), format="csr")
        return out

    ops = [_sparse_ops(b) for b in bases]
    I = [sp.identity(d, format="csr") for d in dims]
    n_t = [ops[i]["n"] - ham.n_g[i] * I[i] for i in range(N)]
    n2_t = [ops[i]["n2"] - 2 * ham.n_g[i] * ops[i]["n"] + ham.n_g[i] ** 2 * I[i] for i in range(N)]
    H = ham.constant * embed({})
    for i in range(N):
        H = H + 4 * ham.E_C[i, i] * embed({i: n2_t[i]})
        if ops[i]["phi"] is not None:
            H = H + 0.5 * ham.E_L[i, i] * embed({i: ops[i]["phi"] @ ops[i]["phi"]})
            H = H + ham.flux_linear[i] * embed({i: ops[i]["phi"]})
        for j in range(i + 1, N):
            H = H + 8 * ham.E_C[i, j] * embed({i: n_t[i], j: n_t[j]})
            if ops[i]["phi"] is not None and ops[j]["phi"] is not None:
                H = H + ham.E_L[i, j] * embed({i: ops[i]["phi"], j: ops[j]["phi"]})
    for E_k, w, delta in ham.cosines:
        X = embed({i: sp.csr_matrix(bases[i].expiphi(w[i])) for i in range(N) if abs(w[i]) > 1e-12})
        X = np.exp(1j * delta) * X
        H = H - 0.5 * E_k * (X + X.conj().T)
    return H.tocsr()


def exact_ground_state(ham, bases, *, k=1, guess=None, tol=1e-9, maxiter=2000, **kwargs):
    """Lowest ``k`` eigenpairs of :func:`exact_hamiltonian`.

    Returns ``(energy, vector)`` for ``k == 1`` and ``(energies, vectors)``
    with ascending energies and vectors as columns otherwise.  Small problems
    are diagonalized densely.  Large ones use LOBPCG with a diagonal
    preconditioner, seeded with ``guess`` (typically the product state vector,
    which is close to the answer) and random vectors.
    """
    H = exact_hamiltonian(ham, bases, **kwargs)
    n = H.shape[0]
    if n <= 2000:
        values, vectors = np.linalg.eigh(H.toarray())
        values, vectors = values[:k], vectors[:, :k]
    else:
        rng = np.random.default_rng(0)
        block = k + 3
        X = rng.normal(size=(n, block)) + 1j * rng.normal(size=(n, block))
        if guess is not None:
            X[:, 0] = np.asarray(guess, dtype=complex)
        diagonal = H.diagonal().real
        shift = diagonal.min() - 1.0
        M = sp.diags(1.0 / (diagonal - shift)).tocsr()
        values, vectors = spla.lobpcg(H, X, M=M, largest=False, tol=tol, maxiter=maxiter)
        order = np.argsort(values)[:k]
        values, vectors = values[order].real, vectors[:, order]
    if k == 1:
        return float(values[0]), vectors[:, 0]
    return np.asarray(values, dtype=float), vectors


def subspace_infidelity(state, exact_vectors):
    """``1 - sum_s |<product|s>|^2`` over the columns of ``exact_vectors``.

    For a near-degenerate ground multiplet (a cos 2 phi transmon's doublet) the
    product state localizes in one well and its overlap with any single member
    is about one half; the overlap with the multiplet is the useful measure.
    """
    v = state.vector()
    return float(1 - sum(abs(np.vdot(v, exact_vectors[:, s])) ** 2 for s in range(exact_vectors.shape[1])))


def infidelity(state, exact_vector):
    """``1 - |<product|exact>|^2`` for a product state on the same bases."""
    overlap = np.vdot(state.vector(), exact_vector)
    return float(1 - abs(overlap) ** 2)
