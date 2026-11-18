"""How far to shear a harmonic mode onto a junction mode: the rule of docs/shear-rule.md.

In a frame where a junction mode ``J`` (a phase inside a cosine) and a harmonic mode ``f`` couple only through
``E_L[f, J] x_f x_J``, the shear ``x_f = y_f + a x_J`` by the full amount ``a* = -E_L[f, J] / E_L[f, f]`` removes that
coupling and puts a charge coupling in its place. When a hierarchical solve truncates between ``f`` and ``J`` the best
shear is usually partial: the fraction

    s* = (X + c) / (X + 2 c + P),   c = 4 E_C[J, J] / Omega_f,   P = (8 E_C[J, J] / Omega_f)^2 <n_J^2>,

of ``a*``, with ``X = <dx_J^2>`` and ``<n_J^2>`` the ground-state moments of the junction mode's own Hamiltonian and
``Omega_f = sqrt(8 E_C[f, f] E_L[f, f])``. Equivalently ``s* = Omega_f / (Omega_f + omega_eff)``: full shear for a
frame mode much stiffer than the junction's motion, none for a much softer one. The moments come from a grid solve of
the one-mode Hamiltonian (a few hundred points, eighth-order stencils), and since the shear changes that Hamiltonian
and ``Omega_f``, the fractions are iterated to self-consistency.

The pairs are given explicitly: the shear only matters for a pair that the solve truncates between, which depends on
the tree the caller has in mind.
"""
from dataclasses import dataclass, field

import numpy as np

from .product_state import PhaseGrid


@dataclass(frozen=True)
class JunctionMoments:
    """Ground-state moments of junction mode ``J``'s own Hamiltonian in the final frame, and its grid."""

    J: int
    dx2: float          # <(x - <x>)^2>
    n2: float           # <n^2>
    center: float
    extent: float
    points: int
    edge_weight: float


@dataclass(frozen=True)
class PairShear:
    """The rule for one (harmonic, junction) pair in the final frame."""

    f: int
    J: int
    Omega: float        # frame-mode frequency sqrt(8 E_C[f, f] E_L[f, f]), GHz
    omega1: float       # 4 E_C[J, J] / <dx^2>
    omega2: float       # 8 E_C[J, J] sqrt(<n^2> / <dx^2>)
    a_star: float       # the full shear -E_L[f, J] / E_L[f, f] of the input frame
    s: float            # the fraction s*
    A: float            # the applied entry S[f, J] = s* a*


@dataclass(frozen=True)
class ShearInfo:
    pairs: list
    junctions: dict
    iterations: int
    converged: bool
    change: float       # largest change of any s* in the last iteration
    history: list = field(default_factory=list)

    def __str__(self):
        lines = [f"shear rule: {self.iterations} iteration(s), " +
                 ("converged" if self.converged else "not converged") + f" (last change {self.change:.1e})",
                 "  f   J    Omega   omega1   omega2      a*      s*       S[f, J]"]
        for p in self.pairs:
            lines.append(f"{p.f:3d} {p.J:3d} {p.Omega:8.3f} {p.omega1:8.3f} {p.omega2:8.3f} {p.a_star:8.4f} "
                         f"{p.s:7.4f} {p.A:12.6g}")
        lines.append("  J   <dx^2>    <n^2>   grid (center, extent, points)   edge weight")
        for m in self.junctions.values():
            lines.append(f"{m.J:3d} {m.dx2:8.4f} {m.n2:8.4f}   ({m.center:.3f}, {m.extent:.2f}, {m.points})"
                         f"   {m.edge_weight:.1e}")
        return "\n".join(lines)


def _junction_cosines(ham, J):
    """``(E, k, delta)`` of the cosines on mode ``J``, which must act on ``J`` alone."""
    terms = []
    for E, w, d in ham.cosines:
        if abs(w[J]) <= 1e-12:
            continue
        others = np.delete(w, J)
        if np.any(np.abs(others) > 1e-12):
            raise ValueError(f"mode {J} appears in a cosine with other modes ({w}); the rule needs single-mode "
                             "junction cosines")
        terms.append((E, w[J], d))
    return terms


def _grid(EC, EL, terms, extent_scale, max_points):
    """Extent and point count sized like ``default_bases`` for one mode."""
    zpf = (2 * EC / EL) ** 0.25
    omega = np.sqrt(8 * EC * EL)
    energy = sum(abs(E) for E, _, _ in terms)
    k_max = max((abs(k) for _, k, _ in terms), default=0.0)
    extent = max(6 * zpf, np.sqrt(2 * (2 * energy + 12 * omega + 4 * EC) / EL)) * extent_scale
    step = zpf / 6
    if k_max > 0:
        step = min(step, np.pi / (6 * k_max))
    return extent, min(int(np.ceil(2 * extent / step)) + 1, max_points)


def _moments(ham, J, *, extent_scale, max_points, edge_tol, widenings=6):
    """Ground-state moments of ``4 E_C n^2 + 1/2 E_L x^2 + b x - sum E cos(k x + delta)`` for mode ``J``.

    The offset charge of an extended mode is a gauge choice and drops out. The grid is centered on the minimum of the
    quadratic part and widened until the ground state's weight on the two end points is below ``edge_tol``.
    """
    EC, EL, b = ham.E_C[J, J], ham.E_L[J, J], ham.flux_linear[J]
    if EC <= 0 or EL <= 0:
        raise ValueError(f"junction mode {J} needs positive E_C and E_L in this frame, got {EC}, {EL}")
    terms = _junction_cosines(ham, J)
    extent, points = _grid(EC, EL, terms, extent_scale, max_points)
    center = -b / EL
    for _ in range(widenings):
        grid = PhaseGrid(extent, points)
        x = center + grid.grid
        V = 0.5 * EL * x**2 + b * x - sum(E * np.cos(k * x + d) for E, k, d in terms)
        H = 4 * EC * np.real(grid.n2()) + np.diag(V)
        _, vec = np.linalg.eigh(H)
        psi = vec[:, 0] / np.linalg.norm(vec[:, 0])
        edge = grid.edge_weight(psi)
        if edge <= edge_tol:
            break
        extent, points = 1.5 * extent, int(np.ceil(1.5 * points))
    else:
        raise ValueError(f"junction mode {J}: the ground state still reaches the grid edge (weight {edge:.1e}) on "
                         f"±{extent / 1.5:.3g}; raise extent_scale")
    p = psi**2
    mean = p @ x
    dx2 = p @ (x - mean) ** 2
    n2 = psi @ (np.real(grid.n2()) @ psi)
    return JunctionMoments(J, float(dx2), float(n2), float(center), float(extent), int(points), float(edge))


def shear_S1(ham, pairs, *, tol=1e-6, maxiter=50, extent_scale=1.0, max_points=400, edge_tol=1e-8):
    """Shear each harmonic mode ``f`` onto junction mode ``J`` by the rule's fraction, for the given ``(f, J)`` pairs.

    :param ham: NumericHamiltonian in a frame where each pair couples through ``E_L[f, J]`` and not through ``E_C``.
    :param pairs: ``(f, J)`` mode indices in ``ham.types`` order: ``f`` harmonic (``"h"``), ``J`` a junction mode
        (``"e"``) whose cosines act on it alone, ``E_L[f, J] != 0``, ``E_C[f, J] == 0``. A frame mode may be paired
        with several junctions and a junction with several frame modes.
    :param tol: Stop when no fraction changes by more than ``tol`` between iterations.
    :param maxiter: At most this many iterations (each one grid solve per junction).
    :param extent_scale, max_points, edge_tol: The junction grids (sized like ``default_bases``, widened until the
        ground state's edge weight is below ``edge_tol``).
    :returns: ``(S, info)``: ``S`` with ``x_old = S x_new``, the identity plus ``S[f, J] = s* a*``, for
        ``ham.transformed(S)`` or ``Result.transform(S)``; ``info`` a ShearInfo (printable table).
    :raises ValueError: An invalid pair, or one the rule does not cover.
    """
    pairs = [(int(f), int(J)) for f, J in pairs]
    if not pairs:
        raise ValueError("give at least one (f, J) pair")
    if len(set(pairs)) != len(pairs):
        raise ValueError("a pair is given twice")
    N = ham.size
    scale_L, scale_C = np.max(np.abs(ham.E_L)), np.max(np.abs(ham.E_C))
    a_star = {}
    for f, J in pairs:
        if not (0 <= f < N and 0 <= J < N) or f == J:
            raise ValueError(f"pair ({f}, {J}): mode indices must be distinct and below {N}")
        if ham.types[f] != "h" or ham.E_L[f, f] <= 0:
            raise ValueError(f"pair ({f}, {J}): mode {f} must be harmonic with E_L[f, f] > 0, "
                             f"got type {ham.types[f]!r}")
        if ham.types[J] != "e" or not _junction_cosines(ham, J):
            raise ValueError(f"pair ({f}, {J}): mode {J} must be an extended junction mode (in a cosine), "
                             f"got type {ham.types[J]!r}")
        if abs(ham.E_L[f, J]) <= 1e-12 * scale_L:
            raise ValueError(f"pair ({f}, {J}): E_L[f, J] = 0, nothing to shear")
        if abs(ham.E_C[f, J]) > 1e-12 * scale_C:
            raise ValueError(f"pair ({f}, {J}): E_C[f, J] = {ham.E_C[f, J]}; the rule needs the pair coupled through "
                             "E_L only")
        a_star[f, J] = -ham.E_L[f, J] / ham.E_L[f, f]
    junction_modes = sorted({J for _, J in pairs})
    s = {pair: 0.0 for pair in pairs}
    S = np.eye(N)
    history, change, converged = [], np.inf, False
    for iteration in range(1, maxiter + 1):
        sheared = ham.transformed(S)
        moments = {J: _moments(sheared, J, extent_scale=extent_scale, max_points=max_points, edge_tol=edge_tol)
                   for J in junction_modes}
        rows, new = [], {}
        for f, J in pairs:
            m, EC = moments[J], sheared.E_C[J, J]
            Omega = np.sqrt(8 * sheared.E_C[f, f] * sheared.E_L[f, f])
            c = 4 * EC / Omega
            P = (8 * EC / Omega) ** 2 * m.n2
            new[f, J] = (m.dx2 + c) / (m.dx2 + 2 * c + P)
            rows.append((f, J, Omega, 4 * EC / m.dx2, 8 * EC * np.sqrt(m.n2 / m.dx2)))
        change = max(abs(new[p] - s[p]) for p in pairs)
        s = new
        history.append(dict(s))
        S = np.eye(N)
        for f, J in pairs:
            S[f, J] = s[f, J] * a_star[f, J]
        if change <= tol:
            converged = True
            break
    info_pairs = [PairShear(f, J, float(Om), float(w1), float(w2), float(a_star[f, J]), float(s[f, J]),
                            float(S[f, J])) for f, J, Om, w1, w2 in rows]
    return S, ShearInfo(info_pairs, moments, iteration, converged, float(change), history)
