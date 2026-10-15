"""Decoupling stages for choose_S1: which couplings the solver tries to zero, in sequence.

A Stage names the matrices whose couplings become solver terms, each term keyed
(name, i, j). Square matrices contribute their upper off-diagonal entries;
rectangular matrices, one column per retained mode, contribute their entries in
noncompact columns. Built-in names:

- "J": junction matrix omega_J diag(E_J) omega_J^T, one placeholder E_J per junction
- "L": inductive matrix L~*
- "Cinv": charging matrix C~^{-1}
- "wJ" (rows): junction incidence omega_J~^T, one row per junction

A custom entry is (name, matrix) or (name, matrix, law), with the matrix in
coords.types column order. A square matrix transforms by congruence S^T A S
(law "flux", the default, like L~*) or contragrediently S^{-1} G S^{-T} (law
"charge", like C~^{-1}); a rectangular one as M S, like omega_J~^T.
"""

from dataclasses import dataclass
from typing import NamedTuple

import sympy as sym

from .patterns import most_zeros_first


@dataclass(frozen=True)
class Stage:
    """One decoupling stage: square and rectangular matrices, and the search order over their terms."""

    square: tuple = ()
    rows: tuple = ()
    order: object = most_zeros_first


# sircuitenum's choose_Z: junction couplings first, then inductive and charging couplings.
SIRCUITENUM = (Stage(square=("J",)), Stage(square=("L", "Cinv")))

# (old type, new type) blocks of S_1 the first stage opens as free entries, after the seed.
# sircuitenum's trans_blocks: compact into extended, extended and harmonic into extended,
# harmonic into harmonic. Compact into compact belongs to the seed, which keeps it integer.
SIRCUITENUM_BLOCKS = frozenset({("c", "e"), ("e", "e"), ("h", "e"), ("h", "h")})
ALLOWED_BLOCKS = frozenset({("c", "e"), ("c", "h"), ("e", "e"), ("e", "h"), ("h", "e"), ("h", "h")})


def trans_blocks(blocks):
    """sircuitenum's 3x3 trans_blocks, rows and columns (compact, extended, harmonic)."""
    blocks = frozenset(blocks)
    if blocks - ALLOWED_BLOCKS:
        raise ValueError(f"blocks {sorted(blocks - ALLOWED_BLOCKS)} are not free S_1 blocks; "
                         f"allowed: {sorted(ALLOWED_BLOCKS)} (compact into compact belongs to the seed)")
    order = ("c", "e", "h")
    return [[int((old, new) in blocks) for new in order] for old in order]

LAWS = ("flux", "charge")


class Target(NamedTuple):
    """A resolved stage matrix in (compact, extended, harmonic) order."""

    name: str
    matrix: sym.Matrix  # square: in its own law; rows: one column per mode
    law: str  # "flux", "charge", or "rows"
    solver: sym.Matrix  # what secondary_decouple transforms by congruence (rows: as M Z)
    invert: bool  # secondary_decouple inverts the congruence (charge law)


def resolve(stage, builtins, P):
    """The stage's Targets; builtins maps names to Targets, P reorders coords columns."""
    for entry in stage.square + stage.rows:
        if isinstance(entry, str) and entry not in builtins:
            raise ValueError(f"Unknown stage matrix {entry!r}; built-ins are {sorted(builtins)}")
    square, rows = [], []
    for entry in stage.square:
        if isinstance(entry, str):
            target = builtins[entry]
            if target is not None:
                square.append(target)
            continue
        name, matrix, law = (*entry, "flux") if len(entry) == 2 else entry
        if law not in LAWS:
            raise ValueError(f"Stage matrix {name!r}: law must be 'flux' or 'charge', not {law!r}")
        A = sym.Matrix(matrix)
        if A.shape != (P.rows, P.rows) or A != A.T:
            raise ValueError(f"Stage matrix {name!r} must be symmetric with one row and column per retained mode")
        A = P.T * A * P
        square.append(Target(name, A, law, A if law == "flux" else A.inv(), law == "charge"))
    for entry in stage.rows:
        if isinstance(entry, str):
            target = builtins[entry]
            if target is not None:
                rows.append(target)
            continue
        name, matrix = entry
        M = sym.Matrix(matrix)
        if M.cols != P.rows:
            raise ValueError(f"Stage matrix {name!r} needs one column per retained mode")
        M = M * P
        rows.append(Target(name, M, "rows", M, False))
    return square, rows


def zero_keys(targets, S, compact):
    """Keys (name, i, j) of the targets' entries that vanish after S, as in the solver terms."""
    S = sym.Matrix(S)
    zeros = set()
    for t in targets:
        if t.law == "rows":
            T = t.matrix * S
            entries = [(k, j) for k in range(T.rows) for j in range(T.cols) if j not in compact]
        else:
            T = S.T * t.matrix * S if t.law == "flux" else S.inv() * t.matrix * S.inv().T
            entries = [(i, j) for i in range(T.rows) for j in range(i + 1, T.cols)]
        zeros.update((t.name, i, j) for i, j in entries if sym.cancel(T[i, j]) == 0)
    return frozenset(zeros)
