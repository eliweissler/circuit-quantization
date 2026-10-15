"""Search orders for choose_S1: which sets of couplings to try to zero, cheapest first.

A search order takes the coupling terms of one decoupling stage, as keys
(name, i, j) for the off-diagonal entry (i, j) of the matrix called name, and
yields (cost, term indices) in non-decreasing cost. The solver returns every
solvable set at the first cost that has one, so a good order reaches the best
patterns without solving the ones that cannot win. An order with a seed parameter
also gets the seed being refined, as seed=Seed(var_types, wJ_T), from choose_S1.
"""

from itertools import product
from typing import NamedTuple

import sympy as sym
from sympy.utilities.iterables import multiset_partitions

from ..modes import _cluster_cost
from .equationset import largest_subsets_first


class Seed(NamedTuple):
    """The seed a stage refines, in its (compact, extended, harmonic) column order."""

    var_types: dict  # sircuitenum's var_types: column indices of each mode type
    wJ_T: sym.Matrix  # omega_J~^T times the seed, one row per junction


def most_zeros_first(term_keys, seed=None):
    """sircuitenum's order: every set of couplings, largest first (cost = -size)."""
    return largest_subsets_first(len(term_keys))


def by_cost(cost):
    """Order every set of couplings by cost(keys), keys a tuple of (name, i, j).

    Visits all 2^n subsets, so it suits few couplings; ties keep most_zeros_first's
    order. For many couplings, write an order that yields its candidates directly.
    """
    def order(term_keys, seed=None):
        subsets = [keys for _, keys in largest_subsets_first(len(term_keys))]
        return sorted(((cost(tuple(term_keys[k] for k in keys)), keys) for keys in subsets),
                      key=lambda pair: pair[0])
    return order


def by_clusters(dims=None, name="wJ"):
    """Junction clusters, cheapest first, for a rows stage on the junction incidence (built-in "wJ").

    A pattern is a partition of the modes into clusters together with a cluster for each
    junction; its terms are that junction's entries (name, k, j) in noncompact columns j
    outside its cluster, which are linear in S_1, and its cost is cluster_cost's tuple for
    the partition. A stage cannot change compact columns, so a junction acting on a compact
    mode must sit in that mode's cluster, and partitions splitting a junction's compact
    modes are skipped without solving. Equal term sets keep their cheapest partition.

    :param dims: Basis size per mode type, as in cluster_cost.
    :param name: The rows target holding the junction incidence.
    :returns: An order for Stage(rows=(name,), order=...); it needs choose_S1's seed.
    """
    def order(term_keys, seed=None):
        if seed is None:
            raise ValueError("by_clusters needs the seed's mode types and junction incidence; use it in choose_S1")
        W = seed.wJ_T
        n = W.cols
        types = [None] * n
        for kind, t in (("compact", "c"), ("extended", "e"), ("harmonic", "h")):
            for j in seed.var_types[kind]:
                types[j] = t
        index = {key: i for i, key in enumerate(term_keys)}
        compact = [{j for j in seed.var_types["compact"] if sym.cancel(W[k, j]) != 0} for k in range(W.rows)]
        patterns = []
        for partition in multiset_partitions(list(range(n))) if n else []:
            cluster_of = {j: a for a, block in enumerate(partition) for j in block}
            choices = [sorted({cluster_of[j] for j in compact[k]}) or range(len(partition)) for k in range(W.rows)]
            if any(len(c) > 1 and compact[k] for k, c in enumerate(choices)):
                continue
            cost = _cluster_cost([tuple(block) for block in partition], types, dims)
            for assignment in product(*choices):
                keys = tuple(sorted(index[(name, k, j)] for k, a in enumerate(assignment)
                                    for j in range(n) if cluster_of[j] != a and (name, k, j) in index))
                patterns.append((cost, keys))
        seen, ordered = set(), []
        for cost, keys in sorted(patterns, key=lambda pattern: pattern[0]):
            if keys not in seen:
                seen.add(keys)
                ordered.append((cost, keys))
        return ordered
    return order
