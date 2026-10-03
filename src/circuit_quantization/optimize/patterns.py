"""Search orders for choose_S1: which sets of couplings to try to zero, cheapest first.

A search order takes the coupling terms of one decoupling stage, as keys
(name, i, j) for the off-diagonal entry (i, j) of the matrix called name, and
yields (cost, term indices) in non-decreasing cost. The solver returns every
solvable set at the first cost that has one, so a good order reaches the best
patterns without solving the ones that cannot win.
"""

from .equationset import largest_subsets_first


def most_zeros_first(term_keys):
    """sircuitenum's order: every set of couplings, largest first (cost = -size)."""
    return largest_subsets_first(len(term_keys))


def by_cost(cost):
    """Order every set of couplings by cost(keys), keys a tuple of (name, i, j).

    Visits all 2^n subsets, so it suits few couplings; ties keep most_zeros_first's
    order. For many couplings, write an order that yields its candidates directly.
    """
    def order(term_keys):
        subsets = [keys for _, keys in largest_subsets_first(len(term_keys))]
        return sorted(((cost(tuple(term_keys[k] for k in keys)), keys) for keys in subsets),
                      key=lambda pair: pair[0])
    return order
