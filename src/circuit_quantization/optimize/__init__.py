"""Optional search for the residual transformation S_1 (Sec III.C), ported from sircuitenum.

Needs the optional dependencies: the Singular binary (conda-forge ``singular``),
``z3-solver``, and ``func-timeout``; ``environment.yml`` installs all three.
Nothing in the core package imports this subpackage.

    from circuit_quantization.optimize import choose_instance, choose_S1
    S1s, h_class = choose_S1(result.lag, result.coords)        # families, free entries symbolic
    S1, weight = choose_instance(result.lag, result.coords, S1s[0])
    best = result.transform(S1)
"""

from .choose import (
    Candidate, candidate, choose_instance, choose_S1, cluster_score, compact_alignments, h_hash, instance,
    junction_seeds,
)
from .patterns import Seed, by_clusters, by_cost, most_zeros_first
from .stages import ALLOWED_BLOCKS, SIRCUITENUM, SIRCUITENUM_BLOCKS, Stage

__all__ = [
    "ALLOWED_BLOCKS", "Candidate", "SIRCUITENUM", "SIRCUITENUM_BLOCKS", "Seed", "Stage", "by_clusters", "by_cost",
    "candidate", "choose_S1", "choose_instance", "cluster_score", "compact_alignments", "h_hash", "instance",
    "junction_seeds", "most_zeros_first",
]
