"""Optional search for the residual transformation S_1 (Sec III.C), ported from sircuitenum.

Needs the optional dependencies: the Singular binary (conda-forge ``singular``),
``z3-solver``, and ``func-timeout``; ``environment.yml`` installs all three.
Nothing in the core package imports this subpackage.

    from circuit_quantization.optimize import choose_S1
    S1s, h_class = choose_S1(result.lag, result.coords)
    best = result.transform(S1s[0])
"""

from .choose import Candidate, choose_S1, h_hash
from .patterns import by_cost, most_zeros_first

__all__ = ["Candidate", "by_cost", "choose_S1", "h_hash", "most_zeros_first"]
