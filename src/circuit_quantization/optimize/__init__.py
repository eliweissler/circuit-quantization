"""Optional search for the residual transformation S_1 (Sec III.C), ported from sircuitenum.

Needs the optional dependencies: the Singular binary (conda-forge ``singular``),
``z3-solver``, and ``func-timeout``; ``environment.yml`` installs all three.
Nothing in the core package imports this subpackage.

    from circuit_quantization.optimize import choose_S1
    S1s, h_class = choose_S1(result.lag, result.coords)
    best = result.transform(S1s[0])
"""

from .choose import choose_S1

__all__ = ["choose_S1"]
