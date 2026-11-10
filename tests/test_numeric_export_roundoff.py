"""Numeric export regression for the Julia residual-frame workload."""
import numpy as np
import sympy as sym
import circuit_quantization as cq
from circuits import coupled_zero_pis


def test_numerified_coupled_zero_pis_random_float_frame_exports():
    circuit = coupled_zero_pis()
    loops = cq.Loops.from_tree(circuit, cq.default_tree(circuit))
    loops = cq.Loops(circuit, loops.G, loops.names, loops.closure_branches, dict.fromkeys(loops.names, 0))
    result = cq.quantize(circuit, loops)
    parameters = set(circuit.variables)
    values = {s: (1e-15 * (i+1) if s.name.startswith('C') else
                  1e-8 * (i+1) if s.name.startswith('L') else 2.)
              for i, s in enumerate(sorted(parameters, key=str))}
    result = cq.numerify(result, values)
    rng = np.random.default_rng(1042)
    S = np.eye(len(result.coords.types))
    for i, old in enumerate(result.coords.types):
        for j, new in enumerate(result.coords.types):
            if new != 'c' and (old, new) in {('c','e'),('c','h'),('e','e'),('e','h'),('h','e'),('h','h')}:
                S[i,j] += rng.normal(scale=.12)
    data = cq.to_spec(result.transform(S))
    assert cq.load_spec(data).E_L == cq.load_spec(data).E_L.T
