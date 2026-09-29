"""End-to-end pipeline: Sec I -> II -> III -> V -> IV."""

from collections import Counter
from dataclasses import dataclass

import sympy as sym

from .circuit import Circuit, Loops
from .flux import irrotational_B
from .hamiltonian import Hamiltonian, hamiltonian
from .lagrangian import Lagrangian, lagrangian
from .modes import Coordinates, check_basis, check_coordinates, decouple, mode_basis, residual_freedom, transform


@dataclass(frozen=True)
class Result:
    circuit: Circuit
    loops: Loops
    lag: Lagrangian
    coords: Coordinates
    B: sym.ImmutableMatrix
    ham: Hamiltonian

    @property
    def H(self):
        return self.ham.H

    def residual_freedom(self, keep_harmonic=True):
        """Symbolic S_1 allowed on these coordinates, in coords.types order (eq:S1_structure)."""
        return residual_freedom(self.coords, keep_harmonic)

    def transform(self, S1, *, simplify=False):
        """The same circuit and loops quantized in coordinates Z S_1 (Sec III.C)."""
        return _result(self.circuit, self.loops, self.lag, transform(self.lag, self.coords, S1), simplify)

    def _repr_latex_(self):
        counts = ", ".join(f"{t}: {k}" for t, k in Counter(self.coords.types).items())
        return f"$\\displaystyle H_{{\\mathrm{{GHz}}}} = {sym.latex(self.H)}$  (modes {counts})"


def quantize(circuit, loops, Z_full=None, types=None, *, simplify=False):
    """Quantize with an automatic or explicit basis; optionally simplify the final H."""
    if loops.circuit != circuit:
        raise ValueError("loops must belong to the circuit being quantized")
    lag = lagrangian(circuit)
    if Z_full is None and types is None:
        Z_full, types = mode_basis(lag)
    elif Z_full is None or types is None:
        raise ValueError("Z_full and types must be supplied together")
    check_basis(lag, Z_full, types)
    coords = decouple(lag, Z_full, types)
    check_coordinates(lag, coords)
    return _result(circuit, loops, lag, coords, simplify)


def _result(circuit, loops, lag, coords, simplify):
    """Flux assignment and Hamiltonian for given coordinates (Sec V, Sec IV)."""
    B = irrotational_B(lag, coords, loops)
    ham = hamiltonian(lag, coords, loops, B, circuit.t, simplify=simplify)
    return Result(circuit=circuit, loops=loops, lag=lag, coords=coords, B=B, ham=ham)
