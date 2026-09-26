"""End-to-end pipeline: Sec I -> II -> III -> V -> IV."""

from collections import Counter
from dataclasses import dataclass

import sympy as sym

from .circuit import Circuit, Loops
from .flux import irrotational_B
from .hamiltonian import Hamiltonian, hamiltonian
from .lagrangian import Lagrangian, lagrangian
from .modes import Coordinates, check_basis, decouple, mode_basis


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

    def branch_index(self, name):
        return self.circuit.branch_names.index(name)

    def __getattr__(self, name):
        # Expose every named matrix (res.Z, res.C_int, res.E_C, ...) directly.
        for part in ("lag", "coords", "ham", "loops"):
            obj = object.__getattribute__(self, part)
            if hasattr(obj, name):
                return getattr(obj, name)
        raise AttributeError(name)

    def _repr_latex_(self):
        counts = ", ".join(f"{t}: {k}" for t, k in Counter(self.coords.types).items())
        return f"$\\displaystyle H/h = {sym.latex(self.H)}$ GHz  (modes {counts})"


def quantize(circuit, loops, Z_full=None, types=None, coords=None):
    lag = lagrangian(circuit)
    if coords is None:
        if Z_full is None:
            Z_full, types = mode_basis(lag)
        check_basis(lag, Z_full, types)
        coords = decouple(lag, Z_full, types)
    B = irrotational_B(lag, coords, loops)
    ham = hamiltonian(lag, coords, loops, B, circuit.t)
    return Result(circuit=circuit, loops=loops, lag=lag, coords=coords, B=B, ham=ham)
