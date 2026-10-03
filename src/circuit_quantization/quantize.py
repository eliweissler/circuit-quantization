"""End-to-end pipeline: Sec I -> II -> III -> V -> IV."""

from collections import Counter
from dataclasses import dataclass

import sympy as sym

from .circuit import Circuit
from .flux import irrotational_B
from .hamiltonian import Hamiltonian, hamiltonian
from .lagrangian import Lagrangian, lagrangian
from .loops import Loops
from .modes import Coordinates, decouple, mode_basis, residual_freedom, transform


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
        """Symbolic S_1 allowed on these coordinates, in coords.types order (eq:S1_structure).

        :param bool keep_harmonic: Preserve harmonic columns when True; allow
            extended/harmonic mixing when False.
        :returns: Symbolic residual transformation matrix in the current coordinate order.
        """
        return residual_freedom(self.coords, keep_harmonic)

    def transform(self, S1, *, simplify=False):
        """The same circuit and loops quantized in coordinates Z S_1 (Sec III.C).

        :param S1: Invertible residual transformation with new node matrix Z * S1.
            Must obey allowed mode mixing and compact-period constraints.
        :param bool simplify: Simplify the rebuilt Hamiltonian.
        :returns: New Result with the same B; the original is unchanged.
        :raises ValueError: Disallowed mixing or an invalid transformation.
        """
        return _result(self.circuit, self.loops, self.lag, transform(self.lag, self.coords, S1), simplify, B=self.B)

    def _repr_latex_(self):
        counts = ", ".join(f"{t}: {k}" for t, k in Counter(self.coords.types).items())
        return f"$\\displaystyle H_{{\\mathrm{{GHz}}}} = {sym.latex(self.H)}$  (modes {counts})"


def quantize(circuit, loops, Z_full=None, types=None, *, simplify=False, B=None):
    """Quantize with an automatic or explicit basis; optionally simplify the final H.

    :param circuit: Circuit with SI element values and junction energies in GHz.
    :param loops: Complete Loops basis belonging to this circuit.
    :param Z_full: Square node-to-coordinate matrix, including discarded modes;
        rows follow circuit.internal_nodes. Omit together with types for an automatic basis.
    :param types: Column mode labels c, e, h, fr, fz, cy. Required with Z_full.
    :param bool simplify: Factor coefficients and simplify trigonometric terms in H.
    :param B: Optional branch-by-loop flux assignment, in circuit.branch_names
        and loops.names order. Must satisfy loops.G * B = I. None selects the
        irrotational assignment; a custom assignment can introduce an EMF drive.
    :returns: Result containing lag, coords, B, ham, and H in GHz.
    :raises ValueError: Mismatched circuit/loops or an invalid coordinate basis.
    """
    if loops.circuit != circuit:
        raise ValueError("loops must belong to the circuit being quantized")
    lag = lagrangian(circuit)
    if Z_full is None and types is None:
        coords = decouple(lag, *mode_basis(lag), check=False)  # valid by construction
    elif Z_full is None or types is None:
        raise ValueError("Z_full and types must be supplied together")
    else:
        coords = decouple(lag, Z_full, types)
    return _result(circuit, loops, lag, coords, simplify, B=B)


def _result(circuit, loops, lag, coords, simplify, *, B=None):
    """Flux assignment and Hamiltonian for given coordinates (Sec V, Sec IV)."""
    if B is None:
        B = irrotational_B(lag, coords, loops)
    else:
        if not (loops.G * sym.ImmutableMatrix(B) - sym.eye(loops.G.rows)).applyfunc(sym.cancel).is_zero_matrix:
            raise ValueError("B must satisfy G * B = I")  # exact only after cancel, e.g. irrotational B
    ham = hamiltonian(lag, coords, loops, B, circuit.t, simplify=simplify)
    return Result(circuit=circuit, loops=loops, lag=lag, coords=coords, B=B, ham=ham)
