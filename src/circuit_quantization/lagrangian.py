"""Element and incidence matrices of the node Lagrangian (Sec II)."""

from dataclasses import dataclass

import sympy as sym

from .circuit import incidence


@dataclass(frozen=True)
class Lagrangian:
    Omega: sym.ImmutableMatrix  # rows int | gnd | src (no gnd if floating), columns C | J | L | V
    Omega_int: sym.ImmutableMatrix
    C_b: sym.ImmutableMatrix  # eq:Cb_diag
    C_int: sym.ImmutableMatrix  # eq:C_full_partition
    C_int_src: sym.ImmutableMatrix  # eq:C_full_partition, note the sign
    Lb_inv: sym.ImmutableMatrix  # L_b^{-1}, inductors only
    Lstar_int: sym.ImmutableMatrix  # eq:L_full_partition
    omega_J_int: sym.ImmutableMatrix
    omega_L_int: sym.ImmutableMatrix
    V: sym.ImmutableMatrix  # source-node voltages
    J: tuple  # branch indices of junctions
    L: tuple  # branch indices of inductors
    EJ: tuple


def lagrangian(circuit):
    """:param circuit: Circuit with symbolic or numeric element values in SI units.
    :returns: Lagrangian containing incidence, capacitance, inductive, and source matrices.
    """
    branches = circuit.branches
    n = len(circuit.internal_nodes)
    Omega = incidence(circuit)
    Omega_int = Omega[:n, :]
    J = tuple(k for k, b in enumerate(branches) if b.type == "J")
    L = tuple(k for k, b in enumerate(branches) if b.type == "L")

    C_b = sym.ImmutableMatrix(sym.diag(*(b.C for b in branches)))  # eq:Cb_diag
    C_full = Omega * C_b * Omega.T  # eq:C_full_partition
    C_int = C_full[:n, :n]
    # Source rows come last in Omega, after the internal and ground rows.
    C_int_src = -C_full[:n, C_full.cols - len(circuit.source_nodes) :]

    omega_J_int = Omega_int[:, list(J)]
    omega_L_int = Omega_int[:, list(L)]
    Lb_inv = sym.ImmutableMatrix(sym.diag(*(1 / branches[k].L for k in L)) if L else sym.zeros(0, 0))
    Lstar_int = omega_L_int * Lb_inv * omega_L_int.T  # eq:L_full_partition

    # Each source branch joins ground to its own source node; V follows source-node order.
    voltage = {}
    for b in branches:
        if b.type == "V":
            source_node = b.tail if b.head == circuit.ground else b.head
            voltage[source_node] = b.V
    V = sym.ImmutableMatrix(len(circuit.source_nodes), 1, [voltage[node] for node in circuit.source_nodes])

    return Lagrangian(
        Omega=Omega, Omega_int=Omega_int, C_b=C_b, C_int=C_int, C_int_src=C_int_src,
        Lb_inv=Lb_inv, Lstar_int=Lstar_int, omega_J_int=omega_J_int, omega_L_int=omega_L_int,
        V=V, J=J, L=L, EJ=tuple(branches[k].EJ for k in J),
    )
