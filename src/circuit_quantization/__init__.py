"""Symbolic quantization of superconducting circuits.

Companion code to the circuit quantization notes; each module follows one section:

    units.py        Sec VI   constants and converters
    circuit.py      Sec I    Branch, Circuit, Loops, cycle_bases, fundamental_cycles
    lagrangian.py   Sec II   element and incidence matrices
    modes.py        Sec III  mode classification, decoupling, residual freedom
    flux.py         Sec V    irrotational flux assignment
    hamiltonian.py  Sec IV   frequency-unit Hamiltonian

Symbols in the notes and their names here:

    Omega, Omega_int    incidence matrix and its internal rows
    omega_J_int         junction columns of Omega_int
    omega_L_int         inductor columns of Omega_int
    C_b, Lb_inv         branch capacitances, inverse inductances
    C_int, C_int_src    internal and drive capacitance blocks
    Lstar_int           internal susceptance L*_int
    G, Phi_ext          loop matrix and external fluxes [Wb]
    Z, types            retained coordinate transformation and mode types
    R                   retained-coordinate extraction, R Z = I and R Z_2 = 0
    C_tilde, Lstar_tilde, omega_J_tilde, omega_L_tilde   transformed objects
    B                   irrotational flux-assignment matrix
    E_C, E_L, E_Lb      frequency-unit matrices [GHz]
    n_g                 offset charges [Cooper pairs]
    H                   Hamiltonian H/h [GHz]
"""

from .circuit import Branch, Circuit, Loops, cycle_bases, fundamental_cycles, incidence, spanning_trees
from .draw import draw_circuit_diagram
from .flux import irrotational_B
from .hamiltonian import hamiltonian, simplify_hamiltonian
from .lagrangian import lagrangian
from .modes import (
    check_basis,
    check_coordinates,
    coordinates,
    decouple,
    mode_basis,
    mode_dimensions,
    mode_types,
    residual_freedom,
    transform,
)
from .quantize import Result, quantize
from .qpackage_interface import scqubits_yaml, to_scqubits
from .units import (
    farad_to_ghz,
    ghz_to_farad,
    ghz_to_henry,
    henry_to_ghz,
    phase_to_weber,
    subs_constants,
    weber_to_phase,
)
