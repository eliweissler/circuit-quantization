"""Symbolic quantization of lumped superconducting circuits.

Build a :class:`Circuit` from :class:`Branch` elements, choose :class:`Loops`,
and call :func:`quantize` for the symbolic Hamiltonian ``H/h`` in GHz.

    circuit.py      Branch, Circuit, incidence
    loops.py        loop bases and external fluxes
    lagrangian.py   capacitance, inductance, and incidence matrices
    modes.py        mode classification, decoupling, residual freedom
    numerical.py    numerical coordinate searches
    shear.py        partial shears of harmonic modes onto junction modes
    flux.py         irrotational flux assignment
    hamiltonian.py  frequency-unit Hamiltonian
    quantize.py     end-to-end quantization and Result
    io.py           JSON interchange (kronfree-spec)
    draw.py         Schemdraw circuit diagrams (optional: ``[draw]``)
    units.py        constants and unit converters

Extensions, imported explicitly:

    circuit_quantization.optimize        exact symbolic coordinate search (optional: ``[optimize]``)
    circuit_quantization.product_state   experimental product-state solver

Matrix names follow the matrix formulation:

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

from .circuit import Branch, Circuit, incidence
from .draw import draw_circuit_diagram
from .flux import irrotational_B, B_from_shares
from .hamiltonian import hamiltonian, simplify_hamiltonian
from .io import HamiltonianSpec, load_spec, save_spec, to_spec, load_record, to_record
from .lagrangian import lagrangian
from .loops import Loops, branch_dict_to_G, check_loops, cycle_bases, default_tree, spanning_trees, tree_to_G
from .modes import (
    check_coordinates,
    cluster_cost,
    compact_charge_decoupling,
    coordinates,
    decouple,
    hierarchical_decoupling,
    junction_clusters,
    junction_inversions,
    mode_basis,
    mode_dimensions,
    mode_types,
    normal_mode_decoupling,
    residual_freedom,
    transform,
)
from .numerical import cost_localization, cost_normal_modes, cost_targeted, numerical_S1, parameterization
from .product_state import NumericHamiltonian, normal_mode_transformation, numeric_hamiltonian
from .shear import shear_S1
from .quantize import Result, numerify, quantize
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
