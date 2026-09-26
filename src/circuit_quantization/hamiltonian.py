"""Frequency-unit symbolic Hamiltonian construction."""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .coordinates import CoordinateSystem
from .flux import FluxAssignment
from .matrices import CircuitMatrices
from .units import (
    ELEMENTARY_CHARGE,
    GHZ,
    PLANCK_CONSTANT,
    REDUCED_FLUX_QUANTUM,
)


@dataclass(frozen=True)
class HamiltonianData:
    hamiltonian: sp.Expr
    phases: sp.ImmutableMatrix
    cooper_pair_numbers: sp.ImmutableMatrix
    external_phases: sp.ImmutableMatrix
    offset_charge_node: sp.ImmutableMatrix
    offset_charge: sp.ImmutableMatrix
    offset_cooper_pairs: sp.ImmutableMatrix
    charging_energy: sp.ImmutableMatrix
    inductive_energy: sp.ImmutableMatrix
    branch_inductive_energy: sp.ImmutableMatrix


def build_hamiltonian(
    matrices: CircuitMatrices,
    coordinates: CoordinateSystem,
    flux_assignment: FluxAssignment,
) -> HamiltonianData:
    """Return H/h in GHz in dimensionless phase/charge coordinates."""
    mode_count = coordinates.dynamical_count
    phases = sp.Matrix(
        [sp.Symbol(f"varphi_{index + 1}", real=True) for index in range(mode_count)]
    )
    numbers = sp.Matrix(
        [sp.Symbol(f"n_{index + 1}", real=True) for index in range(mode_count)]
    )

    external_fluxes = sp.Matrix(matrices.external_fluxes)
    external_phases = external_fluxes / REDUCED_FLUX_QUANTUM
    assignment = sp.Matrix(flux_assignment.assignment)
    z = sp.Matrix(coordinates.transformation)

    voltage_charge = -sp.Matrix(matrices.capacitance_internal_source) * sp.Matrix(
        matrices.source_voltages
    )
    emf_charge = (
        sp.Matrix(matrices.omega_internal)
        * sp.Matrix(matrices.capacitance_branch)
        * assignment
        * external_fluxes.applyfunc(
            lambda expression: sp.diff(expression, matrices.circuit.time)
        )
    )
    offset_charge_node = sp.simplify(voltage_charge + emf_charge)
    offset_charge = sp.simplify(z.T * offset_charge_node)
    offset_cooper_pairs = offset_charge / (2 * ELEMENTARY_CHARGE)

    frequency_factor_c = ELEMENTARY_CHARGE**2 / (2 * PLANCK_CONSTANT * GHZ)
    frequency_factor_l = REDUCED_FLUX_QUANTUM**2 / (PLANCK_CONSTANT * GHZ)
    charging_energy = sp.simplify(
        frequency_factor_c * sp.Matrix(coordinates.inverse_capacitance)
    )
    inductive_energy = sp.simplify(
        frequency_factor_l * sp.Matrix(coordinates.susceptance)
    )
    branch_inductive_energy = sp.simplify(
        frequency_factor_l * sp.Matrix(matrices.inductance_branch_inverse)
    )

    shifted_numbers = numbers - offset_cooper_pairs
    charging = 4 * (shifted_numbers.T * charging_energy * shifted_numbers)[0]
    inductive = sp.Rational(1, 2) * (
        phases.T * inductive_energy * phases
    )[0]

    if matrices.inductor_indices and matrices.cycle_rank:
        b_inductor = assignment[list(matrices.inductor_indices), :]
        inductive += (
            phases.T
            * sp.Matrix(coordinates.inductor_incidence_transformed).T
            * branch_inductive_energy
            * b_inductor
            * external_phases
        )[0]

    josephson = sp.S.Zero
    transformed_junction = sp.Matrix(
        coordinates.junction_incidence_transformed
    )
    for local_index, branch_index in enumerate(matrices.junction_indices):
        branch = matrices.branches[branch_index]
        argument = (transformed_junction[local_index, :] * phases)[0]
        if matrices.cycle_rank:
            argument += (assignment[branch_index, :] * external_phases)[0]
        josephson -= branch.josephson_energy * sp.cos(sp.simplify(argument))

    # Preserve the note's quadratic structure. A global simplify/expand here can
    # turn exact SI constants and symbolic capacitances into enormous rational
    # expressions and is prohibitively slow for otherwise small circuits.
    hamiltonian = charging + inductive + josephson
    freeze = sp.ImmutableMatrix
    return HamiltonianData(
        hamiltonian=hamiltonian,
        phases=freeze(phases),
        cooper_pair_numbers=freeze(numbers),
        external_phases=freeze(external_phases),
        offset_charge_node=freeze(offset_charge_node),
        offset_charge=freeze(offset_charge),
        offset_cooper_pairs=freeze(offset_cooper_pairs),
        charging_energy=freeze(charging_energy),
        inductive_energy=freeze(inductive_energy),
        branch_inductive_energy=freeze(branch_inductive_energy),
    )
