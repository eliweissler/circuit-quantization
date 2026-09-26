import pytest
import sympy as sp

import circuit_quantization as cq


def test_transmon_hamiltonian_in_ghz():
    e_c, e_j = sp.symbols("E_C E_J", positive=True)
    circuit = cq.Circuit(
        (
            cq.Branch(
                "J",
                0,
                1,
                capacitance=cq.ghz_to_farad(e_c),
                josephson_energy=e_j,
            ),
        ),
        ground=0,
    )
    result = cq.quantize(circuit)
    n = result.hamiltonian_data.cooper_pair_numbers[0]
    phase = result.hamiltonian_data.phases[0]
    assert sp.simplify(result.hamiltonian - (4 * e_c * n**2 - e_j * sp.cos(phase))) == 0


def test_fluxonium_irrotational_assignment_and_hamiltonian():
    e_c, e_l, e_j, flux = sp.symbols("E_C E_L E_J Phi", positive=True)
    circuit = cq.Circuit(
        (
            cq.Branch(
                "J",
                0,
                1,
                capacitance=cq.ghz_to_farad(e_c),
                josephson_energy=e_j,
            ),
            cq.Branch("L", 0, 1, inductance=cq.ghz_to_henry(e_l)),
        ),
        ground=0,
        loop_fluxes=(cq.LoopFlux("loop", {"J": 1, "L": -1}, flux),),
    )
    result = cq.quantize(circuit)
    g = result.matrices.loop_matrix
    b = result.flux_assignment.assignment
    z = result.coordinates.transformation
    assert sp.simplify(g * b) == sp.eye(1)
    assert sp.simplify(
        z.T
        * result.matrices.omega_internal
        * result.matrices.capacitance_branch
        * b
    ) == sp.zeros(1, 1)
    assert sp.simplify(result.hamiltonian_data.charging_energy[0] - e_c) == 0
    assert sp.simplify(result.hamiltonian_data.inductive_energy[0] - e_l) == 0


def test_colocated_inductor_capacitance_gives_inverse_capacitance_flux_divider():
    c_j, c_l, inductance, flux = sp.symbols("C_J C_L L Phi", positive=True)
    circuit = cq.Circuit(
        (
            cq.Branch("J", 0, 1, capacitance=c_j, josephson_energy=5),
            cq.Branch("L", 0, 1, capacitance=c_l, inductance=inductance),
        ),
        ground=0,
        loop_fluxes=(cq.LoopFlux("loop", {"J": 1, "L": -1}, flux),),
    )
    matrices = cq.assemble(circuit)
    coordinates = cq.choose_coordinates(matrices)
    assignment = cq.assign_flux(matrices, coordinates)
    expected = sp.Matrix([[c_l / (c_j + c_l)], [-c_j / (c_j + c_l)]])
    assert sp.simplify(assignment.assignment - expected) == sp.zeros(2, 1)


def test_driven_loop_has_source_residual_but_zero_projected_emf():
    c_j, c_g, inductance = sp.symbols("C_J C_g L", positive=True)
    t = sp.Symbol("t", real=True)
    flux = sp.Function("Phi")(t)
    voltage = sp.Function("V")(t)
    circuit = cq.Circuit(
        (
            cq.Branch("J", 0, 1, capacitance=c_j, josephson_energy=5),
            cq.Branch("L", 0, 1, inductance=inductance),
            cq.Branch("Cg", 1, 2, capacitance=c_g),
            cq.Branch("Vsrc", 0, 2, voltage=voltage),
        ),
        ground=0,
        loop_fluxes=(cq.LoopFlux("device", {"J": 1, "L": -1}, flux),),
        time=t,
    )
    with pytest.warns(cq.IncompleteLoopBasisWarning):
        result = cq.quantize(circuit)
    projected = sp.simplify(
        result.coordinates.transformation.T
        * result.matrices.omega_internal
        * result.matrices.capacitance_branch
        * result.flux_assignment.assignment
    )
    assert projected == sp.zeros(result.coordinates.dynamical_count, 2)
    assert len(result.flux_assignment.residual_basis) == 1
    # In the chosen irrotational representative only the classical gate voltage
    # survives in the retained offset charge.
    assert not result.hamiltonian_data.offset_charge.has(sp.diff(flux, t))


def test_voltage_drive_offset_charge_uses_note_sign_and_no_half_factor():
    e_j, c_gate, voltage = sp.symbols("E_J C_g V", positive=True)
    c_j_positive = sp.Symbol("C_J", positive=True)
    circuit = cq.Circuit(
        (
            cq.Branch("J", 0, 1, capacitance=c_j_positive, josephson_energy=e_j),
            cq.Branch("Cg", 1, 2, capacitance=c_gate),
            cq.Branch("V", 0, 2, voltage=voltage),
        ),
        ground=0,
    )
    result = cq.quantize(circuit)
    ng = result.hamiltonian_data.offset_cooper_pairs[0]
    assert sp.simplify(
        result.hamiltonian_data.offset_charge_node[0] + c_gate * voltage
    ) == 0
    n = result.hamiltonian_data.cooper_pair_numbers[0]
    charging = result.hamiltonian + e_j * sp.cos(result.hamiltonian_data.phases[0])
    expected_ec = cq.farad_to_ghz(c_j_positive + c_gate)
    assert sp.simplify(charging - 4 * expected_ec * (n - ng) ** 2) == 0
