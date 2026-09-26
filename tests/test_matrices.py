import pytest
import sympy as sp

import circuit_quantization as cq


def test_parallel_branches_generate_cycle_basis():
    circuit = cq.Circuit(
        (
            cq.Branch("J", 0, 1, capacitance=1e-15, josephson_energy=5),
            cq.Branch("L", 0, 1, inductance=1e-7),
        ),
        ground=0,
    )
    matrices = cq.assemble(circuit)
    assert matrices.loop_matrix.shape == (1, 2)
    assert matrices.loop_matrix * matrices.omega.T == sp.zeros(1, 2)


def test_partial_physical_basis_warns_and_completes():
    loop = cq.LoopFlux("physical", {"J1": 1, "L": -1}, value=0)
    circuit = cq.Circuit(
        (
            cq.Branch("J1", 0, 1, capacitance=1e-15, josephson_energy=5),
            cq.Branch("J2", 0, 1, capacitance=2e-15, josephson_energy=6),
            cq.Branch("L", 0, 1, inductance=1e-7),
        ),
        ground=0,
        loop_fluxes=(loop,),
    )
    with pytest.warns(cq.IncompleteLoopBasisWarning):
        matrices = cq.assemble(circuit)
    assert matrices.loop_matrix.shape == (2, 3)
    assert matrices.loop_names[0] == "physical"
    assert matrices.external_fluxes == sp.zeros(2, 1)
    assert matrices.loop_matrix * matrices.omega.T == sp.zeros(2, 2)


def test_source_nodes_and_capacitance_partition():
    c_j, c_g, voltage = sp.symbols("C_J C_g V", positive=True)
    circuit = cq.Circuit(
        (
            cq.Branch("J", 0, 1, capacitance=c_j, josephson_energy=5),
            cq.Branch("Cg", 1, 2, capacitance=c_g),
            cq.Branch("V", 0, 2, voltage=voltage),
        ),
        ground=0,
    )
    matrices = cq.assemble(circuit)
    assert matrices.internal_nodes == (1,)
    assert matrices.source_nodes == (2,)
    assert matrices.capacitance_internal == sp.Matrix([[c_j + c_g]])
    assert matrices.capacitance_internal_source == sp.Matrix([[c_g]])
    assert matrices.source_voltages == sp.Matrix([voltage])


def test_invalid_junction_without_intrinsic_capacitance():
    with pytest.raises(ValueError, match="positive capacitance"):
        cq.Branch("J", 0, 1, josephson_energy=5)

