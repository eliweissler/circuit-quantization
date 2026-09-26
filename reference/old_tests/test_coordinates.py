import sympy as sp

import circuit_quantization as cq


def test_transmon_molecule_rank_deficient_nd_decoupling():
    c_j, c_13, c_24 = sp.symbols("C_J C_13 C_24", positive=True)
    circuit = cq.Circuit(
        (
            cq.Branch("J1", 1, 2, capacitance=c_j, josephson_energy=5),
            cq.Branch("J2", 3, 4, capacitance=c_j, josephson_energy=5),
            cq.Branch("C13", 1, 3, capacitance=c_13),
            cq.Branch("C24", 2, 4, capacitance=c_24),
        ),
        ground=0,
    )
    matrices = cq.assemble(circuit)
    coordinates = cq.choose_coordinates(matrices)
    assert len(coordinates.mode_indices[cq.ModeType.COMPACT]) == 2
    assert len(coordinates.full_mode_indices[cq.ModeType.FREE]) == 1
    assert len(coordinates.full_mode_indices[cq.ModeType.CYCLIC]) == 1
    nd_start = coordinates.dynamical_count
    nondynamical = coordinates.full_transformation[:, nd_start:]
    assert sp.simplify(
        coordinates.transformation.T
        * matrices.capacitance_internal
        * nondynamical
    ) == sp.zeros(2, 2)
    assert sp.simplify(
        matrices.omega_junction_internal.T * nondynamical
    ) == sp.zeros(2, 2)


def test_frozen_mode_is_removed_sequentially():
    c = sp.Symbol("C", positive=True)
    l = sp.Symbol("L", positive=True)
    circuit = cq.Circuit(
        (
            cq.Branch("C", 0, 1, capacitance=c),
            cq.Branch("L1", 0, 1, inductance=l),
            cq.Branch("L2", 1, 2, inductance=l),
        ),
        ground=0,
    )
    matrices = cq.assemble(circuit)
    coordinates = cq.choose_coordinates(matrices)
    assert len(coordinates.full_mode_indices[cq.ModeType.FROZEN]) == 1
    assert coordinates.dynamical_count == 1
    nd = coordinates.full_transformation[:, 1:]
    assert sp.simplify(
        coordinates.transformation.T * matrices.susceptance_internal * nd
    ) == sp.zeros(1, nd.cols)


def test_zero_pi_mode_classification():
    c, c_j, inductance, e_j = sp.symbols("C C_J L E_J", positive=True)
    circuit = cq.Circuit(
        (
            cq.Branch("J12", 1, 2, capacitance=c_j, josephson_energy=e_j),
            cq.Branch("J34", 3, 4, capacitance=c_j, josephson_energy=e_j),
            cq.Branch("L14", 1, 4, inductance=inductance),
            cq.Branch("L23", 2, 3, inductance=inductance),
            cq.Branch("C13", 1, 3, capacitance=c),
            cq.Branch("C24", 2, 4, capacitance=c),
        ),
        ground=0,
    )
    coordinates = cq.choose_coordinates(cq.assemble(circuit))
    assert len(coordinates.mode_indices[cq.ModeType.COMPACT]) == 1
    assert len(coordinates.mode_indices[cq.ModeType.EXTENDED]) == 1
    assert len(coordinates.mode_indices[cq.ModeType.HARMONIC]) == 1
    assert len(coordinates.full_mode_indices[cq.ModeType.CYCLIC]) == 1
