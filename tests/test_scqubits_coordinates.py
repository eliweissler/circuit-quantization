import numpy as np
import pytest
import sympy as sp

import circuit_quantization as cq

scq = pytest.importorskip("scqubits")


def _same_span(left, right):
    left = sp.Matrix(left)
    right = sp.Matrix(right)
    return left.rank() == right.rank() == sp.Matrix.hstack(left, right).rank()


def test_transmon_molecule_coordinates_equivalent_to_scqubits():
    yaml = """branches:
- [JJ, 1, 2, 5, 1]
- [JJ, 3, 4, 5, 1]
- [C, 1, 3, 1]
- [C, 2, 4, 1]
"""
    symbolic = scq.SymbolicCircuit.from_yaml(
        yaml, from_file=False, initiate_sym_calc=False
    )
    sc_z_array, sc_modes = symbolic.variable_transformation_matrix()
    sc_z = sp.Matrix(np.asarray(sc_z_array, dtype=int))

    circuit = cq.Circuit(
        (
            cq.Branch("J1", 1, 2, capacitance=sp.Integer(1), josephson_energy=5),
            cq.Branch("J2", 3, 4, capacitance=sp.Integer(1), josephson_energy=5),
            cq.Branch("C13", 1, 3, capacitance=sp.Integer(1)),
            cq.Branch("C24", 2, 4, capacitance=sp.Integer(1)),
        ),
        ground=0,
    )
    ours = cq.choose_coordinates(cq.assemble(circuit))

    assert len(ours.mode_indices[cq.ModeType.COMPACT]) == len(sc_modes["periodic"]) == 2
    assert len(ours.full_mode_indices[cq.ModeType.FREE]) == len(sc_modes["free"]) == 1
    assert len(ours.full_mode_indices[cq.ModeType.CYCLIC]) == len(sc_modes["sigma"]) == 1

    our_nd_indices = (
        ours.full_mode_indices[cq.ModeType.FREE]
        + ours.full_mode_indices[cq.ModeType.FROZEN]
        + ours.full_mode_indices[cq.ModeType.CYCLIC]
    )
    sc_nd_indices = tuple(index - 1 for index in sc_modes["free"] + sc_modes["sigma"])
    our_nd = ours.full_transformation[:, list(our_nd_indices)]
    sc_nd = sc_z[:, list(sc_nd_indices)]
    assert _same_span(our_nd, sc_nd)

    our_compact = ours.full_transformation[
        :, list(ours.full_mode_indices[cq.ModeType.COMPACT])
    ]
    sc_compact = sc_z[:, [index - 1 for index in sc_modes["periodic"]]]
    # Compact columns may differ by discarded free/global directions.
    assert _same_span(
        sp.Matrix.hstack(our_compact, our_nd),
        sp.Matrix.hstack(sc_compact, sc_nd),
    )

