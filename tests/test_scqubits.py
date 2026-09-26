import numpy as np
import pytest
import sympy as sym

from circuit_quantization import Branch, Circuit, lagrangian, mode_basis

scq = pytest.importorskip("scqubits")


def same_span(A, B):
    return sym.Matrix(A).rank() == sym.Matrix(B).rank() == sym.Matrix.hstack(A, B).rank()


def test_transmon_molecule_matches_scqubits():
    yaml = """branches:
- [JJ, 1, 2, 5, 1]
- [JJ, 3, 4, 5, 1]
- [C, 1, 3, 1]
- [C, 2, 4, 1]
"""
    symbolic = scq.SymbolicCircuit.from_yaml(yaml, from_file=False, initiate_sym_calc=False)
    sc_Z, sc_modes = symbolic.variable_transformation_matrix()
    sc_Z = sym.Matrix(np.asarray(sc_Z, dtype=int))

    circuit = Circuit([
        Branch("J1", 1, 2, C=1, EJ=5), Branch("J2", 3, 4, C=1, EJ=5),
        Branch("C13", 1, 3, C=1), Branch("C24", 2, 4, C=1),
    ], ground=0)
    Z, types = mode_basis(lagrangian(circuit))
    assert types.count("c") == len(sc_modes["periodic"]) == 2
    assert types.count("fr") == len(sc_modes["free"]) == 1
    assert types.count("cy") == len(sc_modes["sigma"]) == 1

    ours_nd = Z[:, [j for j, t in enumerate(types) if t in ("fr", "cy")]]
    theirs_nd = sc_Z[:, [i - 1 for i in sc_modes["free"] + sc_modes["sigma"]]]
    assert same_span(ours_nd, theirs_nd)
    ours_c = Z[:, [j for j, t in enumerate(types) if t == "c"]]
    theirs_c = sc_Z[:, [i - 1 for i in sc_modes["periodic"]]]
    # Compact columns may differ by discarded free and global directions.
    assert same_span(sym.Matrix.hstack(ours_c, ours_nd), sym.Matrix.hstack(theirs_c, theirs_nd))
