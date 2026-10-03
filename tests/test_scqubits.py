import numpy as np
import pytest
import sympy as sym

from circuit_quantization import (
    Branch, Circuit, Loops, decouple, default_tree, irrotational_B,
    lagrangian, mode_basis, to_scqubits,
)
from circuits import random_circuit

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
    ])
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


@pytest.mark.parametrize("seed", range(30))
def test_random_circuits_match_scqubits_modes_and_flux_allocation(seed):
    # scqubits imposes the irrotational condition on every node (D = 0); we impose it
    # on the retained modes (Z^T D = 0).  Sec V.D.2: the two can differ only by a
    # node-flux shift along discarded directions, which no junction sees, which
    # reaches inductors only along frozen directions, and which vanishes without
    # non-dynamical modes.
    circuit = random_circuit(seed)
    lag = lagrangian(circuit)
    Z, types = mode_basis(lag)
    theirs = to_scqubits(circuit, symbolic=True, use_dynamic_flux_grouping=True)
    counts = {k: len(v) for k, v in theirs.var_categories.items()}
    assert counts == {"periodic": types.count("c"), "extended": types.count("e") + types.count("h"),
                      "free": types.count("fr"), "frozen": types.count("fz"), "sigma": types.count("cy")}
    if not set(types) & {"c", "e", "h"}:  # both agree there is nothing to quantize (seed 7)
        with pytest.raises(ValueError, match="no dynamical modes"):
            decouple(lag, Z, types)
        return
    coords = decouple(lag, Z, types)
    loops = Loops.from_tree(circuit, default_tree(circuit))
    B = irrotational_B(lag, coords, loops)

    offsets = theirs._time_dependent_flux_distribution()
    B_theirs = sym.Matrix([[sym.sympify(x).coeff(f) for f in theirs.external_fluxes] for x in offsets])
    B_theirs = B_theirs * (loops.G * B_theirs).inv()  # the same allocation in our loop basis

    compared = {"J"} | ({"L"} if "fz" not in types else set()) | ({"C"} if len(coords.types) == len(types) else set())
    for k, branch in enumerate(circuit.branches):
        if branch.type in compared:
            for j in range(B.cols):
                assert abs(float(B[k, j] - B_theirs[k, j])) < 1e-8, (branch.name, j)
