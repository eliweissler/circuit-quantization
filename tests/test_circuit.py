import pytest
import sympy as sym

from circuit_quantization import Branch, Circuit, incidence
from circuits import coupled, harmonium

C, EJ, L = sym.symbols("C E_J L", positive=True)


@pytest.mark.parametrize("kwargs, message", [
    (dict(EJ=EJ), "needs C > 0"),
    (dict(C=C, L=L, EJ=EJ), "at most one"),
    (dict(C=C, V=1), "must have C = 0"),
    (dict(C=sym.Symbol("x")), "provably positive"),
])
def test_branch_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Branch("b", 1, 0, **kwargs)


def test_source_rules():
    with pytest.raises(ValueError, match="connect to ground"):
        Circuit([Branch("J", 1, 0, C=C, EJ=EJ), Branch("V", 1, 2, V=1)], ground=0)
    with pytest.raises(ValueError, match="Only capacitors"):
        Circuit([Branch("J", 1, 2, C=C, EJ=EJ), Branch("V", 0, 2, V=1)], ground=0)


def test_ordering_and_incidence_match_notes():
    circuit = harmonium()
    assert circuit.nodes == (1, 2, 3, 4, 5)
    assert circuit.branch_names == ("C", "J1", "J2", "J3", "L1", "L2", "L3")
    assert incidence(circuit) == sym.Matrix([  # eq:harm_Omega
        [1, 1, 1, 0, 0, 0, 0],
        [0, -1, 0, 0, 1, 0, 0],
        [0, 0, -1, 0, 0, 1, 0],
        [0, 0, 0, 1, -1, -1, 1],
        [-1, 0, 0, -1, 0, 0, -1],
    ])


def test_branches_sorted_by_type_keep_user_order():
    circuit = coupled()
    assert circuit.branch_names[:8] == ("Ct", "Cc", "Cf", "CV1", "CV2", "CV3", "CV4", "Cg")
    assert circuit.nodes == (1, 2, 3, 4, 5, 6, 7)
