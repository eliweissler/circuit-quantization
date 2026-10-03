"""sircuitenum circuits as circuit_quantization Circuits, and the sircuitenum fixtures.

sircuitenum describes a circuit by element labels per edge, e.g. ``[("J", "C"), ("L",)]``
with ``edges = [(0, 1), (1, 2)]``. Each element runs from the smaller node to the larger
(gen_w), and its value is the symbol named by its label, so equal labels share a value:
a junction ``J_1`` has E_J = J_1 and capacitance C_J_1. With no ground node the circuit
floats.
"""

import json
from pathlib import Path

import sympy as sym
from sympy import *  # noqa: F401,F403  (srepr round trip)

import circuit_quantization as cq

FIXTURES = json.loads(Path(__file__).with_name("sircuitenum_fixtures.json").read_text())


def positive(name):
    return sym.Symbol(name, positive=True)


def to_circuit(circuit, edges, ground=None):
    """The sircuitenum circuit as a Circuit; parallel elements on one edge become parallel branches."""
    branches = []
    for k, (labels, (a, b)) in enumerate(zip(circuit, edges)):
        tail, head = min(a, b), max(a, b)
        for label in labels:
            name = f"{label}_e{k}"
            if "J" in label:
                C_J = positive("C_J" + label.replace("J", ""))
                branches.append(cq.Branch(name, tail, head, C=C_J, EJ=positive(label)))
            elif "L" in label:
                branches.append(cq.Branch(name, tail, head, L=positive(label)))
            else:
                branches.append(cq.Branch(name, tail, head, C=positive(label)))
    return cq.Circuit(branches, ground=ground)


def fixture(key):
    """sircuitenum's var_trans_basis Z0 and var_types, and its node-space matrices, for one circuit."""
    data = dict(FIXTURES[key])
    for name in ("Z0", "cMat", "lMat", "wJ"):
        data[name] = eval(data[name])
    data["EJ"] = tuple(eval(x) for x in data["EJ"])
    return data
