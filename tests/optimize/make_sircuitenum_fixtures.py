"""Regenerate sircuitenum_fixtures.json from sircuitenum itself (not run by pytest).

The ported quantize tests feed sircuitenum's starting basis Z0 and mode labels
(var_trans_basis) and its node-space matrices (gen_cap_mat, gen_ind_mat, gen_w)
to the vendored functions. Run in an environment with sircuitenum installed:

    python tests/optimize/make_sircuitenum_fixtures.py
"""

import json
from pathlib import Path

import sympy as sym
from sircuitenum import quantize, utils

# (key, circuit, edges, renumber, numbered): every circuit the ported tests build.
CIRCUITS = [
    ("transmon", [("J", "C")], [(0, 1)], False, False),
    ("zero_pi", [("J",), ("J",), ("L",), ("L",), ("C",), ("C",)], [(0, 1), (2, 3), (0, 3), (1, 2), (0, 2), (1, 3)], False, False),
    ("lc_chain_numbered", [("L", "C"), ("L", "C"), ("L",), ("J",)], [(1, 2), (3, 4), (1, 3), (2, 4)], False, True),
    ("lc_chain_labelled", [("L_1", "C_1"), ("L_2", "C_2"), ("L_3",), ("J_1",)], [(1, 2), (3, 4), (1, 3), (2, 4)], True, False),
    ("j_jl_cjl", [("J",), ("J", "L"), ("C", "J", "L")], [(0, 1), (0, 2), (1, 2)], False, False),
]


def main():
    out = {}
    for key, circuit, edges, renumber, numbered in CIRCUITS:
        if numbered:
            circuit = utils.add_elem_number(circuit)
        if renumber:
            edges = utils.renumber_nodes(edges)
        Z0, var_types = quantize.var_trans_basis(circuit, edges)
        wJ, EJ = quantize.gen_w(circuit, edges, w_elem="J", return_params=True)
        out[key] = {
            "circuit": [list(e) for e in circuit],
            "edges": [list(e) for e in edges],
            "Z0": sym.srepr(sym.Matrix(Z0)),
            "var_types": var_types,
            "cMat": sym.srepr(quantize.gen_cap_mat(circuit, edges)),
            "lMat": sym.srepr(quantize.gen_ind_mat(circuit, edges)),
            "wJ": sym.srepr(sym.Matrix(wJ)),
            "EJ": [sym.srepr(x) for x in EJ],
        }
    path = Path(__file__).with_name("sircuitenum_fixtures.json")
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(out)} circuits)")


if __name__ == "__main__":
    main()
