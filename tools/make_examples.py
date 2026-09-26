"""Write the Appendix B example notebooks (run once; the notebooks are committed)."""

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

SETUP = """import sympy as sym
import circuit_quantization as cq
from circuit_quantization import Branch, Circuit, Loops, quantize

sym.init_printing()
F, H, W = cq.ghz_to_farad, cq.ghz_to_henry, cq.phase_to_weber"""

EXAMPLES = {
    "harmonium": ("The harmonium (notes Appendix B.1, fig:harmonium)", """\
EC, EC1, EC2, EC3, EL1, EL2, EL3, EJ1, EJ2, EJ3 = sym.symbols(
    "E_C E_CJ1 E_CJ2 E_CJ3 E_L1 E_L2 E_L3 E_J1 E_J2 E_J3", positive=True)
vk, vf = sym.symbols("varphi_k varphi_f", real=True)

circuit = Circuit([
    Branch("C", 1, 5, C=F(EC)),
    Branch("J1", 1, 2, C=F(EC1), EJ=EJ1),
    Branch("J2", 1, 3, C=F(EC2), EJ=EJ2),
    Branch("J3", 4, 5, C=F(EC3), EJ=EJ3),
    Branch("L1", 2, 4, L=H(EL1)),
    Branch("L2", 3, 4, L=H(EL2)),
    Branch("L3", 4, 5, L=H(EL3)),
], ground=5)
cycles = {  # eq:harm_G
    "k": {"J1": 1, "J2": -1, "L1": 1, "L2": -1},
    "f": {"J3": 1, "L3": -1},
    "cb": {"C": 1, "J1": -1, "J3": -1, "L1": -1},
}
loops = Loops(circuit, cycles, {"k": W(vk), "f": W(vf), "cb": 0})
Z = sym.Matrix([[1, 0, 0, 0], [0, 1, 1, 1], [0, 1, -1, 1], [0, 1, 0, 0]])  # eq:harm_Z
res = quantize(circuit, loops, Z, ("c", "e", "e", "e"))
res""", ["res.E_C", "res.E_L", "res.B"]),
    "zero_pi": ("The symmetric 0-pi qubit (notes Appendix B.2)", """\
CJ, C, L, EJ = sym.symbols("C_J C L E_J", positive=True)
circuit = Circuit([
    Branch("J1", 1, 2, C=CJ, EJ=EJ),
    Branch("J2", 3, 4, C=CJ, EJ=EJ),
    Branch("L1", 2, 3, L=L),
    Branch("L2", 4, 1, L=L),
    Branch("C1", 1, 3, C=C),
    Branch("C2", 2, 4, C=C),
], ground=5)  # floating: ground carries no branch
cycles = {  # eq:zp_G
    "ring": {"J1": 1, "J2": 1, "L1": 1, "L2": 1},
    "t1": {"J1": 1, "L1": 1, "C1": -1},
    "t2": {"J2": 1, "L1": 1, "C2": -1},
}
loops = Loops(circuit, cycles, {"t1": 0, "t2": 0})
Z = sym.Matrix([[-1, -1, 1, 1], [1, 1, 1, 1], [-1, 1, -1, 1], [1, -1, -1, 1]]) / 2  # eq:zp_Z
res = quantize(circuit, loops, Z, ("e", "c", "h", "cy"))
res""", ["res.C_tilde", "res.Lstar_tilde", "res.B"]),
    "kite": ("The cos(2 phi) transmon (notes Appendix B.3)", """\
ECS, EC1, EC2, EL1, EL2, EJ1, EJ2 = sym.symbols("E_CS E_CJ1 E_CJ2 E_L1 E_L2 E_J1 E_J2", positive=True)
vext = sym.Symbol("varphi_ext", real=True)
circuit = Circuit([
    Branch("CS", 1, 4, C=F(ECS)),
    Branch("J1", 1, 2, C=F(EC1), EJ=EJ1),
    Branch("J2", 1, 3, C=F(EC2), EJ=EJ2),
    Branch("L1", 2, 4, L=H(EL1)),
    Branch("L2", 3, 4, L=H(EL2)),
], ground=4)
cycles = {"kite": {"J1": 1, "J2": -1, "L1": 1, "L2": -1}, "cb": {"CS": 1, "J1": -1, "L1": -1}}  # eq:kite_G
loops = Loops(circuit, cycles, {"kite": W(vext), "cb": 0})
Z = sym.Matrix([[1, 0, 0], [0, 1, 1], [0, -1, 1]])  # eq:kite_Z
res = quantize(circuit, loops, Z, ("c", "e", "e"))
res""", ["res.E_C", "cq.residual_freedom(res.coords)"]),
    "rhombus": ("The asymmetric rhombus (notes Appendix B.4)", """\
CJ, CB, Cg, EJ, alpha = sym.symbols("C_J C_B C_g E_J alpha", positive=True)
circuit = Circuit([
    Branch("J1", 1, 2, C=CJ, EJ=EJ),
    Branch("J2", 2, 3, C=CJ, EJ=EJ),
    Branch("J3", 3, 4, C=CJ, EJ=EJ),
    Branch("J4", 4, 1, C=alpha * CJ, EJ=alpha * EJ),
    Branch("CB", 1, 3, C=CB),
    *(Branch(f"Cg{i}", i, 5, C=Cg) for i in range(1, 5)),
], ground=5)
cycles = cq.fundamental_cycles(circuit)  # or eq:rh_G
loops = Loops(circuit, cycles, {name: 0 for name in cycles if name != "J4"})
res = quantize(circuit, loops)  # automatic basis: three compact modes, free mode removed
res""", ["res.coords.types", "res.B.subs(Cg, 0).applyfunc(sym.simplify)"]),
}

for name, (title, build, shows) in EXAMPLES.items():
    cells = [new_markdown_cell(f"# {title}\n\nEach step follows the notes; equation labels are given in comments."),
             new_code_cell(SETUP), new_code_cell(build)]
    cells += [new_code_cell(expr) for expr in shows]
    nbformat.write(new_notebook(cells=cells), f"examples/{name}.ipynb")
    print("wrote", name)
