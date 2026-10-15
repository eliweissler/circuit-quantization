"""Circuits from the notes, built exactly as drawn there."""

import sympy as sym

from circuit_quantization import Branch, Circuit

pos = lambda names: sym.symbols(names, positive=True)  # noqa: E731


def harmonium():
    """fig:harmonium, branches oriented low -> high node (eq:harm_Omega)."""
    C, CJ1, CJ2, CJ3, L1, L2, L3, EJ1, EJ2, EJ3 = pos("C C_J1 C_J2 C_J3 L_1 L_2 L_3 E_J1 E_J2 E_J3")
    return Circuit([
        Branch("C", 1, 5, C=C),
        Branch("J1", 1, 2, C=CJ1, EJ=EJ1),
        Branch("J2", 1, 3, C=CJ2, EJ=EJ2),
        Branch("J3", 4, 5, C=CJ3, EJ=EJ3),
        Branch("L1", 2, 4, L=L1),
        Branch("L2", 3, 4, L=L2),
        Branch("L3", 4, 5, L=L3),
    ], ground=5)


HARMONIUM_CYCLES = {  # eq:harm_G
    "k": {"J1": 1, "J2": -1, "L1": 1, "L2": -1},
    "f": {"J3": 1, "L3": -1},
    "cb": {"C": 1, "J1": -1, "J3": -1, "L1": -1},
}
HARMONIUM_Z = sym.Matrix([[1, 0, 0, 0], [0, 1, 1, 1], [0, 1, -1, 1], [0, 1, 0, 0]])  # eq:harm_Z


def zero_pi():
    """fig:zeropi_kite left, symmetric; floating, with no ground (eq:zp_Omega)."""
    CJ, C, L, EJ = pos("C_J C L E_J")
    return Circuit([
        Branch("J1", 1, 2, C=CJ, EJ=EJ),
        Branch("J2", 3, 4, C=CJ, EJ=EJ),
        Branch("L1", 2, 3, L=L),
        Branch("L2", 4, 1, L=L),
        Branch("C1", 1, 3, C=C),
        Branch("C2", 2, 4, C=C),
    ])


def zero_pi_asymmetric():
    """zero_pi with every element distinct, so the compact shear and the e-h normal modes are both nontrivial."""
    CJ1, CJ2, C1, C2, L1, L2, EJ1, EJ2 = pos("C_J1 C_J2 C_1 C_2 L_1 L_2 E_J1 E_J2")
    return Circuit([
        Branch("J1", 1, 2, C=CJ1, EJ=EJ1),
        Branch("J2", 3, 4, C=CJ2, EJ=EJ2),
        Branch("L1", 2, 3, L=L1),
        Branch("L2", 4, 1, L=L2),
        Branch("C1", 1, 3, C=C1),
        Branch("C2", 2, 4, C=C2),
    ])


def fluxonium_pair():
    """Two grounded fluxoniums coupled by C_c: two extended modes, no compact mode, so no shear."""
    C1, C2, Cc, CJ1, CJ2, L1, L2, EJ1, EJ2 = pos("C_1 C_2 C_c C_J1 C_J2 L_1 L_2 E_J1 E_J2")
    return Circuit([
        Branch("C1", 1, 0, C=C1),
        Branch("J1", 1, 0, C=CJ1, EJ=EJ1),
        Branch("L1", 1, 0, L=L1),
        Branch("C2", 2, 0, C=C2),
        Branch("J2", 2, 0, C=CJ2, EJ=EJ2),
        Branch("L2", 2, 0, L=L2),
        Branch("Cc", 1, 2, C=Cc),
    ], ground=0)


def transmon_fluxonium():
    """A grounded transmon coupled by C_c to a grounded fluxonium: one compact, one extended mode, so only the shear."""
    Ct, Cf, Cc, CJ1, CJ2, Lf, EJ1, EJ2 = pos("C_t C_f C_c C_J1 C_J2 L_f E_J1 E_J2")
    return Circuit([
        Branch("Ct", 1, 0, C=Ct),
        Branch("J1", 1, 0, C=CJ1, EJ=EJ1),
        Branch("Cf", 2, 0, C=Cf),
        Branch("J2", 2, 0, C=CJ2, EJ=EJ2),
        Branch("Lf", 2, 0, L=Lf),
        Branch("Cc", 1, 2, C=Cc),
    ], ground=0)


ZERO_PI_CYCLES = {  # eq:zp_G
    "ring": {"J1": 1, "J2": 1, "L1": 1, "L2": 1},
    "t1": {"J1": 1, "L1": 1, "C1": -1},
    "t2": {"J2": 1, "L1": 1, "C2": -1},
}
ZERO_PI_Z = sym.Matrix([[-1, -1, 1, 1], [1, 1, 1, 1], [-1, 1, -1, 1], [1, -1, -1, 1]]) / 2  # eq:zp_Z


def kite():
    """cos(2 phi) transmon, island 1, arm midpoints 2 and 3, ground 4 (eq:kite_node)."""
    CS, CJ1, CJ2, L1, L2, EJ1, EJ2 = pos("C_S C_J1 C_J2 L_1 L_2 E_J1 E_J2")
    return Circuit([
        Branch("CS", 1, 4, C=CS),
        Branch("J1", 1, 2, C=CJ1, EJ=EJ1),
        Branch("J2", 1, 3, C=CJ2, EJ=EJ2),
        Branch("L1", 2, 4, L=L1),
        Branch("L2", 3, 4, L=L2),
    ], ground=4)


KITE_CYCLES = {"kite": {"J1": 1, "J2": -1, "L1": 1, "L2": -1}, "cb": {"CS": 1, "J1": -1, "L1": -1}}  # eq:kite_G
KITE_Z = sym.Matrix([[1, 0, 0], [0, 1, 1], [0, -1, 1]])  # eq:kite_Z


def rhombus():
    """fig:rhombus with equal ground capacitances; J_k from node k to k+1."""
    CJ, CB, Cg, EJ, alpha = pos("C_J C_B C_g E_J alpha")
    return Circuit([
        Branch("J1", 1, 2, C=CJ, EJ=EJ),
        Branch("J2", 2, 3, C=CJ, EJ=EJ),
        Branch("J3", 3, 4, C=CJ, EJ=EJ),
        Branch("J4", 4, 1, C=alpha * CJ, EJ=alpha * EJ),
        Branch("CB", 1, 3, C=CB),
        *(Branch(f"Cg{i}", i, 5, C=Cg) for i in range(1, 5)),
    ], ground=5)


RHOMBUS_CYCLES = {  # eq:rh_G
    "ring": {"J1": 1, "J2": 1, "J3": 1, "J4": 1},
    "shunt": {"J1": -1, "J2": -1, "CB": 1},
    "g12": {"J1": -1, "Cg1": 1, "Cg2": -1},
    "g23": {"J2": -1, "Cg2": 1, "Cg3": -1},
    "g34": {"J3": -1, "Cg3": 1, "Cg4": -1},
}
RHOMBUS_Z = sym.Matrix([  # eq:rh_Z at equal C_g, given there as Z^{-1}
    [-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1], [sym.Rational(1, 4)] * 4,
]).inv()


def coupled():
    """fig:coupled_circuit_example: transmon on island 3 coupled to a fluxonium, three sources."""
    Ct, Cc, Cf, Cg, CV1, CV2, CV3, CV4, CJ1, CJ2, CJ3, Lf = pos(
        "C_t C_c C_f C_g C_V1 C_V2 C_V3 C_V4 C_J1 C_J2 C_J3 L_f")
    EJ1, EJ2, EJ3, V1, V2, V3 = sym.symbols("E_J1 E_J2 E_J3 V_1 V_2 V_3", positive=True)
    return Circuit([
        Branch("Ct", 1, 3, C=Ct),
        Branch("Cc", 1, 2, C=Cc),
        Branch("Cf", 2, 4, C=Cf),
        Branch("CV1", 1, 5, C=CV1),
        Branch("CV2", 1, 6, C=CV2),
        Branch("CV3", 2, 6, C=CV3),
        Branch("CV4", 2, 7, C=CV4),
        Branch("Cg", 3, 4, C=Cg),
        Branch("J1", 1, 3, C=CJ1, EJ=EJ1),
        Branch("J2", 1, 3, C=CJ2, EJ=EJ2),
        Branch("J3", 2, 4, C=CJ3, EJ=EJ3),
        Branch("Lf", 2, 4, L=Lf),
        Branch("V1", 4, 5, V=V1),
        Branch("V2", 4, 6, V=V2),
        Branch("V3", 4, 7, V=V3),
    ], ground=4)


def parallel(*branches):
    """Branches all from node 1 to ground 0."""
    return Circuit(list(branches), ground=0)


def coupled_zero_pis(scheme="a"):
    """Two symmetric zero-pi qubits coupled through capacitors C_c.

    Circuits of the zero-pi two-qubit gate paper: qubit nodes 1-4 and 5-8, each
    with C_0 to ground (node 0).  Scheme (a), a pure n_theta1 n_theta2 coupler:
    nodes 1, 4, 5, 8 couple to node 10 and nodes 2, 3, 6, 7 to node 9.
    Scheme (b), the mixed coupler: node pairs (1, 8), (2, 7), (3, 6), (4, 5).
    """
    CJ, C, L, C0, Cc, EJ = pos("C_J C L C_0 C_c E_J")
    branches = []
    for q, (a, b, c, d) in enumerate(((1, 2, 3, 4), (5, 6, 7, 8)), start=1):
        branches += [
            Branch(f"J{q}a", a, b, C=CJ, EJ=EJ), Branch(f"J{q}b", c, d, C=CJ, EJ=EJ),
            Branch(f"L{q}a", b, c, L=L), Branch(f"L{q}b", d, a, L=L),
            Branch(f"C{q}a", a, c, C=C), Branch(f"C{q}b", b, d, C=C),
            *(Branch(f"C0_{n}", n, 0, C=C0) for n in (a, b, c, d)),
        ]
    if scheme == "a":
        branches += [Branch(f"Cc_{n}", n, 10, C=Cc) for n in (1, 4, 5, 8)]
        branches += [Branch(f"Cc_{n}", n, 9, C=Cc) for n in (2, 3, 6, 7)]
    else:
        branches += [Branch(f"Cc_{a}{b}", a, b, C=Cc) for a, b in ((1, 8), (2, 7), (3, 6), (4, 5))]
    return Circuit(branches, ground=0)


def _coupled_zero_pi_Z():
    # Rows theta, phi, zeta, Sigma of the paper's transform, whose inverse is its transpose.
    M = sym.Matrix([[-1, 1, 1, -1], [-1, 1, -1, 1], [1, 1, -1, -1], [1, 1, 1, 1]]) / 2
    Z = sym.zeros(10, 10)
    Z[0:4, 0:4] = M.T
    Z[4:8, 4:8] = M.T
    Z[8:10, 8:10] = sym.Matrix([[1, 1], [1, -1]]) / 2  # varphi_+- = varphi_9 +- varphi_10
    return Z


COUPLED_ZERO_PI_Z = _coupled_zero_pi_Z()
COUPLED_ZERO_PI_TYPES = ("c", "e", "h", "fr", "c", "e", "h", "fr", "fr", "fr")


def random_circuit(seed, nodes=4, extra=3):
    """A grounded random circuit with small rational element values.

    A random spanning tree ties every node to ground, then `extra` random
    branches are added. Inductors carry no capacitance, so a node reached only
    through inductors is frozen.
    """
    import random

    rng = random.Random(seed)
    value = lambda: sym.Rational(rng.randint(1, 9), rng.randint(1, 9))  # noqa: E731
    branches = []
    for node in range(1, nodes + 1):
        edge = (node, rng.randrange(node))
        branches.append(_random_branch(rng, value, len(branches), edge))
    for _ in range(extra):
        branches.append(_random_branch(rng, value, len(branches), tuple(rng.sample(range(nodes + 1), 2))))
    return Circuit(branches, ground=0)


def _random_branch(rng, value, index, edge):
    kind = rng.choice("CJL")
    if kind == "L":
        return Branch(f"L{index}", *edge, L=value())
    return Branch(f"{kind}{index}", *edge, C=value(), EJ=value() if kind == "J" else None)
