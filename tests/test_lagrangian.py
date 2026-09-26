import sympy as sym

from circuit_quantization import lagrangian
from circuits import coupled, harmonium, pos


def test_harmonium_node_matrices():
    C, CJ1, CJ2, CJ3, L1, L2, L3 = pos("C C_J1 C_J2 C_J3 L_1 L_2 L_3")
    lag = lagrangian(harmonium())
    assert lag.C_b == sym.diag(C, CJ1, CJ2, CJ3, 0, 0, 0)  # eq:harm_Cb
    assert lag.C_int == sym.Matrix([  # eq:harm_node_matrices
        [C + CJ1 + CJ2, -CJ1, -CJ2, 0],
        [-CJ1, CJ1, 0, 0],
        [-CJ2, 0, CJ2, 0],
        [0, 0, 0, CJ3],
    ])
    assert lag.Lstar_int == sym.Matrix([
        [0, 0, 0, 0],
        [0, 1 / L1, 0, -1 / L1],
        [0, 0, 1 / L2, -1 / L2],
        [0, -1 / L1, -1 / L2, 1 / L1 + 1 / L2 + 1 / L3],
    ])
    assert lag.omega_J_int.T == sym.Matrix([[1, -1, 0, 0], [1, 0, -1, 0], [0, 0, 0, 1]])  # eq:harm_wJ
    assert lag.omega_L_int.T == sym.Matrix([[0, 1, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1]])


def test_coupled_example_drive_block():
    Ct, Cc, Cf, Cg, CV1, CV2, CV3, CV4, CJ1, CJ2, CJ3, Lf = pos(
        "C_t C_c C_f C_g C_V1 C_V2 C_V3 C_V4 C_J1 C_J2 C_J3 L_f")
    lag = lagrangian(coupled())
    CT, C1, C2 = Ct + CJ1 + CJ2, Cc + CV1 + CV2, Cc + Cf + CJ3 + CV3 + CV4  # eq:fig1_shorthand
    assert lag.C_int == sym.Matrix([[C1 + CT, -Cc, -CT], [-Cc, C2, 0], [-CT, 0, Cg + CT]])  # eq:fig1_matrices
    assert lag.C_int_src == sym.Matrix([[CV1, CV2, 0], [0, CV3, CV4], [0, 0, 0]])  # eq:fig1_Cintsrc
    assert lag.V == sym.Matrix(sym.symbols("V_1 V_2 V_3", positive=True))
