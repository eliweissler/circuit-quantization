import sympy as sp
import numpy as np

N_int = 2
N_gnd = 1
N_J = 3

# Define shapes
Phi_int = sp.MatrixSymbol('Phi_int', N_int, 1)
Phi_gnd = sp.MatrixSymbol('Phi_gnd', N_gnd, 1) # scalar if N_gnd=1

omega_J_int = sp.MatrixSymbol('w_J_int', N_int, N_J)
omega_J_gnd = sp.MatrixSymbol('w_J_gnd', N_gnd, N_J) # 1 x N_J row vector

term1 = omega_J_int.T * Phi_int  # (N_J x N_int) * (N_int x 1) -> N_J x 1
term2_written = omega_J_gnd * Phi_gnd # (1 x N_J) * (1 x 1) -> 1 x N_J
term2_correct = omega_J_gnd.T * Phi_gnd # (N_J x 1) * (1 x 1) -> N_J x 1

try:
    res = term1 + term2_written
except Exception as e:
    print("CHALLENGE REBUTTAL: Addition fails with:", type(e).__name__)

try:
    res_correct = term1 + term2_correct
    print("CHALLENGE REBUTTAL: Addition succeeds when transposed, shape is", res_correct.shape)
except Exception as e:
    pass

