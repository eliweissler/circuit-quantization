import sympy as sp

N_int = 2
N_J = 3

Phi_int = sp.MatrixSymbol('Phi_int', N_int, 1)
Phi_gnd_scalar = sp.Symbol('Phi_gnd_scalar')

omega_J_int = sp.MatrixSymbol('w_J_int', N_int, N_J)
omega_J_gnd = sp.MatrixSymbol('w_J_gnd', 1, N_J) # 1 x N_J row vector

term1 = omega_J_int.T * Phi_int  # N_J x 1 column vector
term2_written = omega_J_gnd * Phi_gnd_scalar # scalar mult: 1 x N_J row vector
term2_correct = omega_J_gnd.T * Phi_gnd_scalar # scalar mult: N_J x 1 column vector

try:
    res = term1 + term2_written
except Exception as e:
    print("CHALLENGE REBUTTAL: Addition fails with:", type(e).__name__)

try:
    res_correct = term1 + term2_correct
    print("CHALLENGE REBUTTAL: Addition succeeds when transposed, shape is", res_correct.shape)
except Exception as e:
    pass

