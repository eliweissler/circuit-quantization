import sympy as sp

N_int = 3
N_J = 2
N_gnd = 1

# Define dimensions
omega_J_int = sp.MatrixSymbol('omega_J_int', N_int, N_J)
omega_J_gnd = sp.MatrixSymbol('omega_J_gnd', N_gnd, N_J)
Phi_int = sp.MatrixSymbol('Phi_int', N_int, 1)
Phi_gnd = sp.MatrixSymbol('Phi_gnd', N_gnd, 1)

try:
    # This evaluates \omega_{J,int}^T \Phi_{int} + \omega_{J,gnd} \Phi_{gnd}
    term1 = omega_J_int.T * Phi_int
    term2 = omega_J_gnd * Phi_gnd
    expr = term1 + term2
    print("SUCCESS")
except Exception as e:
    print("ERROR:", str(e))
