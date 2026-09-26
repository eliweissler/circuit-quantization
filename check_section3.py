import sympy as sp

N_int = 5
N_dyn = 2
N_non_dyn = 3
N_J = 1

omega_J_int_T = sp.MatrixSymbol('omega_J_int_T', N_J, N_int)
Z = sp.MatrixSymbol('Z', N_int, N_dyn + N_non_dyn)

# Eq 33 defines omega_tilde
omega_tilde = omega_J_int_T * Z

try:
    res = omega_tilde * Z
    print("omega_tilde * Z shape:", res.shape)
except Exception as e:
    print("omega_tilde * Z raises:", type(e).__name__)

N_fr = 2
N_fz = 1
Mc = sp.MatrixSymbol('Mc', N_fr, N_dyn)
Ml = sp.MatrixSymbol('Ml', N_fz, N_dyn)

try:
    M_add = Mc + Ml
except Exception as e:
    print("Mc + Ml raises:", type(e).__name__)

