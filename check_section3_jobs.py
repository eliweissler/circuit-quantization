import sympy as sp

print("Checking eq:ZY_blocks dimensional inconsistency...")
N_int, N_dyn, N_fr, N_fz, N_cy = 5, 2, 2, 1, 1

Z1 = sp.MatrixSymbol('Z1', N_int, N_dyn)
Z_fr = sp.MatrixSymbol('Z_fr', N_int, N_fr)
Z_fz = sp.MatrixSymbol('Z_fz', N_int, N_fz)
Z_cy = sp.MatrixSymbol('Z_cy', N_int, N_cy)
Z2 = sp.BlockMatrix([[Z_fr, Z_fz, Z_cy]])

Mc = sp.MatrixSymbol('Mc', N_fr, N_dyn)
Ml = sp.MatrixSymbol('Ml', N_fz, N_dyn)

try:
    M_add = Mc + Ml
except Exception as e:
    print("Mc + Ml raises:", type(e).__name__)

print("Checking eq:transJZ2 inconsistency...")
omega_1 = sp.MatrixSymbol('omega_1', 1, N_dyn)
omega_2 = sp.ZeroMatrix(1, N_fr + N_fz + N_cy)
omega_tilde = sp.BlockMatrix([[omega_1, omega_2]])

Z = sp.BlockMatrix([[Z1, Z2]])

try:
    res = omega_tilde * Z
    res.shape
except Exception as e:
    print("omega_tilde * Z raises:", type(e).__name__)

