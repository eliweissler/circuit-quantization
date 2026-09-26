import sympy as sp

# 1. Check dimensional mismatch in eq:ZY_blocks
print("--- Check eq:ZY_blocks ---")
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

# 2. Check eq:transJZ2 inconsistency
print("--- Check eq:transJZ2 ---")
# The text claims: omega_tilde * Z = [omega_1, 0]
# But omega_tilde is a row vector of size N_dyn + N_non_dyn
omega_1 = sp.MatrixSymbol('omega_1', 1, N_dyn)
omega_2 = sp.ZeroMatrix(1, N_fr + N_fz + N_cy)
omega_tilde = sp.BlockMatrix([[omega_1, omega_2]])

# Z maps from N_dyn + N_non_dyn to N_int
Z = sp.BlockMatrix([[Z1, Z2]])

try:
    res = omega_tilde * Z
    print("omega_tilde * Z shape:", res.shape)
except Exception as e:
    print("omega_tilde * Z raises:", type(e).__name__)

Y = sp.BlockMatrix([[sp.Identity(N_dyn), sp.ZeroMatrix(N_dyn, N_fr + N_fz + N_cy)], 
                    [sp.MatrixSymbol('M', N_fr + N_fz + N_cy, N_dyn), sp.Identity(N_fr + N_fz + N_cy)]])
res_Y = omega_tilde * Y
print("omega_tilde * Y evaluates to:", res_Y.blocks[0,0], res_Y.blocks[0,1])

