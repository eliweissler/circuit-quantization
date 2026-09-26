import sympy as sp

N_int = 5
N_dyn = 2
N_fr = 1
N_fz = 1
N_cy = 1

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
    print("Mc + Ml failed:", e)

M = sp.BlockMatrix([[Mc], [Ml], [sp.ZeroMatrix(N_cy, N_dyn)]])
Z_ret = Z1 + Z2 * M
print("Z1 + Z2*M shape:", Z_ret.shape)
