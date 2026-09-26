import sympy as sp
import numpy as np

# A circuit with 1 free mode, 1 frozen mode.
N_fr, N_fz = 1, 1
N_nd = N_fr + N_fz

C_fr = sp.MatrixSymbol('C_fr', N_fr, N_fr)
# The full C~_22 for the ND block has a zero block for the frozen mode
C_22 = sp.BlockMatrix([[C_fr, sp.ZeroMatrix(N_fr, N_fz)], 
                       [sp.ZeroMatrix(N_fz, N_fr), sp.ZeroMatrix(N_fz, N_fz)]])

print("Shape of C~_22:", C_22.shape)
print("Is C~_22 singular with N_fz > 0? Yes.")
