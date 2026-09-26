import sympy as sp

N_int = 3
N_dyn = 2
N_non_dyn = 1
N_J = 1

Z = sp.MatrixSymbol('Z', N_int, N_int)
omega_J_int_T = sp.MatrixSymbol('omega_J_int_T', N_J, N_int)

omega_tilde = omega_J_int_T * Z
# We are given that omega_tilde is partitioned into [omega_1, 0]
# where omega_1 is N_J x N_dyn.
omega_1 = sp.MatrixSymbol('omega_1', N_J, N_dyn)
omega_2 = sp.ZeroMatrix(N_J, N_non_dyn)
omega_tilde_explicit = sp.BlockMatrix([[omega_1, omega_2]])

# Eq 53 claims: omega_tilde * Z == [omega_1, 0]
res_Z = sp.Matrix(omega_tilde_explicit) * sp.Matrix(Z)
print("omega_tilde * Z evaluates to:")
print(res_Z)

# What it meant to write: omega_tilde * Y == [omega_1, 0]
Y11 = sp.Identity(N_dyn)
Y12 = sp.ZeroMatrix(N_dyn, N_non_dyn)
Y21 = sp.MatrixSymbol('M', N_non_dyn, N_dyn)
Y22 = sp.Identity(N_non_dyn)
Y = sp.BlockMatrix([[Y11, Y12], [Y21, Y22]])

res_Y = sp.Matrix(omega_tilde_explicit) * sp.Matrix(Y)
print("omega_tilde * Y evaluates to:")
print(res_Y)

