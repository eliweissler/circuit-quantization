import sympy as sp
Z11, Z12, Z21, Z22 = sp.symbols('Z11 Z12 Z21 Z22')
Z = sp.Matrix([[Z11, Z12], [Z21, Z22]])

t1, t2 = sp.symbols('t1 t2')
Phi_tilde = sp.Matrix([[t1], [t2]])

Phi_int_sub = Z * Phi_tilde

c11, c12, c21, c22 = sp.symbols('c11 c12 c21 c22')
C_int = sp.Matrix([[c11, c12], [c21, c22]])
T_int = (Phi_int_sub.T * C_int * Phi_int_sub)[0,0] / 2
C_tilde = Z.T * C_int * Z
T_tilde = (Phi_tilde.T * C_tilde * Phi_tilde)[0,0] / 2

assert sp.simplify(T_int - T_tilde) == 0, 'eq:transformed_objects failed'
print("Pass")
