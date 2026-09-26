import sympy as sp
N_v, N_L, N_ext = 2, 2, 1

# Explicit symbols instead of MatrixSymbols to avoid non-commutative bugs
L1, L2 = sp.symbols('L1 L2')
L_b_inv = sp.Matrix([[L1, 0], [0, L2]])

p1, p2 = sp.symbols('p1 p2')
Phi = sp.Matrix([[p1], [p2]])

w11, w12, w21, w22 = sp.symbols('w11 w12 w21 w22')
omega_L = sp.Matrix([[w11, w12], [w21, w22]])

b1, b2 = sp.symbols('b1 b2')
B_L = sp.Matrix([[b1], [b2]])

e1 = sp.symbols('e1')
Phi_ext = sp.Matrix([[e1]])

Phi_b_L = omega_L.T * Phi + B_L * Phi_ext

U_L_sub = (Phi_b_L.T * L_b_inv * Phi_b_L)[0,0] / 2
U_L_expanded = ((Phi.T * omega_L * L_b_inv * omega_L.T * Phi)[0,0]/2 + (Phi.T * omega_L * L_b_inv * B_L * Phi_ext)[0,0] + (Phi_ext.T * B_L.T * L_b_inv * B_L * Phi_ext)[0,0]/2)

assert sp.simplify(U_L_sub - U_L_expanded) == 0, 'eq:U_full failed'
print("Pass")
