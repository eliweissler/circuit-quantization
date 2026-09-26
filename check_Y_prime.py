import sympy as sp

C11, C12, C21, C22, M = sp.symbols('C11 C12 C21 C22 M')

C = sp.Matrix([[C11, C12], [C21, C22]])
Y_prime = sp.Matrix([[1, M], [0, 1]])

res = Y_prime.T * C * Y_prime
print(res[0,0])
print(res[0,1])
print(res[1,0])
print(res[1,1])
