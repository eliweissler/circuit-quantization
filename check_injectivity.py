import numpy as np

# A 2x1 injective matrix
omega = np.array([[1], [np.sqrt(2)]])
# The lattice P = {a in R | omega * a in Z^2}
# a must be an integer, AND a * sqrt(2) must be an integer.
# This implies a = 0.
# The rank is 0, not 1!

print("If omega is [1, sqrt(2)]^T, injectivity holds, but rank is 0.")

