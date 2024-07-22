import numpy as np
import sympy


# matrix = np.array([[1.01,2,3,2],[1,0,2,4],[4,4,2,1],[4,3,2,1]], dtype=np.float32)
matrix = np.array(np.random.randn(5, 5), dtype=np.float32)
sympy_matrix = sympy.Matrix(matrix)

gram_matrix = matrix.T @ matrix

# print(matrix)
# print(gram_matrix)
# print(matrix)
# matrix[0] -= (float(matrix[0][0]) / matrix[1][0]) * matrix[1]

print(matrix)
print(matrix.T @ matrix)
print(np.sum())
# L, W = np.linalg.eig(gram_matrix)
# print(L.real)
# print(W)
# print((W * L.real) @ np.array([1,2,3,4,5]))
# print(matrix @ np.array([1,2,3,4,5]))

# B_matrix = np.array(sympy_matrix.rref()[0])
# C_matrix = np.array(sympy_matrix.columnspace())
# C_matrix = np.reshape(C_matrix, newshape=(C_matrix.shape[1], C_matrix.shape[0]))
# F_matrix = B_matrix[~np.all(B_matrix == 0, axis=1)]
# print(matrix)
# print(B_matrix)
# print(B_matrix[0:2])
# print(sympy_matrix.rref()[1])
# print(np.linalg.eigvals(matrix).real)
# print(C_matrix @ F_matrix)
