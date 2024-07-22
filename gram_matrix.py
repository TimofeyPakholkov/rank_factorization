import numpy as np


# matrix = np.array([[1,1,3,1,5],[2,3,56,5,2],[3,1,11,6,7],[4,6,6,7,63],[5,6,1,8,1]], dtype=np.float16)
matrix = np.array(np.random.randn(5, 5) * 10, dtype=np.float16)
matrix_transposed = matrix.T
gram_matrix = matrix_transposed @ matrix
print(matrix)
print(gram_matrix)
for i in range(gram_matrix.shape[0]):
    for j in range(gram_matrix.shape[1]):
        if i != j:
            gram_matrix[i][j] /= np.sqrt(gram_matrix[i][i]) * np.sqrt(gram_matrix[j][j])
print(gram_matrix)
dot_product_sum = np.zeros((gram_matrix.shape[0],))
for i in range(gram_matrix.shape[0]):
    dot_product_sum[i] = np.sum(np.abs(gram_matrix[i])) - gram_matrix[i][i]
print(dot_product_sum)
