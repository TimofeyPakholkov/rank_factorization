from safetensors import safe_open
import numpy as np
import matplotlib.pyplot as plt

# Путь к файлу safetensors
file_path = "model.safetensors"

# Открытие файла и чтение содержимого
with safe_open(file_path, framework="numpy") as f:
    # Получение всех ключей
    keys = f.keys()

    # Чтение тензоров по ключам
    for key in keys:
        tensor = f.get_tensor(key)
        if len(tensor.shape) == 1:
            continue
        if tensor.shape[0] == 2048:
            print(f"Tensor '{key}':\n{tensor}")
        # print(f"Tensor '{key}':\n{tensor}")
        # print(tensor.shape)
        print(key, tensor.shape)
        U, S, V = np.linalg.svd(tensor)
        x_axis = np.array(range(np.min(tensor.shape)))
        # print(S)
        # print(y_axis)
        plt.plot(x_axis, S)
        plt.show()

        if tensor.shape[0] == 2048:
            R = 256
            U = U.T[0:R].T
            S = np.diag(S[0:R])
            V = V[0:R]

            print(U.shape, S.shape, V.shape)
            approx_matrix = U @ S @ V
            print(approx_matrix)
            print(np.sum(np.abs(tensor - approx_matrix)))
            
