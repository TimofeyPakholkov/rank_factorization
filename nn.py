import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from safetensors import safe_open
import numpy as np

# Путь к файлу safetensors
file_path = "model.safetensors"
tensor = 1

# Открытие файла и чтение содержимого
with safe_open(file_path, framework="numpy") as f:
    # Получение всех ключей
    keys = f.keys()
    print("Keys:", keys)

    # Чтение тензоров по ключам
    for key in keys:
        tensor = f.get_tensor(key)
        print(f"Tensor '{key}':\n{tensor}")
        print(tensor.shape)
        break

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, r, output_size):
        super(SimpleNeuralNetwork, self).__init__()
        # self.layer1 = nn.Linear(input_size, r)
        self.layer2 = nn.Linear(r, output_size)
        
    def forward(self, x):
        # x = self.layer1(x)  # Линейный слой
        x = self.layer2(x)  # Линейный слой
        return x

# Параметры
input_size = 512
r = 256
output_size = 512
batch_size = input_size

# labels = torch.randn(input_size, output_size)
labels = torch.tensor(np.array(tensor), dtype=torch.float32)
# labels = torch.tensor(np.array([[-0.8332, -2.5315, -0.1515, -0.3114,  1.7202,  0.7262],
#         [ 0.3093,  0.9528, -0.3027,  0.9132,  1.1356, -2.4898],
#         [-0.0733,  0.0513, -0.0386, -0.3240, -0.7363, -0.2733],
#         [ 0.0209, -0.3517,  0.5106, -0.7846, -2.0237,  1.6777],
#         [ 1.4122, -0.5706,  1.4275,  0.2872, -0.4248,  1.4295]]))

# print(labels)
# print(labels.shape)
# print(labels.T[0:r].T)
# print(labels.T[0:r].T.shape)
# exit()

# Пример входных данных и целевых меток
# inputs = torch.randn(batch_size, input_size)

# inputs = torch.tensor(np.eye(batch_size, input_size))
inputs = torch.tensor(labels.T[0:r].T)

# print(inputs)
# print(labels)

# Создание модели
model = SimpleNeuralNetwork(input_size, r, output_size)

# Определение функции потерь и оптимизатора
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1)

l_rate = 1

for i in range(1, 10000):
    model.train()
    # Прямой проход (forward pass)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    if (i==1):
        print(loss)

    # Обратный проход (backward pass) и оптимизация
    if i % 5000 == 0:
        l_rate /= 2
        print(l_rate)
        optimizer = optim.SGD(model.parameters(), lr=l_rate)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 1000 == 0:
        print(f'Loss: {loss.item()}')

model.eval()
outputs = model(inputs)
print(outputs)
print(labels)
