import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

import numpy as np

input_size = 28
num_actions = 20
hidden_units = 64
time_steps = 20

model_keras = Sequential([
    LSTM(hidden_units, input_shape=(time_steps, input_size), unroll=True),
    Dense(num_actions, activation='softmax')
])

model_keras.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy')


import torch
import torch.nn as nn
import torch.optim as optim

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_units, time_steps):
        super(CustomLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_units, batch_first=True)
        self.time_steps = time_steps

    def forward(self, x):
        # Assuming x has shape (batch_size, time_steps, input_size)
        output, _ = self.lstm(x)
        return output[:, -1, :]  # Return only the output of the last time step

class ModelTorch(nn.Module):
    def __init__(self, input_size, num_actions, hidden_units, time_steps):
        super(ModelTorch, self).__init__()
        self.lstm_layer = CustomLSTM(input_size, hidden_units, time_steps)
        self.dense_layer = nn.Linear(hidden_units, num_actions)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lstm_output = self.lstm_layer(x)
        dense_output = self.dense_layer(lstm_output)
        return self.softmax(dense_output)

model_torch = ModelTorch(input_size, num_actions, hidden_units, time_steps)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_torch.parameters(), lr=0.001)

model_torch.eval()

data = np.array([[0] * 28] * 20).reshape(1, 20, 28)

start_keras = time.time()
for i in range(100):
    model_keras.predict(data, verbose=0)
end_keras = time.time()
print("Keras 100 predict iterations took {:.2f} seconds".format(end_keras - start_keras))
print("Keras: {:.2f} ms per prediction".format((end_keras - start_keras) / 100 * 1000))

start_torch = time.time()
for i in range(100):
    with torch.no_grad():
        model_torch(torch.from_numpy(data).to(torch.float32))
end_torch = time.time()
print("Torch 100 predict iterations took {:.2f} seconds".format(end_torch - start_torch))
print("Torch: {:.2f} ms per prediction".format((end_torch - start_torch) / 100 * 1000))