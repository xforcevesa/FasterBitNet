import torch
from bitnet import BitLinearNew

torch.set_default_device("cuda")

# Create a random tensor of shape (16, 10)
x = torch.randn(16, 1000, 512)

# Create an instance of the BitLinearNew class with input size 10, output size 20, and 2 groups
layer = BitLinearNew(512, 20)

import time

start_time = time.time()

# Perform a forward pass through the BitLinearNew layer with input x
output = layer(x)

# Print the output tensor
# print(output)
print(output.shape)

print("--- %s seconds ---" % (time.time() - start_time))

layer = torch.nn.Linear(512, 20)

start_time = time.time()

# Perform a forward pass through the torch.nn.Linear layer with input x
output = layer(x)

# Print the output tensor
# print(output)
print(output.shape)

print("--- %s seconds ---" % (time.time() - start_time))
