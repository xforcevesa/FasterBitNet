import time
import torch
from torch.nn import functional as F
from gemm_lowbit_cpp import gemm_lowbit_forward

# torch.set_default_device("cuda")  # Set default device to CUDA

# Example usage
a = torch.ones(3000, 3000, dtype=torch.int8, device="cuda")  # Example tensor
b = torch.ones(3000, 3000, dtype=torch.int8, device="cuda")  # Example tensor
c = torch.zeros(3000, 3000, dtype=torch.int8, device="cuda")  # Output tensor

tic = time.time_ns()  # Example timer
# Call the custom CUDA GEMM operation
gemm_lowbit_forward(a, b, c)
toc = time.time_ns()  # Example timer

print(f"Time taken: {(toc - tic) / 1e6:.6f} milliseconds")  # Example timer output

tic = time.time_ns()  # Example timer
ab = F.linear(a, b)  # Example matrix multiplication
toc = time.time_ns()  # Example timer

print(f"Time taken: {(toc - tic) / 1e6:.6f} milliseconds")  # Example timer output

print(c.shape, ab.shape)  # Example output comparison

print(F.mse_loss(c, ab).mean().cpu().item())  # Example loss value
