import time
import torch
from gemm_lowbit_cpp import gemm_lowbit_forward

torch.set_default_device("cuda")  # Set default device to CUDA

# Example usage
a = torch.ones(1000, 2000, dtype=torch.half, device="cuda")  # Example tensor
b = torch.ones(2000, 3000, dtype=torch.half, device="cuda")  # Example tensor
c = torch.empty(1000, 3000, dtype=torch.half, device="cuda")  # Output tensor

tic = time.time_ns()  # Example timer
# Call the custom CUDA GEMM operation
gemm_lowbit_forward(a, b, c)
toc = time.time_ns()  # Example timer

print(f"Time taken: {(toc - tic) / 1e6:.6f} milliseconds")  # Example timer output

tic = time.time_ns()  # Example timer
ab = a @ b  # Example matrix multiplication
toc = time.time_ns()  # Example timer

print(f"Time taken: {(toc - tic) / 1e6:.6f} milliseconds")  # Example timer output

print(c.shape, ab.shape)  # Example output comparison

print(((c - ab) ** 2).mean())  # Example loss value
