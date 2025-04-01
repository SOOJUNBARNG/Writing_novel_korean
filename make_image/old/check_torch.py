# import torch
# print(torch.__version__)
# print("CUDA Available:", torch.cuda.is_available())


import torch

device = "cuda"

# Increase batch size to use more VRAM
x = torch.randn(10000, 10000, device=device)  # Large tensor
y = torch.randn(10000, 10000, device=device)
z = x @ y  # Matrix multiplication (high memory usage)

print("Computation done on:", z.device)