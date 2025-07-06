import torch
x = torch.rand(5, 3)

print("Torch version:", torch.__version__)
print("ROCm version:", torch.version.hip)
print(torch.cuda.is_available())
print(x)

