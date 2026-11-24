import torch

print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    # a tiny calculation on the GPU to prove it really works
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x
    print("GPU matrix multiply OK, result shape:", tuple(y.shape))
else:
    print("No GPU found. Training would run on CPU (slow).")