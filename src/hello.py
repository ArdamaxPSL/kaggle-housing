import torch
import numpy as np

# Check if CUDA (GPU support) is available
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# Create a simple tensor
x = torch.rand(3, 3)
print("\nRandom Tensor:")
print(x)

# Basic tensor operations
y = x * 2
print("\nTensor multiplied by 2:")
print(y)

# Convert NumPy array to PyTorch tensor and back
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
tensor_from_numpy = torch.from_numpy(numpy_array)
back_to_numpy = tensor_from_numpy.numpy()

print("\nNumPy to PyTorch to NumPy conversion:")
print("Original NumPy array:")
print(numpy_array)
print("As PyTorch tensor:")
print(tensor_from_numpy)