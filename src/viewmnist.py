import torch
from torchvision import datasets
import matplotlib.pyplot as plt

# Load FashionMNIST dataset
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True
)

# Labels for FashionMNIST
labels_map = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

# Display first few images
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = i
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img, cmap="gray")
plt.show()

# To save a single image as PNG:
img, label = training_data[0]
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.savefig("fashion_mnist_example.png")