"""
MNIST:
PCA: 0.006
SVD: 0.006

FMNIST:
PCA: 0.008
SVD: 0.008

"""




import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import mean_squared_error
import numpy as np
import cv2 as cv
import os

dataset = "fmnist"  # "mnist" or "fmnist"
reducer_type = "svd" # pca or svd

if dataset == "mnist":
    load_dataset = datasets.MNIST
elif dataset == "fmnist":
    load_dataset = datasets.FashionMNIST

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = load_dataset(root='./datasets', train=True, download=True, transform=transform)
test_dataset = load_dataset(root='./datasets', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

train_data = next(iter(train_loader))[0].reshape(-1, 784).numpy()
test_data = next(iter(test_loader))[0].reshape(-1, 784).numpy()

if reducer_type == "pca":
    reducer = PCA(n_components=78)
else:
    reducer = TruncatedSVD(n_components=78)

train_data_pca = reducer.fit_transform(train_data)

test_data_pca = reducer.transform(test_data)

test_data_reconstructed = reducer.inverse_transform(test_data_pca)

test_data_reconstructed = np.clip(test_data_reconstructed, 0, 1)


mse = mean_squared_error(test_data, test_data_reconstructed)

save_dir = os.path.join(dataset, reducer_type)
os.makedirs(save_dir, exist_ok=True)

x = test_data_reconstructed.reshape(-1, 28, 28)
for a in range(100):
    cur = (x[a] * 255).astype(np.uint8)
    cv.imwrite(os.path.join(save_dir, f"{a}.bmp"), cur)

print(f"Mean Squared Error (MSE) on the test set: {mse}")
