import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from torchvision.transforms import ToTensor

# GPU kullanmayacağız
device = torch.device("cpu")
os.makedirs("results", exist_ok=True)

def load_fmnist_examples(n_samples=10):
    dataset = torchvision.datasets.FashionMNIST(root="datasets", train=False, download=True, transform=ToTensor())
    images = torch.stack([dataset[i][0] for i in range(n_samples)])
    return images.numpy()

def flatten(images):
    return images.reshape(images.shape[0], -1)

def unflatten(images_flat, shape=(1, 28, 28)):
    return images_flat.reshape(-1, *shape)

def run_pca(images, n_components=78):
    x_flat = flatten(images)
    pca = PCA(n_components=min(n_components, x_flat.shape[0], x_flat.shape[1]))
    x_reduced = pca.fit_transform(x_flat)
    x_reconstructed = pca.inverse_transform(x_reduced)
    return unflatten(x_reconstructed)

def run_svd(images, n_components=78):
    x_flat = flatten(images)
    svd = TruncatedSVD(n_components=min(n_components, x_flat.shape[0], x_flat.shape[1]-1))
    x_reduced = svd.fit_transform(x_flat)
    x_reconstructed = svd.inverse_transform(x_reduced)
    return unflatten(x_reconstructed)

def mse(x, y):
    return np.mean((x - y) ** 2)

def save_images(original, ff, pca, svd, idx):
    fig, axes = plt.subplots(1, 4, figsize=(8, 2))
    for ax, img, title in zip(axes, [original, ff, pca, svd], ["Original", "FF", "PCA", "SVD"]):
        ax.imshow(img.squeeze(), cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"results/compare_{idx}.png")
    plt.close()

def load_ff_reconstructions(path="models/fmnist_unsup-model.pt", n_samples=10):
    from src import models, utils
    from omegaconf import OmegaConf

    opt = OmegaConf.load("config.yaml")
    opt = utils.parse_args(opt)

    model = models.Model(opt)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    dataset = torchvision.datasets.FashionMNIST(root="datasets", train=False, download=True, transform=ToTensor())
    inputs = torch.stack([dataset[i][0] for i in range(n_samples)])
    inputs_flat = inputs.view(inputs.size(0), -1).to(device)

    with torch.no_grad():
        reconstructed = model.linear_classifier(model.model(inputs_flat)).view(-1, 1, 28, 28)
    return inputs.numpy(), reconstructed.cpu().numpy()

def show_results_table_and_plot(results_dict):
    df = pd.DataFrame.from_dict(results_dict, orient='index', columns=['MSE'])
    df.index.name = 'Method'
    df = df.reset_index()

    print("\n--- Reconstruction Error Table ---")
    print(df.to_string(index=False))

    df.to_csv("reconstruction_results_report_style.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(df["Method"], df["MSE"], color='steelblue')
    plt.title("Reconstruction Error (MSE) by Method")
    plt.xlabel("Method")
    plt.ylabel("MSE")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("reconstruction_error_plot.png")
    plt.show()

def run_all():
    results = {}
    original, ff = load_ff_reconstructions()
    pca = run_pca(original)
    svd = run_svd(original)

    results["FF"] = mse(original, ff)
    results["PCA"] = mse(original, pca)
    results["SVD"] = mse(original, svd)

    for i in range(len(original)):
        save_images(original[i], ff[i], pca[i], svd[i], i)

    with open("results/mse_scores.csv", "w") as f:
        f.write("Method,MSE\n")
        for method, score in results.items():
            f.write(f"{method},{score:.6f}\n")

    print("Karşılaştırmalar tamamlandı. Görseller ve MSE değerleri 'results/' klasörüne kaydedildi.")
    show_results_table_and_plot(results)

if __name__ == "__main__":
    run_all()
