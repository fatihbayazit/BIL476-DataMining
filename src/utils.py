import os
import random
from datetime import timedelta

import numpy as np
import torch
import torchvision
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from torchvision.transforms import Compose, ToTensor
from torch.utils.data import Subset

from src import datasets, models
import wandb

def parse_args(opt):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    print(OmegaConf.to_yaml(opt))

    return opt

def get_input_layer_size(opt):
    if opt.input.dataset in ["mnist", "fmnist", "kmnist"]:
        return 784
    else:
        raise ValueError("Unknown dataset.")

def get_model_and_optimizer(opt, device):
    model = models.Model(opt)
    model = model.to(device)
    print(model, "\n")

    optimizer = torch.optim.SGD(
        [
            {
                "params": model.parameters(),
                "lr": opt.training.learning_rate,
                "weight_decay": opt.training.weight_decay,
                "momentum": opt.training.momentum,
            }
        ]
    )

    return model, optimizer

def get_data(opt, partition):
    dataset = datasets.Dataset(opt, partition, num_classes=10)

    g = torch.Generator()
    g.manual_seed(opt.seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.input.batch_size,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=0,  # güvenli varsayılan değer (istersen 8 yapabilirsin)
        pin_memory=True,
        persistent_workers=False,
    )

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_MNIST_partition(opt, partition):
    if opt.input.dataset == "mnist":
        load_dataset = torchvision.datasets.MNIST
    elif opt.input.dataset == "fmnist":
        load_dataset = torchvision.datasets.FashionMNIST
    elif opt.input.dataset == "kmnist":
        load_dataset = torchvision.datasets.KMNIST
    else:
        raise NotImplementedError

    transform = Compose([ToTensor()])
    
    if partition in ["train", "val"]:
        mnist = load_dataset(
            os.path.join(get_original_cwd(), "datasets"),
            train=True,
            download=True,
            transform=transform,
        )
        if partition == "train":
            mnist = Subset(mnist, range(50000))
        elif partition == "val":
            mnist = Subset(mnist, range(50000, 60000))
    elif partition == "test":
        mnist = load_dataset(
            os.path.join(get_original_cwd(), "datasets"),
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise NotImplementedError
    return mnist

# 🟢 DÜZENLENDİ: .cuda yerine to(device)
def dict_to_device(tensor_dict, device):
    for key, value in tensor_dict.items():
        tensor_dict[key] = value.to(device, non_blocking=True)
    return tensor_dict

# 🟢 DÜZENLENDİ: device parametresi eklendi
def preprocess_inputs(opt, inputs, labels, device):
    inputs = dict_to_device(inputs, device)
    labels = dict_to_device(labels, device)
    return inputs, labels

def print_results(partition, iteration_time, scalar_outputs, epoch=None):
    if epoch is not None:
        print(f"Epoch {epoch} \t", end="")

    print(
        f"{partition} \t \t"
        f"Time: {timedelta(seconds=iteration_time)} \t",
        end="",
    )
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            print(f"{key}: {value:.4f} \t", end="")
    print()
    partition_scalar_outputs = {}
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            partition_scalar_outputs[f"{partition}_{key}"] = value
    wandb.log(partition_scalar_outputs, step=epoch)

def save_model(model):
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(model.state_dict(), "models" + os.path.sep + f"{wandb.run.name}-model.pt")
    wandb.save(f"{wandb.run.name}-model.pt")

def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if isinstance(value, float):
            result_dict[key] += value / num_steps
        else:
            result_dict[key] += value.item() / num_steps
    return result_dict
