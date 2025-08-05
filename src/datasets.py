import numpy as np
import torch
import cv2 as cv
import random
from src import utils

class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=10):
        self.opt = opt
        self.dataset = utils.get_MNIST_partition(opt, partition)
        self.num_classes = num_classes
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes

    def __getitem__(self, index):
        if self.opt.training.type == "backpropagation":
            return self.__getitem__bp(index)
        elif self.opt.training.type == "supervised":
            return self.__getitem__sup(index)
        elif self.opt.training.type == "unsupervised":
            return self.__getitem__unsup(index)
        else:
            raise ValueError(f"Unknown training type: {self.opt.training.type}")

    def __getitem__bp(self, index):
        sample, class_label = self.dataset[index]
        return {"pos_images": sample}, {"class_labels": class_label}

    def __getitem__sup(self, index):
        sample, class_label = self.dataset[index]

        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample_sup(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample.clone())
        all_sample = self._get_all_sample(sample)

        return {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
            "all_sample": all_sample
        }, {"class_labels": class_label}

    def __getitem__unsup(self, index):
        sample, class_label = self.dataset[index]
        neg_sample = self._get_neg_sample_unsup(sample, class_label)

        return {
            "pos_images": sample,
            "neg_images": neg_sample,
        }, {"class_labels": class_label}

    def __len__(self):
        return len(self.dataset)

    def _get_pos_sample(self, sample, class_label):
        pos_sample = sample.clone()
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        ).float()
        pos_sample[0, 0, : self.num_classes] = one_hot_label
        return pos_sample

    def _get_neg_sample_sup(self, sample, class_label):
        neg_sample = sample.clone()
        classes = list(range(self.num_classes))
        classes.remove(class_label)
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        ).float()
        neg_sample[0, 0, : self.num_classes] = one_hot_label
        return neg_sample

    def _get_neg_sample_unsup(self, sample, class_label):
        mask, inverse_mask = self.create_mask()
        mask = torch.tensor(mask, dtype=torch.float32)
        inverse_mask = torch.tensor(inverse_mask, dtype=torch.float32)

        neg_sample = sample.clone()
        while True:
            data, label = self.dataset[random.randint(0, len(self.dataset) - 1)]
            if label != class_label:
                neg_sample = neg_sample * mask + data * inverse_mask
                break
        return neg_sample.float()

    def _get_neutral_sample(self, z):
        z[0, 0, : self.num_classes] = self.uniform_label
        return z

    def _get_all_sample(self, sample):
        all_samples = torch.zeros((self.num_classes,) + sample.shape)
        for i in range(self.num_classes):
            one_hot_label = torch.nn.functional.one_hot(
                torch.tensor(i), num_classes=self.num_classes
            ).float()
            all_samples[i] = sample.clone()
            all_samples[i, 0, 0, : self.num_classes] = one_hot_label
        return all_samples

    def create_mask(self):
        if self.opt.input.dataset in ["mnist", "fmnist", "kmnist"]:
            size = (28, 28)
        else:
            raise Exception("Unknown dataset")

        random_bits = np.random.randint(2, size=size, dtype=np.uint8)
        random_image = random_bits * 255

        kernel = np.array([[0.0625, 0.125, 0.0625],
                           [0.125,  0.25,  0.125],
                           [0.0625, 0.125, 0.0625]])

        for _ in range(10):
            random_image = cv.filter2D(random_image, -1, kernel)

        binary_mask = (random_image >= 127.5).astype(np.float32)
        inverse_mask = (random_image < 127.5).astype(np.float32)

        return binary_mask, inverse_mask
