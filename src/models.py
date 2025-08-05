import math
import torch
import torch.nn as nn

from src import utils
import numpy as np
import cv2 as cv
import os

class Model(torch.nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.act_fn = nn.ReLU()

        input_layer_size = utils.get_input_layer_size(opt)

        self.model = nn.Sequential(nn.Linear(input_layer_size, self.opt.model.hidden_dim))
        self.ff_loss = nn.BCEWithLogitsLoss()
        self.linear_classifier = nn.Sequential(nn.Linear(self.opt.model.hidden_dim, input_layer_size))
        self.classification_loss = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()

        self._init_weights()
        self.chose_functions()

    def chose_functions(self):
        if self.opt.training.type == "backpropagation":
            self.forward = self.forward_backpropagation

    def _init_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=1 / math.sqrt(m.weight.shape[0]))
                torch.nn.init.zeros_(m.bias)

        for m in self.linear_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def _layer_norm(self, z, eps=1e-8):
        return z / (torch.sqrt(torch.mean(z ** 2, dim=-1, keepdim=True)) + eps)

    def _calc_ff_loss(self, z, labels):
        sum_of_squares = torch.sum(z ** 2, dim=-1)
        logits = sum_of_squares - z.shape[1]
        ff_loss = self.ff_loss(logits, labels.float())

        with torch.no_grad():
            ff_accuracy = (torch.sum((logits > 0.0) == labels) / z.shape[0]).item()
        return ff_loss, ff_accuracy

    def _calc_loss(self, z, posneg_labels, scalar_outputs):
        ff_loss, ff_accuracy = self._calc_ff_loss(z, posneg_labels)
        scalar_outputs["loss_layer"] += ff_loss
        scalar_outputs["ff_accuracy_layer"] += ff_accuracy
        scalar_outputs["Loss"] += ff_loss

    def forward(self, inputs):
        scalar_outputs = {
            "Loss": torch.zeros(1, device=self.device),
            "loss_layer": torch.zeros(1, device=self.device),
            "ff_accuracy_layer": torch.zeros(1, device=self.device),
        }

        z = torch.cat([inputs["pos_images"], inputs["neg_images"]], dim=0)
        posneg_labels = torch.zeros(z.shape[0], device=self.device)
        posneg_labels[:self.opt.input.batch_size] = 1

        z = z.reshape(z.shape[0], -1)
        z = self.model(z)
        z = self.act_fn(z)
        self._calc_loss(z, posneg_labels, scalar_outputs)
        z = z.detach()

        output = self.linear_classifier(z[:self.opt.input.batch_size])
        output = self.sigmoid(output)
        scalar_outputs["Loss"] += self.classification_loss(output, inputs["pos_images"].reshape(self.opt.input.batch_size, -1))
        return scalar_outputs

    def forward_downstream_classification_model(self, inputs, d=0, scalar_outputs=None):
        if scalar_outputs is None:
            scalar_outputs = {"Loss": torch.zeros(1, device=self.device)}

        z_input = inputs["pos_images"].reshape(inputs["pos_images"].shape[0], -1)
        z = self.model(z_input)
        z = self.act_fn(z)

        output = self.linear_classifier(z.detach())
        output = self.sigmoid(output)
        scalar_outputs["Loss"] += self.classification_loss(output, z_input)

        if d == 1:
            self.draw(output)
        return scalar_outputs

    def forward_backpropagation(self, inputs, d=0, scalar_outputs=None):
        if scalar_outputs is None:
            scalar_outputs = {"Loss": torch.zeros(1, device=self.device)}

        z = inputs["pos_images"].reshape(inputs["pos_images"].shape[0], -1)
        z = self.model(z)
        z = self.act_fn(z)

        output = self.linear_classifier(z)
        output = self.sigmoid(output)
        scalar_outputs["Loss"] += self.classification_loss(output, inputs["pos_images"].reshape(z.shape[0], -1))

        if d == 1:
            self.draw(output)
        return scalar_outputs

    def clasify(self, inputs, d, scalar_outputs=None):
        if self.opt.training.type == "backpropagation":
            return self.forward_backpropagation(inputs, d, scalar_outputs)
        else:
            return self.forward_downstream_classification_model(inputs, d, scalar_outputs)

    def draw(self, x):
        save_path = os.path.join(self.opt.input.dataset, self.opt.training.type)
        os.makedirs(save_path, exist_ok=True)
        x = x.reshape(-1, 28, 28)
        for a in range(min(100, x.shape[0])):
            cur = (np.array(x[a].detach().cpu()) * 255).astype(np.uint8)
            cv.imwrite(os.path.join(save_path, f"{a}.bmp"), cur)
