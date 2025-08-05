import time
from collections import defaultdict

import hydra
import torch
from omegaconf import DictConfig

from src import utils
import wandb

# Cihaz seçimi: CUDA varsa kullan, yoksa CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(opt, model, optimizer):
    start_time = time.time()
    train_loader = utils.get_data(opt, "train")
    num_steps_per_epoch = len(train_loader)
    
    best_loss = 1e14
    best_model = type(model)(opt).to(device)

    for epoch in range(opt.training.epochs):
        train_results = defaultdict(float)
        for inputs, labels in train_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels, device)

            optimizer.zero_grad()

            scalar_outputs = model(inputs)
            scalar_outputs["Loss"].backward()

            optimizer.step()

            train_results = utils.log_results(train_results, scalar_outputs, num_steps_per_epoch)

        utils.print_results("train", time.time() - start_time, train_results, epoch)
        start_time = time.time()

        # Validation
        best_loss, best_model = validate_or_test(opt, model, "val", epoch=epoch, loss=best_loss, best_model=best_model)

    print("saving model")
    utils.save_model(best_model)

    return best_model

def validate_or_test(opt, model, partition, epoch=None, loss=1e14, best_model=None):
    test_time = time.time()
    test_results = defaultdict(float)
    scalar_outputs = {}
    data_loader = utils.get_data(opt, partition)
    num_steps_per_epoch = len(data_loader)

    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels, device)

            scalar_outputs = model.clasify(inputs, 1 if partition == "test" else 0)
    
            test_results = utils.log_results(
                test_results, scalar_outputs, num_steps_per_epoch
            )

    utils.print_results(partition, time.time() - test_time, test_results, epoch=epoch)

    if partition == "val":
        if test_results["Loss"] < loss:
            loss = test_results["Loss"]
            best_model.load_state_dict(model.state_dict())

    model.train()
    return loss, best_model

@hydra.main(config_path=".", config_name="config", version_base=None)
def my_main(opt: DictConfig) -> None:
    opt = utils.parse_args(opt)
    run = wandb.init(
        entity="fathbt-tobb-et-",
        project="deneme",
        name=opt.name,
        reinit=False,
        config=dict(opt)
    )

    model, optimizer = utils.get_model_and_optimizer(opt, device)
    model = train(opt, model, optimizer)

    validate_or_test(opt, model, "train")
    validate_or_test(opt, model, "test")
    run.finish()

if __name__ == "__main__":
    my_main()
