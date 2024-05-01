from engine import trainer, val_step
from utils.utils import plot_results
from models import get_model

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import argparse

from torchvision.transforms.v2 import AutoAugmentPolicy, functional as F, InterpolationMode, Transform

from torchvision.transforms import v2
torchvision.disable_beta_transforms_warning()
import torchxrayvision as xrv
import matplotlib.pyplot as plt
import numpy as np
import random
from torch.utils.data.dataset import Subset

from utils.utils import EarlyStopper, get_dataset

import yaml
import json
import time
import os
import wandb
from dataset import RSNADataset, HAM10000Dataset,AptosDataset
from utils.utils import load_model


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/san_final_hyp_config_init.yaml', metavar='DIR', help='configs')

args = parser.parse_args()


config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
print(config)
run = wandb.init(entity='biomed', project='model_soups', config=config)


LEARNING_RATE = float(config["LEARNING_RATE"])
LEARNING_SCHEDULER = config["LEARNING_SCHEDULER"]
BATCH_SIZE = int(config["BATCH_SIZE"])
NUM_EPOCHS = int(config["NUM_EPOCHS"])
NUM_CLASSES = int(config["NUM_CLASSES"])
LINEAR_PROBING = config["LINEAR_PROBING"]
PROBING_EPOCHS = int(config["PROBING_EPOCHS"])
PATIENCE = int(config["PATIENCE"])

LOSS = config["LOSS"]

IMAGE_SIZE = int(config["IMAGE_SIZE"])
MODEL = config["MODEL"]
PRETRAINED = config["PRETRAINED"]

NUM_WORKERS = int(config["NUM_WORKERS"])


DATASET = config["DATASET"]
# RSNA_CSV = config["RSNA_CSV"]
# RSNA_PATH = config["RSNA_PATH"]
# CIFAR_PATH = config["CIFAR_PATH"]
# CIFAR_INDICES = config["CIFAR_INDICES"]

# HAM_TRAIN_CSV = str(config["HAM_TRAIN_CSV"])
# HAM_VAL_CSV = str(config["HAM_VAL_CSV"])
# HAM_TEST_CSV = str(config["HAM_TEST_CSV"])
# HAM_TRAIN_FOLDER = str(config["HAM_TRAIN_FOLDER"])
# HAM_VAL_FOLDER = str(config["HAM_VAL_FOLDER"])
# HAM_TEST_FOLDER = str(config["HAM_TEST_FOLDER"])


# APTOS_CSV = str(config["APTOS_CSV"])
# APTOS_FOLDER = str(config["APTOS_FOLDER"])
TASK = config["TASK"]
PATHS = config["PATH"]
CLASSIFICATION = config["CLASSIFICATION"]
PRETRAINING = config["PRETRAINING"]
SAVE_DIR = config["SAVE_DIR"]

CUDA_DEVICE = int(config["CUDA_DEVICE"])

DEVICE = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else 'cpu')

print(f"Using {DEVICE} device")

def START_seed(seed_value=9):
    seed = seed_value
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 


def main():
    START_seed()

    #run id is date and time of the run
    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    #create folder for this run in runs folder
    os.mkdir(SAVE_DIR + run_id)

    save_dir = SAVE_DIR + run_id
    
    train_loader, val_loader, test_loader = get_dataset(DATASET, PATHS,"Minimal", PRETRAINING, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, TASK)


    #load model
    model = get_model(MODEL, TASK, PRETRAINED, num_classes=NUM_CLASSES)

    model.to(DEVICE)
    torch.compile(model)
    
    #load optimizer
    if LOSS == "MSE":
        loss = torch.nn.MSELoss()
    elif LOSS == "L1Loss":
        loss = torch.nn.L1Loss()
    elif LOSS == "SmoothL1Loss":
        loss = torch.nn.SmoothL1Loss()
    elif LOSS == "CrossEntropyLoss":
        loss = torch.nn.CrossEntropyLoss()
    elif LOSS == "BCEWithLogitsLoss":
        loss = torch.nn.BCEWithLogitsLoss()
        
    else:
        raise Exception("Loss not implemented")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if LEARNING_SCHEDULER == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, verbose=True)
    elif LEARNING_SCHEDULER == "ReduceLROnPlateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    elif LEARNING_SCHEDULER == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)
    elif LEARNING_SCHEDULER == "MultiStepLR":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 20], gamma=0.1)
    else:
        lr_scheduler = None

    early_stopper = EarlyStopper(patience=PATIENCE, min_delta=0.001)

    if LINEAR_PROBING:
        linear_probing_epochs = PROBING_EPOCHS
    else:
        linear_probing_epochs = None
     
    #train model
    results = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        lr_scheduler_name=LEARNING_SCHEDULER,
        device=DEVICE,
        epochs=NUM_EPOCHS,
        save_dir=save_dir,
        early_stopper=early_stopper,
        linear_probing_epochs=linear_probing_epochs,
        dataset = DATASET
    )

    
    checkpoint = torch.load(save_dir + "/best_checkpoint.pth")
    model.load_state_dict(checkpoint['model'])
    model.to(DEVICE)
    torch.compile(model)

    test_loss, test_acc, test_f1, test_recall, test_kappa, test_auc = val_step(model, test_loader, train_loader=train_loader, loss_fn=loss, device = DEVICE, classification = CLASSIFICATION)
    print(test_loss, test_acc, test_f1, test_recall, test_kappa, test_auc)
    config["test_acc"] = test_acc
    config["test_loss"] = test_loss
    config["test_f1"] = test_f1
    config["test_recall"] = test_recall
    config["test_kappa"] = test_kappa
    config["test_auc"] = test_auc
    

    

    wandb.log({"test_loss": test_loss, "test_acc": test_acc})

    train_summary = {
        "config": config,
        "results": results,
    }


    with open(save_dir + "/train_summary.json", "w") as f:
        json.dump(train_summary, f, indent=4)

    plot_results(results, save_dir)

    



if __name__ == "__main__":
    main()


