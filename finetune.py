from engine import trainer, val_step
from utils.utils import plot_results
from models import get_model

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision

from dataset import RSNADataset, HAM10000Dataset, AptosDataset
from torchvision.transforms.v2 import AutoAugmentPolicy, functional as F, InterpolationMode, Transform
from torchvision.transforms import v2
torchvision.disable_beta_transforms_warning()
from timm.data.transforms import RandomResizedCropAndInterpolation
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import argparse

from utils.utils import EarlyStopper
from torch.utils.data.dataset import Subset
from utils.utils import EarlyStopper, get_dataset

import yaml
import json
import time
import os
import wandb
from utils.utils import load_model


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/san_config_finetune.yaml', metavar='DIR', help='configs')
args = parser.parse_args()

config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
print(config)
run = wandb.init(entity='biomed', project='model_soups', config=config)


wandconf = {
        "LEARNING_SCHEDULER": config["LEARNING_SCHEDULER"],
        "BATCH_SIZE": config["BATCH_SIZE"],
        "IMAGE_SIZE": config["IMAGE_SIZE"],
        "MODEL": config["MODEL"],
        "PRETRAINED": config["PRETRAINED"],
        "LOSS": config["LOSS"],
        "SAVE_DIR": config["SAVE_DIR"],
        "RUN_NAME": config["RUN_NAME"],
        "RESUME_PATH": config["RESUME_PATH"]

}

LEARNING_RATE = float(config["LEARNING_RATE"])
LEARNING_SCHEDULER = config["LEARNING_SCHEDULER"]
BATCH_SIZE = int(config["BATCH_SIZE"])
NUM_EPOCHS = int(config["NUM_EPOCHS"])
NUM_CLASSES = int(config["NUM_CLASSES"])
LINEAR_PROBING = config["LINEAR_PROBING"]
# PROBING_EPOCHS = int(config["PROBING_EPOCHS"])
PATIENCE = int(config["PATIENCE"])
SAVE_DIR = str(config["SAVE_DIR"])

LR_RATE_LIST = config["LR_RATE_LIST"]
SEED_LIST = config['SEED']
CLASSIFICATION = config["CLASSIFICATION"]
# NUM_EPOCHS_MINIMAL = config["NUM_EPOCHS_MINIMAL"]
# NUM_EPOCHS_MEDIUM = config["NUM_EPOCHS_MEDIUM"]
# NUM_EPOCHS_HEAVY = config["NUM_EPOCHS_HEAVY"]

PATHS = config["PATH"]
AUGMENT_LIST = config["AUGMENT_LIST"]

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

LOSS = config["LOSS"]

IMAGE_SIZE = int(config["IMAGE_SIZE"])
MODEL = config["MODEL"]
PRETRAINED = config["PRETRAINED"]
CUDA_DEVICE = int(config["CUDA_DEVICE"])
NUM_WORKERS = int(config["NUM_WORKERS"])
PRETRAINING = str(config["PRETRAINING"])
TASK = config["TASK"]

DEVICE = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else 'cpu')

RUN_NAME = str(config["RUN_NAME"])
RESUME_PATH = str(config["RESUME_PATH"])

print(f"Using {DEVICE} device")


def START_seed(start_seed=9):
    seed = start_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 


def main():
    global RESUME_PATH
    START_seed()
    if RESUME_PATH == "":
        run_id = time.strftime("%Y-%m-%d_%H-%M-%S")

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        #create folder for this run in runs folder

        os.mkdir(SAVE_DIR + run_id)

        parent_dir = SAVE_DIR + run_id
    else:
        parent_dir = RESUME_PATH
    run_path = RUN_NAME
    
    
    
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
    

    #create pandas dataframe to store results
    resultsexp = pd.DataFrame(columns=["lr_rate", "seed", "augmentation", "test_acc", "test_loss", "auc"])
    
    hyp_ls = []
    for num, lr_rate in enumerate(LR_RATE_LIST):
        for augment in AUGMENT_LIST:
            for seed_val in SEED_LIST:
                comb = [lr_rate, augment, seed_val]
                hyp_ls.append(comb)

    if RESUME_PATH!="":
        idx = len(os.listdir(RESUME_PATH))-1
        dir_ls = [os.path.join(RESUME_PATH, d) for d in os.listdir(RESUME_PATH)]
        latest_dir = max(dir_ls, key = os.path.getmtime)
        latest_checkpoint = torch.load(os.path.join(latest_dir, 'last_checkpoint.pth'), map_location=DEVICE)
        hyp_ls = hyp_ls[idx:]


    print(hyp_ls)
    print('################################ LEN OF HY_LS #####################', len(hyp_ls))
    for idx, hyperparameters in enumerate(hyp_ls):
        lr_rate = hyperparameters[0]
        augment = hyperparameters[1]
        seed_val = hyperparameters[2]
        
        START_seed(start_seed = seed_val)
        lr_rate = float(lr_rate)
        num_epoch = NUM_EPOCHS

        wandconf["LEARNING_RATE"] = "{:.2e}".format(lr_rate)
        wandconf["NUM_EPOCHS"] = num_epoch
        wandconf["AUGMENTATION"] = augment
        wandconf["SEED"] = seed_val
        print(lr_rate, num_epoch, augment, seed_val)                
        

        model = get_model(MODEL, TASK, PRETRAINED, num_classes=NUM_CLASSES)
        checkpoint = torch.load(run_path + "best_checkpoint.pth", map_location=DEVICE)
        model.load_state_dict(checkpoint['model'])
        model.to(DEVICE)
        torch.compile(model)

        #run id is date and time of the run
        if RESUME_PATH=="":
            run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
            os.mkdir(parent_dir + "/hparam_" + run_id)
            save_dir = parent_dir + "/hparam_" + run_id
        else:
            save_dir = latest_dir

        train_loader, val_loader, test_loader = get_dataset(DATASET, PATHS,augment, PRETRAINING, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, TASK)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_rate)

        if LEARNING_SCHEDULER == "CosineAnnealingLR":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, verbose=True)
        elif LEARNING_SCHEDULER == "CyclicLR":
            lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = LEARNING_RATE, max_lr = LEARNING_RATE * 0.01, cycle_momentum=False)

        if RESUME_PATH !="":
            RESUME_PATH = ""
            print(latest_checkpoint['epoch'])
            if latest_checkpoint['epoch']!=50:
                start_epoch=0
            else:
                continue
        else:
            start_epoch=0

        print(start_epoch)
        early_stopper = EarlyStopper(patience=PATIENCE, min_delta=0.001)

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
            epochs=num_epoch,
            save_dir=save_dir,
            start_epoch = start_epoch,
            dataset = DATASET,
            early_stopper = early_stopper
        )

        checkpoint = torch.load(save_dir + "/best_checkpoint.pth")
        model.load_state_dict(checkpoint['model'])
        model.to(DEVICE)
        torch.compile(model)

        test_loss, test_acc, test_f1, test_recall, test_kappa, test_auc = val_step(model, test_loader, train_loader=train_loader, loss_fn=loss, device = DEVICE, classification = CLASSIFICATION)
        print(test_loss, test_acc, test_f1, test_recall, test_kappa, test_auc)
        wandb.log({"test_loss": test_loss, "test_acc": test_acc})

        config["test_acc"] = test_acc
        config["test_loss"] = test_loss
        config["test_f1"] = test_f1
        config["test_recall"] = test_recall
        config["test_auc"] = test_auc

        train_summary = {
            "config": wandconf,
            "results": results,
        }

        with open(save_dir + "/train_summary.json", "w") as f:
            json.dump(train_summary, f, indent=4)

        plot_results(results, save_dir)

        #append to dataframe
        resultsexp.loc[len(resultsexp)] = [wandconf["LEARNING_RATE"], wandconf["SEED"], wandconf["AUGMENTATION"], test_acc, test_loss, test_auc]        

    resultsexp.to_csv(parent_dir + "/testresults.csv", index=True)

if __name__ == "__main__":
    main()


