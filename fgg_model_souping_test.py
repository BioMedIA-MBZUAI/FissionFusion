from engine import val_step
from utils.utils import souping, greedy_souping
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

from utils.utils import EarlyStopper, get_dataset
from torch.utils.data.dataset import Subset

import yaml
import json
import time
import os
import wandb
from utils.utils import load_model
import sys



parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/fgg_model_souping_test.yaml', metavar='DIR', help='configs')

args = parser.parse_args()


config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

DATASET = config["DATASET"]
CUDA_DEVICE = int(config["CUDA_DEVICE"])
NUM_WORKERS = int(config["NUM_WORKERS"])
PRETRAINING = str(config["PRETRAINING"])
WEIGHT_PATH_FGG = config["WEIGHT_PATH_FGG"]
BATCH_SIZE = int(config["BATCH_SIZE"])
NUM_CLASSES = int(config["NUM_CLASSES"])
IMAGE_SIZE = int(config["IMAGE_SIZE"])
CLASSIFICATION = str(config["CLASSIFICATION"])
val_sort_by = str(config["val_sort_by"])
test_sort_by = str(config["test_sort_by"])
TASK = str(config["TASK"])
PATHS = config["PATH"]
MODEL = config["MODEL"]
DEVICE = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else 'cpu')
print('>>>>>>>>>>>>>>>>>>>>>>>>>>', DATASET,'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
state_dicts = []
val_results = []
test_results = []

#if there is no test directory create it
if not os.path.exists("test"):
    os.makedirs("test")

#if there is no folder named after dataset inside test directory create it
if not os.path.exists(os.path.join("test", "fgg", DATASET, MODEL)):
    os.makedirs(os.path.join("test", "fgg", DATASET, MODEL))

test_save_path = os.path.join("test", "fgg", DATASET, MODEL)

# # Open a text file for logging
log_file_path = os.path.join("test", "fgg", DATASET, MODEL, "output_log.txt")
log_file = open(log_file_path, "w")

# Redirect the standard output to the log file
sys.stdout = log_file


train_loader, val_loader, test_loader = get_dataset(DATASET, PATHS, "Minimal", 
                                                        PRETRAINING, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, TASK)
LOSS = config["LOSS"]
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

results = {}
fgg_cifar_acc = []
fgg_cifar10_1_acc = []
state_dicts = []
val_results = []
test_results = []
value = 0

for idx, folder in enumerate(sorted(os.listdir(WEIGHT_PATH_FGG))):
    if folder.endswith('csv') or folder.endswith('txt'):
        continue
    lr_folder = os.path.join(WEIGHT_PATH_FGG, folder)
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{folder}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    for idx, model in enumerate(sorted(os.listdir(lr_folder))):

        if model.endswith('csv') or model.endswith('txt'):
            continue
        if model not in ['fgg-0.pt', 'fgg-1.pt', 'fgg-5.pt', 'fgg-9.pt', 'fgg-13.pt', 'fgg-17.pt']:
            continue
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{model}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        checkpoint = torch.load(os.path.join(lr_folder, model), map_location=DEVICE)
        state_dicts.append(checkpoint['model_state'])
        model = get_model(MODEL, num_classes=NUM_CLASSES)
        model.load_state_dict(checkpoint['model_state'])
        model = model.to(DEVICE)
    
        test_loss, test_acc, test_f1, test_recall, test_kappa, test_auc = val_step(model, test_loader, train_loader, loss, device = DEVICE, classification = CLASSIFICATION)
        val_loss, val_acc, val_f1, val_recall, val_kappa, val_auc = val_step(model, val_loader, train_loader, loss, device = DEVICE, classification = CLASSIFICATION)

        val_results.append({'Model Name': value,
                                        'Val Accuracy': val_acc,
                                        'Val F1': val_f1,
                                        'Val Recall': val_recall,
                                        'Val Kappa': val_kappa,
                                        'Val AUC': val_auc})
        test_results.append({'Model Name': value,
                                        'Test Accuracy': test_acc,
                                        'Test F1': test_f1,
                                        'Test Recall': test_recall,
                                        'Test Kappa': test_kappa,
                                        'Test AUC': test_auc})
        value+=1
    




results_test_df = pd.DataFrame(test_results)
results_val_df = pd.DataFrame(val_results)

val_copy = results_val_df.copy()
sorted_val = val_copy.sort_values(by= val_sort_by,ascending=False)
sorted_val.to_csv(os.path.join(test_save_path, "VAL_RESULTS.csv"), index=False)

test_copy = results_test_df.copy()
sorted_test = test_copy.sort_values(by= test_sort_by,ascending=False)
sorted_test.to_csv(os.path.join(test_save_path, "TEST_RESULTS.csv"), index=False)
print(f"Best model {val_sort_by}: {sorted_val.iloc[0][val_sort_by]}")
print(f"Second best model {val_sort_by}: {sorted_val.iloc[1][val_sort_by]}")
print(f"Worst model {val_sort_by}: {sorted_val.iloc[-1][val_sort_by]}")


print(f"Best model {test_sort_by}: {sorted_test.iloc[0][test_sort_by]}")
print(f"Second best model {test_sort_by}: {sorted_test.iloc[1][test_sort_by]}")
print(f"Worst model {test_sort_by}: {sorted_test.iloc[-1][test_sort_by]}")
#UNIFORM
print("Uniform souping ...")
alphal = [1 / len(state_dicts) for i in range(len(state_dicts))]
model = get_model(MODEL, num_classes=NUM_CLASSES)
uniform_model = souping(model, state_dicts, alphal)
uniform_model.to(DEVICE)

uniform_test_loss, uniform_test_acc, uniform_test_f1, uniform_test_recall, uniform_test_kappa, uniform_test_auc = val_step(uniform_model, test_loader, train_loader, loss, DEVICE, CLASSIFICATION)

#greedy

print("Greedy souping ...")
model = get_model(MODEL, num_classes=NUM_CLASSES)

val_results = list(results_val_df[val_sort_by])
val_models = list(results_val_df["Model Name"])
greedy_model, best_ingredients = greedy_souping(state_dicts, val_results, MODEL, NUM_CLASSES, val_loader,train_loader, loss, DEVICE , CLASSIFICATION, val_sort_by, val_models =val_models)
greedy_model.to(DEVICE)
print('VAL INGREDIENTS',best_ingredients)
greedy_test_loss, greedy_test_acc, greedy_test_f1, greedy_test_recall, greedy_test_kappa, greedy_test_auc = val_step(greedy_model, test_loader, train_loader, loss, DEVICE, CLASSIFICATION)
# test_results = list(results_test_df[test_sort_by])
# test_models = list(results_test_df["Model Name"])
# greedy_model_test, best_ingredients_test = greedy_souping(state_dicts, test_results, MODEL, NUM_CLASSES, test_loader, train_loader, loss, DEVICE, CLASSIFICATION, test_sort_by, test_models)
# print('TEST INGREDIENTS', best_ingredients_test)
# greedy_model_test.to(DEVICE)
# greedy_test_loss_test, greedy_test_acc_test, greedy_test_f1_test, greedy_test_recall_test, greedy_test_kappa_test, greedy_test_auc_test = val_step(greedy_model_test, test_loader, train_loader, loss, DEVICE, CLASSIFICATION)

#saving results in table
print("Creating table ...")
table = pd.DataFrame(columns=['Model Name', 'Test Accuracy', 'Test F1', 'Test Recall', 'Test Kappa','Test AUC','Augmentation', 'Learning Rate', 'SEED'])

#get best, second best and worst model from sorted_test and add them to the table
best_model = sorted_test.iloc[0]
second_best_model = sorted_test.iloc[1]
worst_model = sorted_test.iloc[-1]

#rename model name in best, second best and worst model
best_model['Model Name'] = f"Best 1: {best_model['Model Name']}"
second_best_model['Model Name'] = f"Best 2: {second_best_model['Model Name']}"
worst_model['Model Name'] = f"Worst: {worst_model['Model Name']}"

table.loc[0] = best_model
table.loc[1] = second_best_model
table.loc[2] = worst_model

#add uniform and greedy to the table
table.loc[3] = {'Model Name': 'Uniform',
                                    'Test Accuracy': uniform_test_acc,
                                    'Test F1': uniform_test_f1,
                                    'Test Recall': uniform_test_recall,
                                    'Test Kappa': uniform_test_kappa,
                                    'Test AUC': uniform_test_auc,
                                    'Augmentation': 'None',
                                    'Learning Rate': 'None',
                                    'SEED': 'None'}

table.loc[4] = {'Model Name': 'Greedy',
                                    'Test Accuracy': greedy_test_acc,
                                    'Test F1': greedy_test_f1,
                                    'Test Recall': greedy_test_recall,
                                    'Test Kappa': greedy_test_kappa,
                                    'Test AUC': greedy_test_auc,
                                    'Augmentation': 'None',
                                    'Learning Rate': 'None',
                                    'SEED': 'None'}



#save the table to csv without index
table.to_csv(os.path.join(test_save_path, "ALL.csv"), index=False)
print(table.to_markdown())
print("Table saved to ALL.csv")