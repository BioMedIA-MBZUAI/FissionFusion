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
parser.add_argument('--config', type=str, default='./configs/hierarchical_souping.yaml', metavar='DIR', help='configs')

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
if not os.path.exists(os.path.join("test", "hs", DATASET, MODEL)):
    os.makedirs(os.path.join("test", "hs", DATASET, MODEL))

test_save_path = os.path.join("test", "hs", DATASET, MODEL)

# # Open a text file for logging
log_file_path = os.path.join("test", "hs", DATASET, MODEL, "output_log.txt")
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

greedy_models = []
uniform_models = []

greedy_model_val_results = []
greedy_model_test_results = []

uniform_model_val_results = []
uniform_model_test_results = []

uniform_model_value = 0
greedy_model_value = 0


for idx, folder in enumerate(sorted(os.listdir(WEIGHT_PATH_FGG))):
    if folder.endswith('csv') or folder.endswith('txt'):
        continue
    lr_folder = os.path.join(WEIGHT_PATH_FGG, folder)
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{folder}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    results = {}
    state_dicts = []
    val_results = []
    test_results = []
    value = 0

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

    test_copy = results_test_df.copy()
    sorted_test = test_copy.sort_values(by= test_sort_by,ascending=False)


    print("Uniform souping ...")
    alphal = [1 / len(state_dicts) for i in range(len(state_dicts))]
    model = get_model(MODEL, num_classes=NUM_CLASSES)
    uniform_model = souping(model, state_dicts, alphal)
    uniform_model.to(DEVICE)

    uniform_val_loss, uniform_val_acc, uniform_val_f1, uniform_val_recall, uniform_val_kappa, uniform_val_auc = val_step(uniform_model, val_loader, train_loader, loss, DEVICE, CLASSIFICATION)
    uniform_test_loss, uniform_test_acc, uniform_test_f1, uniform_test_recall, uniform_test_kappa, uniform_test_auc = val_step(uniform_model, test_loader, train_loader, loss, DEVICE, CLASSIFICATION)

    uniform_model_val_results.append({'Model Name': 'Uniform_' + str(uniform_model_value),
                                    'Val Accuracy': uniform_val_acc,
                                    'Val F1': uniform_val_f1,
                                    'Val Recall': uniform_val_recall,
                                    'Val Kappa': uniform_val_kappa,
                                    'Val AUC': uniform_val_auc})



    uniform_model_test_results.append({'Model Name': 'Uniform_' + str(uniform_model_value),
                                    'Test Accuracy': uniform_test_acc,
                                    'Test F1': uniform_test_f1,
                                    'Test Recall': uniform_test_recall,
                                    'Test Kappa': uniform_test_kappa,
                                    'Test AUC': uniform_test_auc})




    #greedy
    print("Greedy souping ...")
    model = get_model(MODEL, num_classes=NUM_CLASSES)

    val_results = list(results_val_df[val_sort_by])
    val_models = list(results_val_df["Model Name"])
    greedy_model, best_ingredients = greedy_souping(state_dicts, val_results, MODEL, NUM_CLASSES, val_loader,train_loader, loss, DEVICE , CLASSIFICATION, val_sort_by, val_models = val_models)
    greedy_model.to(DEVICE)
    print('VAL INGREDIENTS',best_ingredients)

    greedy_val_loss, greedy_val_acc, greedy_val_f1, greedy_val_recall, greedy_val_kappa, greedy_val_auc = val_step(greedy_model, val_loader, train_loader, loss, DEVICE, CLASSIFICATION)
    greedy_test_loss, greedy_test_acc, greedy_test_f1, greedy_test_recall, greedy_test_kappa, greedy_test_auc = val_step(greedy_model, test_loader, train_loader, loss, DEVICE, CLASSIFICATION)

    greedy_model_val_results.append({'Model Name': 'Greedy_'+ str(greedy_model_value),
                                    'Val Accuracy': greedy_val_acc,
                                    'Val F1': greedy_val_f1,
                                    'Val Recall': greedy_val_recall,
                                    'Val Kappa': greedy_val_kappa,
                                    'Val AUC': greedy_val_auc})


    greedy_model_test_results.append({'Model Name': 'Greedy_'+ str(greedy_model_value),
                                    'Test Accuracy': greedy_test_acc,
                                    'Test F1': greedy_test_f1,
                                    'Test Recall': greedy_test_recall,
                                    'Test Kappa': greedy_test_kappa,
                                    'Test AUC': greedy_test_auc})




    print(f'>>>>>>>>>>>>>>>>>>>>>>{lr_folder}<<<<<<<<<<<<<<<<<<<<<<<')
    print(f'Uniform Acc:{uniform_test_acc}, Uniform F1:{uniform_test_f1}, Uniform Recall:{uniform_test_recall},  Uniform AUC:{uniform_test_auc}')
    print(f'Greedy Acc:{greedy_test_acc}, Greedy F1:{greedy_test_f1}, Greedy Recall:{greedy_test_recall},  Greedy AUC:{greedy_test_auc}')

    greedy_models.append(greedy_model.state_dict())
    uniform_models.append(uniform_model.state_dict())

    uniform_model_value +=1
    greedy_model_value+=1

    
uniform_val_df = pd.DataFrame(uniform_model_val_results)
greedy_val_df = pd.DataFrame(greedy_model_val_results)
final_val_df = pd.concat([uniform_val_df, greedy_val_df])

uniform_test_df = pd.DataFrame(uniform_model_test_results)
greedy_test_df = pd.DataFrame(greedy_model_test_results)
final_test_df = pd.concat([uniform_test_df, greedy_test_df])


#GREEDY OF UNIFORM (GoU)
u_results = list(uniform_val_df[val_sort_by])
u_models = list(uniform_val_df["Model Name"])
greedy_model, best_ingredients = greedy_souping(uniform_models, u_results, MODEL, NUM_CLASSES, val_loader,train_loader, loss, DEVICE , CLASSIFICATION, val_sort_by, val_models = u_models)
greedy_model.to(DEVICE)
print('VAL INGREDIENTS',best_ingredients)
greedy_test_loss, greedy_test_acc, greedy_test_f1, greedy_test_recall, greedy_test_kappa, greedy_test_auc = val_step(greedy_model, test_loader, train_loader, loss, DEVICE, CLASSIFICATION)

final_test_df =  final_test_df._append({'Model Name': 'GoU', 'Test Accuracy': greedy_test_acc, 'Test F1': greedy_test_f1, 'Test Recall': greedy_test_recall, 'Test Kappa': greedy_test_kappa,'Test AUC': greedy_test_auc}, ignore_index=True)

# GREEDY OF GREEDY (GoG)

g_results = list(greedy_val_df[val_sort_by])
g_models = list(greedy_val_df["Model Name"])
greedy_model, best_ingredients = greedy_souping(greedy_models, g_results, MODEL, NUM_CLASSES, val_loader,train_loader, loss, DEVICE , CLASSIFICATION, val_sort_by, val_models = g_models)
greedy_model.to(DEVICE)
print('VAL INGREDIENTS',best_ingredients)
greedy_test_loss, greedy_test_acc, greedy_test_f1, greedy_test_recall, greedy_test_kappa, greedy_test_auc = val_step(greedy_model, test_loader, train_loader, loss, DEVICE, CLASSIFICATION)

final_test_df = final_test_df._append({'Model Name': 'GoG', 'Test Accuracy': greedy_test_acc, 'Test F1': greedy_test_f1, 'Test Recall': greedy_test_recall, 'Test Kappa': greedy_test_kappa,'Test AUC': greedy_test_auc}, ignore_index=True)

final_test_df.to_csv(os.path.join(test_save_path, "HS_results.csv"), index=False)