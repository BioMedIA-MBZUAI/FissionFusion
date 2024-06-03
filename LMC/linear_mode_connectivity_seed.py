


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
import copy
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/linear_mode_connectivity.yaml', metavar='DIR', help='configs')
args = parser.parse_args()

print(args.config)
config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

DATASET = config["DATASET"]
CUDA_DEVICE = int(config["CUDA_DEVICE"])
NUM_WORKERS = int(config["NUM_WORKERS"])
PRETRAINING = str(config["PRETRAINING"])
WEIGHT_PATH = config["WEIGHT_PATH"]
BATCH_SIZE = int(config["BATCH_SIZE"])
NUM_CLASSES = int(config["NUM_CLASSES"])
IMAGE_SIZE = int(config["IMAGE_SIZE"])
CLASSIFICATION = str(config["CLASSIFICATION"])
val_sort_by = str(config["val_sort_by"])
test_sort_by = str(config["test_sort_by"])
TASK = str(config["TASK"])
PATHS = config["PATH"]
MODEL = config["MODEL"]
LOSS = config["LOSS"]
DEVICE = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else 'cpu')
print('>>>>>>>>>>>>>>>>>>>>>>>>>>', DATASET,'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
state_dicts = []
val_results = []
test_results = []


train_loader, val_loader, test_loader = get_dataset(DATASET, PATHS, "Minimal", PRETRAINING, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, TASK)

if not os.path.exists(f'./plots_v2/linear_connectivity/{DATASET}/seed_wrt_aug/'):
    os.makedirs(f'./plots_v2/linear_connectivity/{DATASET}/seed_wrt_aug/')


if not os.path.exists(f'./plots_v2/linear_connectivity/{DATASET}/seed_wrt_lr/'):
    os.makedirs(f'./plots_v2/linear_connectivity/{DATASET}/seed_wrt_lr/')
df = pd.read_csv(f'/home/santoshsanjeev/model_soups/noodles/test/{DATASET}/DeiT-B/VAL_RESULTS.csv')

def get_xy(point, origin, vector_x, vector_y):
    return np.array([np.dot(point - origin, vector_x), np.dot(point - origin, vector_y)])

#for each folder inside WEIGHT_PATH
for idx, folder in enumerate(sorted(os.listdir(WEIGHT_PATH))):
    print(idx)
    if folder.startswith('testresults'):
        continue
    model_path = os.path.join(WEIGHT_PATH, folder)
    #read config in train_summary.json
    train_summary = json.load(open(os.path.join(model_path, "train_summary.json"), 'r'))
    model_config = train_summary["config"]
    
    #load model
    checkpoint = torch.load(os.path.join(model_path, "best_checkpoint.pth"), map_location=DEVICE)
    state_dicts.append(checkpoint['model'])

    LOSS = model_config["LOSS"]
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
    




vary_seed = {}
for ind, row in df.iterrows():
    # if str(row['Learning Rate'])!='1e-06':
    row['Learning Rate'] = str(row['Learning Rate'])
    row['Augmentation'] = str(row['Augmentation'])
    if row['Learning Rate'] in vary_seed.keys():
        if row['Augmentation'] in vary_seed[row['Learning Rate']].keys():
            vary_seed[row['Learning Rate']][row['Augmentation']].append(int(row['Model Name']))
        else:
            vary_seed[row['Learning Rate']][row['Augmentation']] = [int(row['Model Name'])]
    else:
        vary_seed[row['Learning Rate']] = {}
        vary_seed[row['Learning Rate']][row['Augmentation']] = [int(row['Model Name'])]
    # val_accuracy_list[int(row['Model Name'])] = row['Val Accuracy']



print(vary_seed)
# take the vary_seed and create vary_seed_wrt_lr and vary_seed_wrt_aug
vary_seed_wrt_lr = {}
for key, value in vary_seed.items():
    vary_seed_wrt_lr[key] = list(value.values())
print(vary_seed_wrt_lr)

# vary_seed_wrt_aug = {}
# for key, value in vary_seed.items():
#     for inner_key in value.keys():
#         if inner_key in vary_seed_wrt_aug.keys():
#             vary_seed_wrt_aug[inner_key].append(vary_seed[key][inner_key])
#         else:
#             vary_seed_wrt_aug[inner_key] = [vary_seed[key][inner_key]]
# print(vary_seed_wrt_aug)



# vary_seed_list = []
# for value_dict in vary_seed.values():
#     for inner_list in value_dict.values():
#         vary_seed_list.append(inner_list)
# print(vary_seed_list)
# # initial_model_path = '/home/santosh.sanjeev/Projects/model-soups/noodles/runs/san_final_hyp_models/san-initial_models/aptos_final_hyp/deitB_imagenet/2024-01-10_02-40-01/best_checkpoint.pth'





# iterate through all the vary_seed_wrt_lr and then take each pair of models and interpolate them and for each LR get the plot of interploation

# lambda_values = [i / 8 for i in range(9)]
# print(lambda_values)

# colors = {'Heavy': 'red', 'Medium': 'blue', 'Minimal': 'green'}

# # Iterate through vary_seed
# for lr, augmentation_dict in vary_seed.items():
#     plt.figure(figsize=(10, 6))
#     model_interpolations = []
#     for augmentation, combination_sets in augmentation_dict.items():
#         # for combination_sets in combination_sets:
#         print(combination_sets)
#         val_losses = []
#         val_accs = []
#         val_f1s = []
#         val_recalls = []
#         val_kappas = []
#         val_aucs = []

#         for l in lambda_values:
#             if l == 0:
#                 l = 1e-8
#             if l == 1:
#                 l = 1 - 1e-8
#             alphal = [l, 1 - l]
#             print(alphal)
#             final_model = get_model(MODEL, num_classes=NUM_CLASSES)
#             final_model = souping(final_model, [state_dicts[combination_sets[0]], state_dicts[combination_sets[1]]], alphal)
#             final_model = final_model.to(DEVICE)
#             val_loss, val_acc, val_f1, val_recall, val_kappa, val_auc = val_step(final_model, val_loader, train_loader, loss, DEVICE, CLASSIFICATION)
#             val_f1s.append(val_f1)
#         model_interpolations.append((val_f1s.copy(), augmentation))
#         print(model_interpolations)
        
#     # Plotting
#     for model_interp, aug in model_interpolations:
#         plt.plot(lambda_values, model_interp, marker='o', label=f'{aug} augmentation', color=colors[aug])

#     plt.xlabel('Lambda')
#     plt.ylabel('Validation Accuracy')
#     plt.title(f'Performance of Final Model with Interpolated Weights (LR: {lr})')
#     plt.legend()
#     plt.grid(True)

#     # Save the figure
#     plt.savefig(f'./plots_v2/linear_connectivity/{DATASET}/seed_wrt_lr/{lr}_performance_plot.png')

#     # Display the figure


# Define gray background color
# plt.rcParams['axes.facecolor'] = 'lightgray'
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Set font to Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'

vary_seed_wrt_aug = {}

for lr, augmentation_dict in vary_seed.items():
    for augmentation, combination_sets in augmentation_dict.items():
        if augmentation not in vary_seed_wrt_aug:
            vary_seed_wrt_aug[augmentation] = {}
        vary_seed_wrt_aug[augmentation][lr] = combination_sets

lambda_values = [i / 8 for i in range(9)]

all_lrs = set(lr for aug_lr_dict in vary_seed_wrt_aug.values() for lr in aug_lr_dict.keys())
colors = {'1e-07': 'red', '1e-06': 'blue', '5e-06': 'green', '1e-05': 'yellow', '5e-05': 'brown', '0.0001': 'purple', '0.0005':'orange', '0.001':'lightseagreen'}

# Iterate through vary_seed
for augmentation, lr_dict in vary_seed_wrt_aug.items():
    plt.figure(figsize=(10, 6))
    model_interpolations = []
    for lr, combination_sets in lr_dict.items():
        val_f1s = []
        for l in lambda_values:
            if l == 0:
                l = 1e-8
            if l == 1:
                l = 1 - 1e-8
            alphal = [l, 1 - l]
            final_model = get_model(MODEL, num_classes=NUM_CLASSES)
            final_model = souping(final_model, [state_dicts[combination_sets[0]], state_dicts[combination_sets[1]]], alphal)
            final_model = final_model.to(DEVICE)
            val_loss, val_acc, val_f1, val_recall, val_kappa, val_auc = val_step(final_model, val_loader, train_loader, loss, DEVICE, CLASSIFICATION)
            val_f1s.append(val_f1)
        model_interpolations.append((val_f1s.copy(), lr))
        
    # Plotting
    for model_interp, lr in model_interpolations:
        plt.plot(lambda_values, model_interp, marker='o', label=f'LR: {lr}', color=colors[lr])

    plt.xlabel('Lambda (λ)', fontsize=14)  # Increase font size
    plt.ylabel('Validation F1 score', fontsize=14)  # Increase font size
    plt.title(f'Linear Mode Connectivity between models varying only in seed (Augmentation:{augmentation})', fontsize=12)  # Increase title font size
    plt.legend()
    plt.grid(True)

    # Save the figure
    plt.savefig(f'./plots_v2/linear_connectivity/{DATASET}/seed_wrt_aug/{augmentation}_performance_plot.png')


    # Display the figure



# vary_lr = {}
# for ind, row in df.iterrows():
#     row['Learning Rate'] = str(row['Learning Rate'])
#     row['Augmentation'] = str(row['Augmentation'])
#     row['Seed'] = int(row['SEED'])
#     if row['Augmentation'] in vary_lr.keys():
#         if row['Seed'] in vary_lr[row['Augmentation']].keys():
#             vary_lr[row['Augmentation']][row['Seed']].append(int(row['Model Name']))
#         else:
#             vary_lr[row['Augmentation']][row['Seed']] = [int(row['Model Name'])]
#     else:
#         vary_lr[row['Augmentation']] = {}
#         vary_lr[row['Augmentation']][row['Seed']] = [int(row['Model Name'])]
# print(vary_lr)

# # take the vary_seed and create vary_seed_wrt_lr and vary_seed_wrt_aug
# vary_seed_wrt_lr = {}
# for key, value in vary_seed.items():
#     vary_seed_wrt_lr[key] = list(value.values())
# print(vary_seed_wrt_lr)

# # vary_seed_wrt_aug = {}
# # for key, value in vary_seed.items():
# #     for inner_key in value.keys():
# #         if inner_key in vary_seed_wrt_aug.keys():
# #             vary_seed_wrt_aug[inner_key].append(vary_seed[key][inner_key])
# #         else:
# #             vary_seed_wrt_aug[inner_key] = [vary_seed[key][inner_key]]
# # print(vary_seed_wrt_aug)



# # vary_seed_list = []
# # for value_dict in vary_seed.values():
# #     for inner_list in value_dict.values():
# #         vary_seed_list.append(inner_list)
# # print(vary_seed_list)
# # # initial_model_path = '/home/santosh.sanjeev/Projects/model-soups/noodles/runs/san_final_hyp_models/san-initial_models/aptos_final_hyp/deitB_imagenet/2024-01-10_02-40-01/best_checkpoint.pth'





# #iterate through all the vary_seed_wrt_lr and then take each pair of models and interpolate them and for each LR get the plot of interploation

# lambda_values = [i / 8 for i in range(9)]
# print(lambda_values)

# colors = {'Heavy': 'red', 'Medium': 'blue', 'Minimal': 'green'}

# # Iterate through vary_seed
# for lr, augmentation_dict in vary_seed.items():
#     plt.figure(figsize=(10, 6))
#     model_interpolations = []
#     for augmentation, combination_sets in augmentation_dict.items():
#         # for combination_sets in combination_sets:
#         print(combination_sets)
#         val_losses = []
#         val_accs = []
#         val_f1s = []
#         val_recalls = []
#         val_kappas = []
#         val_aucs = []

#         for l in lambda_values:
#             if l == 0:
#                 l = 1e-8
#             if l == 1:
#                 l = 1 - 1e-8
#             alphal = [l, 1 - l]
#             print(alphal)
#             final_model = get_model(MODEL, num_classes=NUM_CLASSES)
#             final_model = souping(final_model, [state_dicts[combination_sets[0]], state_dicts[combination_sets[1]]], alphal)
#             final_model = final_model.to(DEVICE)
#             val_loss, val_acc, val_f1, val_recall, val_kappa, val_auc = val_step(final_model, val_loader, train_loader, loss, DEVICE, CLASSIFICATION)
#             val_f1s.append(val_f1)
#         model_interpolations.append((val_f1s.copy(), augmentation))
#         print(model_interpolations)
        
#     # Plotting
#     for model_interp, aug in model_interpolations:
#         plt.plot(lambda_values, model_interp, marker='o', label=f'{aug} augmentation', color=colors[aug])

#     plt.xlabel('Lambda')
#     plt.ylabel('Validation Accuracy')
#     plt.title(f'Performance of Final Model with Interpolated Weights (LR: {lr})')
#     plt.legend()
#     plt.grid(True)

#     # Save the figure
#     plt.savefig(f'./plots_v2/linear_connectivity/APTOS/seed_wrt_lr/{lr}_performance_plot.png')

#     # Display the figure


# vary_seed_wrt_aug = {}

# for lr, augmentation_dict in vary_seed.items():
#     for augmentation, combination_sets in augmentation_dict.items():
#         if augmentation not in vary_seed_wrt_aug:
#             vary_seed_wrt_aug[augmentation] = {}
#         vary_seed_wrt_aug[augmentation][lr] = combination_sets

# print(vary_seed_wrt_aug)
# lambda_values = [i / 8 for i in range(9)]
# print(lambda_values)
# print(vary_seed_wrt_aug.keys())
# all_lrs = set(lr for aug_lr_dict in vary_seed_wrt_aug.values() for lr in aug_lr_dict.keys())
# colors = {lr: f'C{i}' for i, lr in enumerate(all_lrs)}
# print(colors)
# # Iterate through vary_seed
# for augmentation, lr_dict in vary_seed_wrt_aug.items():
#     plt.figure(figsize=(10, 6))
#     model_interpolations = []
#     for lr, combination_sets in lr_dict.items():
#         print(combination_sets)
#         val_losses = []
#         val_accs = []
#         val_f1s = []
#         val_recalls = []
#         val_kappas = []
#         val_aucs = []
#         for l in lambda_values:
#             if l == 0:
#                 l = 1e-8
#             if l == 1:
#                 l = 1 - 1e-8
#             alphal = [l, 1 - l]
#             print(alphal)
#             final_model = get_model(MODEL, num_classes=NUM_CLASSES)
#             final_model = souping(final_model, [state_dicts[combination_sets[0]], state_dicts[combination_sets[1]]], alphal)
#             final_model = final_model.to(DEVICE)
#             val_loss, val_acc, val_f1, val_recall, val_kappa, val_auc = val_step(final_model, val_loader, train_loader, loss, DEVICE, CLASSIFICATION)
#             val_f1s.append(val_f1)
#         model_interpolations.append((val_f1s.copy(), lr))
#         print(model_interpolations)
        
#     # Plotting
#     for model_interp, lr in model_interpolations:
#         plt.plot(lambda_values, model_interp, marker='o', label=f'LR: {lr}', color=colors[lr])

#     plt.xlabel('Lambda')
#     plt.ylabel('Validation Accuracy')
#     plt.title(f'Performance of Final Model with Interpolated Weights (Augmentation: {augmentation})')
#     plt.legend()
#     plt.grid(True)

#     # Save the figure
#     plt.savefig(f'./plots_v2/linear_connectivity/APTOS/seed_wrt_aug/{augmentation}_performance_plot.png')

#     # Display the figure
