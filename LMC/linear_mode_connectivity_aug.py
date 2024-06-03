


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

# if not os.path.exists(f'./plots_v2/linear_connectivity/{DATASET}/lr_wrt_aug/'):
#     os.makedirs(f'./plots_v2/linear_connectivity/{DATASET}/lr_wrt_aug/')


# if not os.path.exists(f'./plots_v2/linear_connectivity/{DATASET}/lr_wrt_lr/'):
#     os.makedirs(f'./plots_v2/linear_connectivity/{DATASET}/lr_wrt_lr/')
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
    




# vary_seed = {}
# for ind, row in df.iterrows():
#     # if str(row['Learning Rate'])!='1e-06':
#     row['Learning Rate'] = str(row['Learning Rate'])
#     row['Augmentation'] = str(row['Augmentation'])
#     if row['Learning Rate'] in vary_seed.keys():
#         if row['Augmentation'] in vary_seed[row['Learning Rate']].keys():
#             vary_seed[row['Learning Rate']][row['Augmentation']].append(int(row['Model Name']))
#         else:
#             vary_seed[row['Learning Rate']][row['Augmentation']] = [int(row['Model Name'])]
#     else:
#         vary_seed[row['Learning Rate']] = {}
#         vary_seed[row['Learning Rate']][row['Augmentation']] = [int(row['Model Name'])]
#     # val_accuracy_list[int(row['Model Name'])] = row['Val Accuracy']



# print(vary_seed)
# # take the vary_seed and create vary_seed_wrt_lr and vary_seed_wrt_aug
# vary_seed_wrt_lr = {}
# for key, value in vary_seed.items():
#     vary_seed_wrt_lr[key] = list(value.values())
# print(vary_seed_wrt_lr)
vary_aug = {}
for ind, row in df.iterrows():
    row['Learning Rate'] = str(row['Learning Rate'])
    row['Augmentation'] = str(row['Augmentation'])
    row['Seed'] = int(row['SEED'])
    if row['Learning Rate'] in vary_aug.keys():
        if row['Seed'] in vary_aug[row['Learning Rate']].keys():
            vary_aug[row['Learning Rate']][row['Seed']].append(int(row['Model Name']))
        else:
            vary_aug[row['Learning Rate']][row['Seed']] = [int(row['Model Name'])]
    else:
        vary_aug[row['Learning Rate']] = {}
        vary_aug[row['Learning Rate']][row['Seed']] = [int(row['Model Name'])]

print(vary_aug)

lambda_values = [i / 8 for i in range(9)]
import itertools
# Define colors for each augmentation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import itertools

# Define augmentation colors
augmentation_colors = {'Minimal': 'red', 'Medium': 'green', 'Heavy': 'blue'}

# Set the y-axis limits
# plt.ylim(0.2, 1)

# Define gray background color
# plt.rcParams['axes.facecolor'] = 'lightgray'

# Create a colormap that smoothly transitions between two colors
def create_smooth_colormap(color1, color2):
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_gradient', [color1, color2], N=256)
    return cmap

# Iterate through vary_aug
for lr, seed_aug_dict in vary_aug.items():
    plt.figure(figsize=(10, 6))
    
    # Plotting
    for seed, aug_model_ids in seed_aug_dict.items():
        
        selected_combinations = list(itertools.combinations(aug_model_ids, 2))
        for model_id1, model_id2 in selected_combinations:
            aug1 = df[df['Model Name'] == model_id1]['Augmentation'].iloc[0]
            aug2 = df[df['Model Name'] == model_id2]['Augmentation'].iloc[0]
            aug_colors = [augmentation_colors[aug1], augmentation_colors[aug2]]
            
            # Interpolate and compute performance metrics
            val_f1s = []
            for l in lambda_values:
                if l == 0:
                    l = 1e-8
                if l == 1:
                    l = 1 - 1e-8
                alphal = [l, 1 - l]
                final_model = get_model(MODEL, num_classes=NUM_CLASSES)
                final_model = souping(final_model, [state_dicts[model_id1], state_dicts[model_id2]], alphal)
                final_model = final_model.to(DEVICE)
                val_loss, val_acc, val_f1, val_recall, val_kappa, val_auc = val_step(final_model, val_loader, train_loader, loss, DEVICE, CLASSIFICATION)
                val_f1s.append(val_f1)
            
            # Create a smooth colormap gradient
            cmap = create_smooth_colormap(aug_colors[0], aug_colors[1])
            colors = [cmap(i) for i in np.linspace(0, 1, len(lambda_values))]
            
            # Plot interpolation curve segment with gradient color
            for i in range(len(lambda_values) - 1):
                plt.plot(lambda_values[i:i+2], val_f1s[i:i+2], color=colors[i], linestyle='-', linewidth=2)
                
    # Add augmentation-based colors to legend
    legend_handles = []
    for aug, color in augmentation_colors.items():
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=4, label=f'Augmentation: {aug}'))
    
    plt.xlabel('Lambda (λ)', fontsize=14)
    plt.ylabel('Validation F1 score', fontsize=14)
    plt.title(f'Linear Mode Connectivity between models varying only in augmentation (Learning Rate: {lr})', fontsize=12)
    plt.legend(handles=legend_handles, loc='lower right', fontsize='large')
    plt.ylim(0.9, 1)
    plt.grid(True)

    # Save the figure
    plt.savefig(f'./plots_v2/linear_connectivity/test/{DATASET}_interpolation_aug_{lr}.png', bbox_inches='tight')
