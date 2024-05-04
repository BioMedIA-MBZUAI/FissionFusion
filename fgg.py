from utils.utils import souping, greedy_souping
from models import get_model
from engine import trainer, val_step, train_step

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import tabulate

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

from utils.utils import EarlyStopper, get_dataset, cyclic_learning_rate, adjust_learning_rate, train, test, save_checkpoint,cyclic_learning_rate_v2
from torch.utils.data.dataset import Subset

import yaml
import json
import time
import os
import wandb
from utils.utils import load_model
import sys



parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/fgg.yaml', metavar='DIR', help='configs')

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
DEVICE = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else 'cpu')
LR1 = config['LR1']
LR2 = config['LR2']
cycle = config['CYCLE']
WD = config['WD']
MOMENTUM = config["MOMENTUM"]
FGG_SAVE_DIR = config["FGG_SAVE_DIR"]
# assert cycle % 2 == 0, 'Cycle length should be even'

print('>>>>>>>>>>>>>>>>>>>>>>>>>>', DATASET,'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')


def START_seed(start_seed=9):
    seed = start_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed) 
START_seed(start_seed=0)
#if there is no test directory create it
if not os.path.exists(FGG_SAVE_DIR):
    os.makedirs(FGG_SAVE_DIR)

#if there is no folder named after dataset inside test directory create it
if not os.path.exists(os.path.join(FGG_SAVE_DIR, "fgg", DATASET, MODEL)):
    os.makedirs(os.path.join(FGG_SAVE_DIR, "fgg", DATASET, MODEL))

test_save_path = os.path.join(FGG_SAVE_DIR, "fgg", DATASET, MODEL)
# # Open a text file for logging
log_file_path = os.path.join(FGG_SAVE_DIR,"fgg", DATASET, MODEL, "output_log.txt")
print(log_file_path)
log_file = open(log_file_path, "w")

# Redirect the standard output to the log file
sys.stdout = log_file


# train_loader, val_loader, test_loader = get_dataset(DATASET, PATHS, "Minimal", 
#                                                         PRETRAINING, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, TASK)

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


#### create dictionary with learning_rates as keys and only the model paths with heavy augmentations as the keys (the keys are list of keys)
DICTIONARY = {}

for idx, folder in enumerate(sorted(os.listdir(WEIGHT_PATH))):
    if folder.startswith('testresults'):
        continue
    model_path = os.path.join(WEIGHT_PATH, folder)
    #read config in train_summary.json
    train_summary = json.load(open(os.path.join(model_path, "train_summary.json"), 'r'))
    model_config = train_summary["config"]
    if model_config['AUGMENTATION'] == 'Heavy':
        if model_config['LEARNING_RATE'] in DICTIONARY.keys():# is in DICTIONARY.keys():
            DICTIONARY[model_config['LEARNING_RATE']].append(model_path)
        else:
            DICTIONARY[model_config['LEARNING_RATE']] = []
            DICTIONARY[model_config['LEARNING_RATE']].append(model_path)
print(DICTIONARY)


for lr in DICTIONARY.keys():
    state_dicts = []
    val_results = []
    test_results = []
    val_models = [0]
    test_models = [0]

    if not os.path.exists(f'{FGG_SAVE_DIR}/fgg/{DATASET}/{MODEL}/{lr}'):
        os.makedirs(f'{FGG_SAVE_DIR}/fgg/{DATASET}/{MODEL}/{lr}')
    save_dir = f'{FGG_SAVE_DIR}/fgg/{DATASET}/{MODEL}/{lr}'

    print('SAVE_DIRRRRRRRRRRRR', save_dir)
    model_path = random.sample(DICTIONARY[lr], 1)[0]
    print(model_path)
    checkpoint = torch.load(os.path.join(model_path, "best_checkpoint.pth"), map_location=DEVICE)
    train_summary = json.load(open(os.path.join(model_path, "train_summary.json"), 'r'))
    model_config = train_summary["config"]
    state_dicts.append(checkpoint['model'])
    model = get_model(model_config["MODEL"], num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint['model'])
    model = model.to(DEVICE)
    LR1 = 1e-5#float(lr) *0.1 #added
    LR2 = 1e-8#LR1 * 0.001 #added
    optimizer = torch.optim.AdamW(model.parameters(),lr=LR1,weight_decay=WD) #added
    optimizer.load_state_dict(checkpoint['optimizer'])
    train_loader, val_loader, test_loader = get_dataset(DATASET, PATHS, 'Heavy', PRETRAINING, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, TASK)
    val_loss, val_acc, val_f1, val_recall, val_kappa, val_auc = val_step(model, val_loader, train_loader, loss, device = DEVICE, classification = CLASSIFICATION)
    test_loss, test_acc, test_f1, test_recall, test_kappa, test_auc = val_step(model, test_loader, train_loader, loss, device = DEVICE, classification = CLASSIFICATION)

    print('BASE MODEL', val_acc, test_acc)
    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'va_acc', 'te_acc', 'time']
    augmentation = 'Minimal'
    seed = 0
    aug_variable = 0
    seed_variable = 0
    save_checkpoint(
    save_dir,
    int(model_config['SEED'])-1,
    name='fgg',
    model_state=model.state_dict(),
    optimizer_state=optimizer.state_dict()
    )
    
    val_results.append({'Model Name': int(model_config['SEED'])-1,
                                    'Val Accuracy': val_acc,
                                    'Val F1': val_f1,
                                    'Val Recall': val_recall,
                                    'Val Kappa': val_kappa,
                                    'Val AUC': val_auc,})
    test_results.append({'Model Name': int(model_config['SEED'])-1,
                                    'Test Accuracy': test_acc,
                                    'Test F1': test_f1,
                                    'Test Recall': test_recall,
                                    'Test Kappa': test_kappa,
                                    'Test AUC': test_auc,})

    for epoch in range(1, 18):
        time_ep = time.time()
        lr_schedule = cyclic_learning_rate(epoch, cycle, LR1, LR2)
        # train_res = train(train_loader, model, optimizer, loss, lr_schedule=lr_schedule)
        train_loss, train_acc, train_macro_f1, train_macro_recall, train_kappa, train_auc = train_step(model, train_loader,loss, optimizer, DEVICE, classification = CLASSIFICATION, lr_schedule = lr_schedule)
        # test_res = test(val_loader, model, loss)
        val_loss, val_acc, val_f1, val_recall, val_kappa, val_auc = val_step(model, val_loader, train_loader, loss, device = DEVICE, classification = CLASSIFICATION)
        test_loss, test_acc, test_f1, test_recall, test_kappa, test_auc = val_step(model, test_loader, train_loader, loss, device = DEVICE, classification = CLASSIFICATION)


        time_ep = time.time() - time_ep
        if (epoch % cycle + 1) == cycle // 2:
            print(epoch)
            save_checkpoint(
                save_dir,
                epoch,
                name='fgg',
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict()
            )
        state_dicts.append(model.state_dict())
        val_models.append(epoch)
        test_models.append(epoch)
        val_results.append({'Model Name': epoch,
                                'Val Accuracy': val_acc,
                                'Val F1': val_f1,
                                'Val Recall': val_recall,
                                'Val Kappa': val_kappa,
                                'Val AUC': val_auc,})
        test_results.append({'Model Name': epoch,
                                'Test Accuracy': test_acc,
                                'Test F1': test_f1,
                                'Test Recall': test_recall,
                                'Test Kappa': test_kappa,
                                'Test AUC': test_auc,})
            
        
        # values = [epoch, lr_schedule(1.0), train_res['loss'], train_res['accuracy'], val_acc, test_acc, time_ep]
        values = [epoch, lr_schedule(1.0), train_loss,train_auc , val_auc, test_auc, time_ep]

        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.8f')
        print()
        if epoch % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(epoch,val_loss, val_acc, val_f1, val_recall, val_kappa, val_auc)
        print(epoch, test_loss, test_acc, test_f1, test_recall, test_kappa, test_auc)

    results_test_df = pd.DataFrame(test_results)
    results_val_df = pd.DataFrame(val_results)

    val_copy = results_val_df.copy()
    sorted_val = val_copy.sort_values(by= val_sort_by,ascending=False)
    sorted_val.to_csv(os.path.join(save_dir, "VAL_RESULTS.csv"), index=False)

    test_copy = results_test_df.copy()
    sorted_test = test_copy.sort_values(by= test_sort_by,ascending=False)
    sorted_test.to_csv(os.path.join(save_dir, "TEST_RESULTS.csv"), index=False)


log_file.close()