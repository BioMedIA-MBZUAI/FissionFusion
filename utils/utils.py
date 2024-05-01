import torch
from engine import val_step
from models import get_model

from matplotlib import pyplot as plt
import numpy as np
import sklearn
from scipy.interpolate import interp1d
from scipy.integrate import quad
import cv2
from sklearn.metrics import confusion_matrix
from dataset import RSNADataset, HAM10000Dataset, AptosDataset, CheX_Dataset
import torchvision
from torchvision.transforms.v2 import AutoAugmentPolicy, functional as F, InterpolationMode, Transform
from torchvision.transforms import v2
torchvision.disable_beta_transforms_warning()
from timm.data.transforms import RandomResizedCropAndInterpolation
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader


def souping(model, state_dicts, alphal):
    sd = {k : state_dicts[0][k].clone() * alphal[0] for k in state_dicts[0].keys()}
    for i in range(1, len(state_dicts)):
        for k in state_dicts[i].keys():
            sd[k] = sd[k] + state_dicts[i][k].clone() * alphal[i]
    model.load_state_dict(sd)
    return model

def greedy_souping(state_dicts, val_results, model_config, NUM_CLASSES, val_loader, train_loader, loss, DEVICE, CLASSIFICATION, sort_by, val_models = None):
    ranked_candidates = [i for i in range(len(val_models))]
    ranked_candidates.sort(key=lambda x: -val_results[x])

        
    ranked_candidates_names = [i for i in range(len(val_models))]
    ranked_candidates_names.sort(key=lambda x: -val_results[x])

    current_best = val_results[ranked_candidates[0]]

    # if val_models:
    #     ranked_candidates = [val_models[i] for i in ranked_candidates]
    print(ranked_candidates_names)

    print('currentttttttt bestttttttt', current_best)

    
    best_ingredients = ranked_candidates[:1]
    best_ingredients_names = ranked_candidates_names[:1]
    for i in range(1, len(state_dicts)):
        # add current index to the ingredients
        ingredient_indices = best_ingredients + [ranked_candidates[i]]
        ingredient_names = best_ingredients_names + [ranked_candidates_names[i]]
        alphal = [0 for i in range(len(state_dicts))]
        for j in ingredient_indices:
            alphal[j] = 1 / len(ingredient_indices)

        # benchmark and conditionally append
        model = get_model(model_config, num_classes=NUM_CLASSES)

        greedy_model = souping(model, state_dicts, alphal)
        greedy_model.to(DEVICE)
        greedy_val_loss, greedy_val_acc, greedy_val_f1, greedy_val_recall, greedy_val_kappa, greedy_val_auc = val_step(greedy_model, val_loader, train_loader, loss, DEVICE, CLASSIFICATION)
        if 'F1' in sort_by:            
            print(f'Models {ingredient_names} got {greedy_val_f1} on validation.')
            if greedy_val_f1 > current_best:
                current_best = greedy_val_f1
                best_ingredients = ingredient_indices
                best_ingredients_names = ingredient_names
        elif 'Recall' in sort_by:
            print(f'Models {ingredient_names} got {greedy_val_recall} on validation.')
            if greedy_val_recall > current_best:
                current_best = greedy_val_recall
                best_ingredients = ingredient_indices
                best_ingredients_names = ingredient_names

        elif 'Accuracy' in sort_by:
            print(f'Models {ingredient_names} got {greedy_val_acc} on validation.')
            if greedy_val_acc > current_best:
                current_best = greedy_val_acc
                best_ingredients = ingredient_indices
                best_ingredients_names = ingredient_names

        elif 'Kappa' in sort_by:
            print(f'Models {ingredient_names} got {greedy_val_kappa} on validation.')
            if greedy_val_kappa > current_best:
                current_best = greedy_val_kappa
                best_ingredients = ingredient_indices
                best_ingredients_names = ingredient_names

        elif 'AUC' in sort_by:
            print(f'Models {ingredient_names} got {greedy_val_auc}% on validation.')
            if greedy_val_auc > current_best:
                current_best = greedy_val_auc
                best_ingredients = ingredient_indices
                best_ingredients_names = ingredient_names

    alphal = [0 for i in range(len(state_dicts))]
    for j in best_ingredients:
        alphal[j] = 1 / len(best_ingredients)
    greedy_model = souping(model, state_dicts, alphal)

    return greedy_model, best_ingredients_names


def get_dataset(DATASET, paths, augment, PRETRAINING, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, TASK):
    paths = paths.split('""')
    print(paths)
    if augment == "Minimal":
            if PRETRAINING != "ImageNet":
                train_transform = v2.Compose([
                    v2.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.9, 1.0), antialias=True),
                ])
            else:
                train_transform = v2.Compose([
                    v2.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.9, 1.0), antialias=True),
                    v2.ToTensor(),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

                ])

    elif augment == "Medium":
        if PRETRAINING != "ImageNet":

            train_transform = v2.Compose([
                RandomResizedCropAndInterpolation(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.08, 1.0), ratio=(0.75, 1.3333)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=None),
            ])
        else:
            train_transform = v2.Compose([
                RandomResizedCropAndInterpolation(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.08, 1.0), ratio=(0.75, 1.3333)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=None),
                v2.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


    elif augment == "Heavy":
        if PRETRAINING != "ImageNet":
            train_transform = v2.Compose([
                v2.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.7, 1.2), antialias=True),
                v2.RandAugment(num_ops=2, magnitude=15, interpolation = InterpolationMode.BILINEAR),
            ])
        else:
            train_transform = v2.Compose([
                v2.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.7, 1.2), antialias=True),
                v2.RandAugment(num_ops=2, magnitude=15, interpolation = InterpolationMode.BILINEAR),
                v2.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

            ])
    if PRETRAINING != "ImageNet":
        val_transform = v2.Compose([
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            ])
    else:
        val_transform = v2.Compose([
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])


    if DATASET == "Cifar10":
            ##Cifar Dataset
            trainset = torchvision.datasets.CIFAR10(root=paths[0], train=True, transform=train_transform, download=True)
            valset = torchvision.datasets.CIFAR10(root=paths[0], train=True, transform=val_transform, download=True)
            test_dataset = torchvision.datasets.CIFAR10(root=paths[0], train=False, transform=val_transform, download=True)

            idxs = np.load(paths[1]).astype('int')
            val_indices = []
            train_indices = []
            
            for i in range(len(idxs)):
                if idxs[i]:
                    val_indices.append(i)
                else:
                    train_indices.append(i)

            val_dataset = Subset(valset, val_indices)
            train_dataset = Subset(trainset, train_indices)

    elif DATASET == "Cifar100":
            ##Cifar Dataset
            trainset = torchvision.datasets.CIFAR100(root=paths[0], train=True, transform=train_transform, download=True)
            valset = torchvision.datasets.CIFAR100(root=paths[0], train=True, transform=val_transform, download=True)
            test_dataset = torchvision.datasets.CIFAR100(root=paths[0], train=False, transform=val_transform, download=True)

            idxs = np.load(paths[1]).astype('int')
            val_indices = []
            train_indices = []
            
            for i in range(len(idxs)):
                if idxs[i]:
                    val_indices.append(i)
                else:
                    train_indices.append(i)

            val_dataset = Subset(valset, val_indices)
            train_dataset = Subset(trainset, train_indices)
    
    elif DATASET == "Rsna":
        ##RSNA Dataset
        train_dataset = RSNADataset(csv_file=paths[1], data_folder=paths[0], split="train", pretraining = PRETRAINING, transform=train_transform)
        val_dataset = RSNADataset(csv_file=paths[1], data_folder=paths[0], split="val", pretraining = PRETRAINING, transform=val_transform)
        test_dataset = RSNADataset(csv_file=paths[1], data_folder=paths[0], split="test", pretraining = PRETRAINING, transform=val_transform)

    elif DATASET == "HAM":
        ##HAM Dataset
        train_dataset = HAM10000Dataset(csv_file=paths[3], data_folder=paths[0], pretraining = PRETRAINING, transform=train_transform)
        val_dataset = HAM10000Dataset(csv_file=paths[4], data_folder=paths[1], pretraining = PRETRAINING, transform=val_transform)
        test_dataset = HAM10000Dataset(csv_file=paths[5], data_folder=paths[2], pretraining = PRETRAINING, transform=val_transform)
    elif DATASET == "APTOS":
        ##APTOS Dataset
        train_dataset = AptosDataset(csv_file=paths[1], data_folder=paths[0], split = 'train', pretraining = PRETRAINING, task = TASK, transform=train_transform)
        val_dataset = AptosDataset(csv_file=paths[1], data_folder=paths[0], split = 'val', pretraining = PRETRAINING, task = TASK, transform=val_transform)
        test_dataset = AptosDataset(csv_file=paths[1], data_folder=paths[0], split = 'test', pretraining = PRETRAINING, task = TASK, transform=val_transform)
    
    elif DATASET == "CHEXPERT":
        ##APTOS Dataset
        train_dataset = CheX_Dataset(csv_file=paths[3], data_folder=paths[0], split = 'train', pretraining = PRETRAINING, task = TASK, transform=train_transform)
        val_dataset = CheX_Dataset(csv_file=paths[4], data_folder=paths[1], split = 'val', pretraining = PRETRAINING, task = TASK, transform=val_transform)
        test_dataset = CheX_Dataset(csv_file=paths[5], data_folder=paths[2], split = 'test', pretraining = PRETRAINING, task = TASK, transform=val_transform)
    

    print(DATASET, len(train_dataset), len(val_dataset), len(test_dataset))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader, test_loader
    

def plot_results(results, save_dir, name = None):

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(results['train_loss'], label='Train loss')
    plt.plot(results['val_loss'], label='Validation loss')

    
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(results['train_acc'], label='Train accuracy')
    plt.plot(results['val_acc'], label='Validation accuracy')
    plt.plot(results['train_f1'], label='Train F1')
    plt.plot(results['train_recall'], label='Train Recall')
    
    plt.plot(results['val_f1'], label='Val F1')
    plt.plot(results['val_recall'], label='Val Recall')
    plt.plot(results['train_kappa'], label='Train Kappa')
    plt.plot(results['val_kappa'], label='Val Kappa')
    plt.legend()
    if name:
        plt.savefig(save_dir + name)
    else:
        plt.savefig(save_dir + '/LossAccuracy.png')

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False




import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from torchvision.datasets import CIFAR10
from torchvision.transforms import *
from tqdm import tqdm


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, feature_dim, num_classes, normalize=False, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.classification_head = torch.nn.Linear(feature_dim, num_classes)
        self.normalize = normalize
        if not self.normalize:
            print('normalize skipped.')

        if initial_weights is not None and type(initial_weights) == tuple:
            print('tuple.')
            w, b = initial_weights
            self.classification_head.weight = torch.nn.Parameter(w.clone())
            self.classification_head.bias = torch.nn.Parameter(b.clone())
        else:
            if initial_weights is None:
                initial_weights = torch.zeros_like(self.classification_head.weight)
                torch.nn.init.kaiming_uniform_(initial_weights, a=math.sqrt(5))
            self.classification_head.weight = torch.nn.Parameter(initial_weights.clone())
            # Note: modified. Initial bug in forgetting to zero bias.
            self.classification_head.bias = torch.nn.Parameter(torch.zeros_like(self.classification_head.bias))

        # Note: modified. Get rid of the language part.
        delattr(self.model, 'transformer')

    def forward(self, images):
        features = self.model.encode_image(images).float()
        if self.normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        logits = self.classification_head(features)
        return logits
    

# def get_model(state_dicts, alphal):
#   model, _ = clip.load('ViT-B/32')
#   feature_dim = state_dicts[0]['classification_head.weight'].shape[1]
#   num_classes = state_dicts[0]['classification_head.weight'].shape[0]
#   normalize = True
#   model = ModelWrapper(model, feature_dim, num_classes, normalize)
#   sd = {k : state_dicts[0][k].clone() * alphal[0] for k in state_dicts[0].keys()}
#   for i in range(1, len(state_dicts)):
#       for k in state_dicts[i].keys():
#           sd[k] = sd[k] + state_dicts[i][k].clone() * alphal[i]
#   model.load_state_dict(sd)
#   return model

def zeroshot_classifier(model, classnames, templates, device):
    # print('Building zero-shot classifier.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in (classnames):
            texts = [template(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return 100*zeroshot_weights.t()


openai_imagenet_template = [
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a photo of many {c}.',
    lambda c: f'a sculpture of a {c}.',
    lambda c: f'a photo of the hard to see {c}.',
    lambda c: f'a low resolution photo of the {c}.',
    lambda c: f'a rendering of a {c}.',
    lambda c: f'graffiti of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a cropped photo of the {c}.',
    lambda c: f'a tattoo of a {c}.',
    lambda c: f'the embroidered {c}.',
    lambda c: f'a photo of a hard to see {c}.',
    lambda c: f'a bright photo of a {c}.',
    lambda c: f'a photo of a clean {c}.',
    lambda c: f'a photo of a dirty {c}.',
    lambda c: f'a dark photo of the {c}.',
    lambda c: f'a drawing of a {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'the plastic {c}.',
    lambda c: f'a photo of the cool {c}.',
    lambda c: f'a close-up photo of a {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a painting of the {c}.',
    lambda c: f'a painting of a {c}.',
    lambda c: f'a pixelated photo of the {c}.',
    lambda c: f'a sculpture of the {c}.',
    lambda c: f'a bright photo of the {c}.',
    lambda c: f'a cropped photo of a {c}.',
    lambda c: f'a plastic {c}.',
    lambda c: f'a photo of the dirty {c}.',
    lambda c: f'a jpeg corrupted photo of a {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a rendering of the {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'a photo of one {c}.',
    lambda c: f'a doodle of a {c}.',
    lambda c: f'a close-up photo of the {c}.',
    lambda c: f'a photo of a {c}.',
    lambda c: f'the origami {c}.',
    lambda c: f'the {c} in a video game.',
    lambda c: f'a sketch of a {c}.',
    lambda c: f'a doodle of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a low resolution photo of a {c}.',
    lambda c: f'the toy {c}.',
    lambda c: f'a rendition of the {c}.',
    lambda c: f'a photo of the clean {c}.',
    lambda c: f'a photo of a large {c}.',
    lambda c: f'a rendition of a {c}.',
    lambda c: f'a photo of a nice {c}.',
    lambda c: f'a photo of a weird {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a cartoon {c}.',
    lambda c: f'art of a {c}.',
    lambda c: f'a sketch of the {c}.',
    lambda c: f'a embroidered {c}.',
    lambda c: f'a pixelated photo of a {c}.',
    lambda c: f'itap of the {c}.',
    lambda c: f'a jpeg corrupted photo of the {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a plushie {c}.',
    lambda c: f'a photo of the nice {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the weird {c}.',
    lambda c: f'the cartoon {c}.',
    lambda c: f'art of the {c}.',
    lambda c: f'a drawing of the {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'the plushie {c}.',
    lambda c: f'a dark photo of a {c}.',
    lambda c: f'itap of a {c}.',
    lambda c: f'graffiti of the {c}.',
    lambda c: f'a toy {c}.',
    lambda c: f'itap of my {c}.',
    lambda c: f'a photo of a cool {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a tattoo of the {c}.',
]


def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule

def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule

def cyclic_learning_rate_v2(epoch, cycle, alpha_1, alpha_2):
    # def schedule(iter):
    #     t = ((epoch % cycle) + iter) / cycle
    #     if t < 0.4:  # Increase for 2 epochs
    #         return alpha_2 - ((alpha_2 - alpha_1) / 0.4) * t
    #     else:  # Decrease for 3 epochs
    #         return alpha_1 + ((alpha_2 - alpha_1) / 0.6) * (t - 0.4)
    # return schedule
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.4:  # Increase for 2 epochs
            return alpha_1 + ((alpha_2 - alpha_1) / 0.4) * t
        else:  # Decrease for 3 epochs
            return alpha_2 - ((alpha_2 - alpha_1) / 0.6) * (t - 0.4)
    return schedule




def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)



def train(train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None):
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    model.train()
    for iter, (input, target) in enumerate(tqdm((train_loader))):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        input = input.to('cuda')
        target = target.to('cuda')

        output = model(input)
        loss = criterion(output, target)
        if regularizer is not None:
            loss += regularizer(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }


def test(test_loader, model, criterion, regularizer=None, **kwargs):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0
    model = model.to('cuda')
    model.eval()
    # print(model)
    # print('hooooodfshdkfdfjhsdkjg')
    with torch.no_grad():
        for input, target in test_loader:
            # print('hooooo')
            input = input.to('cuda')
            target = target.to('cuda')

            output = model(input, **kwargs)
            # print(output)
            nll = criterion(output, target)
            loss = nll.clone()
            if regularizer is not None:
                loss += regularizer(model)

            nll_sum += nll.item() * input.size(0)
            loss_sum += loss.item() * input.size(0)
            pred = torch.nn.functional.softmax(output).data.argmax(1, keepdim=True)
            # print(pred,target)
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            # print(correct)
    return {
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }