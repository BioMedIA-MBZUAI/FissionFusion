import torch

from tqdm import tqdm
import os
import wandb
import json
from utils.train_utils import get_weights
from sklearn.metrics import f1_score, recall_score, roc_auc_score



def multilabel_xray_dataset(targets, outputs, criterion, model, weights, device, task_outputs, task_targets, taskweights):

    loss = torch.zeros(1).to(device).float()
    for task in range(targets.shape[1]):
        task_output = outputs[:,task]
        task_target = targets[:,task]
        mask = ~torch.isnan(task_target)
        task_output = task_output[mask]
        task_target = task_target[mask]
        if len(task_target) > 0:
            task_loss = criterion(task_output.float(), task_target.float())
            if taskweights:
                loss += weights[task]*task_loss
            else:
                loss += task_loss
        
        task_outputs[task].append(task_output.detach().cpu().numpy())
        task_targets[task].append(task_target.detach().cpu().numpy())

    # here regularize the weight matrix when label_concat is used #CHANGEEEEE THIS
    label_concat_reg = False
    label_concat = False


    if label_concat_reg:
        if not label_concat:
            raise Exception("cfg.label_concat must be true")
        weight = model.classifier.weight
        num_labels = 13
        num_datasets = weight.shape[0]//num_labels
        weight_stacked = weight.reshape(num_datasets,num_labels,-1)
        label_concat_reg_lambda = torch.tensor(0.1).to(device).float()
        for task in range(num_labels):
            dists = torch.pdist(weight_stacked[:,task], p=2).mean()
            loss += label_concat_reg_lambda*dists
            
    loss = loss.sum()

    # if cfg.featurereg:
    #     feat = model.features(images)
    #     loss += feat.abs().sum()
        
    # if cfg.weightreg:
    #     loss += model.classifier.weight.abs().sum()

    return loss, task_outputs, task_targets



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


def train_step(
        model: torch.nn.Module,
        train_loader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        classification = 'MultiClass',
        lr_schedule = None,
):
    """
    Train model for one epoch.

    Args:
        model: PyTorch model to train.
        train_loader: PyTorch dataloader for training data.
        loss_fn: PyTorch loss function.
        optimizer: PyTorch optimizer.
        device: PyTorch device to use for training.

    Returns:
        Average loss, accuracy, macro F1 score, and macro recall for the epoch.
    """

    model.train()
    train_loss = 0.0
    train_acc = 0.0
    train_targets = []
    train_predictions = []
    if classification == 'MultiLabel':
        task_outputs={}
        task_targets={}
        for task in range(13):
            task_outputs[task] = []
            task_targets[task] = []
    
    num_iters = len(train_loader)
    for iter, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        
        if classification == 'MultiLabel':
            weights = get_weights(device = device, train_loader=train_loader, taskweights=True)
            loss, task_outputs, task_targets = multilabel_xray_dataset(target, output, loss_fn, model, weights, device, task_outputs, task_targets, taskweights=True)
        else:
            loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if classification != "MultiLabel":
            _, predicted = output.max(1)

            train_targets.extend(target.cpu().numpy())
            train_predictions.extend(predicted.cpu().numpy())

            train_acc += predicted.eq(target).sum().item() / len(target)

    if classification != "MultiLabel":
        train_kappa = cohen_kappa_score(train_targets, train_predictions, weights = 'quadratic')
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_macro_f1 = f1_score(train_targets, train_predictions, average='macro')
        train_macro_recall = recall_score(train_targets, train_predictions, average='macro')
        train_auc = roc_auc_score(train_targets, train_predictions, multi_class='ovr', average = 'weighted')

    else:
        for task in range(len(task_targets)):
            task_outputs[task] = np.concatenate(task_outputs[task])
            task_targets[task] = np.concatenate(task_targets[task])


        train_loss /= len(train_loader)
        task_aucs = []
        for task in range(len(task_targets)):
            if len(np.unique(task_targets[task]))> 1:
                task_auc = roc_auc_score(task_targets[task], task_outputs[task])
                #print(task, task_auc)
                task_aucs.append(task_auc)
            else:
                task_aucs.append(np.nan)

        task_aucs = np.asarray(task_aucs)
        train_auc = np.mean(task_aucs[~np.isnan(task_aucs)])
        train_acc, train_macro_f1, train_macro_recall, train_kappa = 0,0,0,0

    return train_loss, train_acc, train_macro_f1, train_macro_recall, train_kappa, train_auc


def val_step(
        model: torch.nn.Module,
        val_loader,
        train_loader,
        loss_fn: torch.nn.Module,
        device: torch.device,
        classification = 'MultiClass',
):
    """
    Evaluate model on val data.

    Args:
        model: PyTorch model to evaluate.
        val_loader: PyTorch dataloader for val data.
        loss_fn: PyTorch loss function.
        device: PyTorch device to use for evaluation.

    Returns:
        Average loss, accuracy, macro F1 score, and macro recall for the validation set.
    """

    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_targets = []
    val_predictions = []
    if classification == 'MultiLabel':
        task_outputs={}
        task_targets={}
        for task in range(13):
            task_outputs[task] = []
            task_targets[task] = []


    with torch.no_grad():
        for data, target in tqdm(val_loader):
            data, target = data.to(device), target.to(device)
            data = data.float()
            output = model(data)
            if classification == 'MultiLabel':
                weights = get_weights(device = device, train_loader=train_loader, taskweights=True)
                loss, task_outputs, task_targets = multilabel_xray_dataset(target, output, loss_fn, model, weights, device, task_outputs, task_targets, taskweights=True)
            else:
                loss = loss_fn(output, target)
            val_loss += loss.item()
            if classification != "MultiLabel":
                _, predicted = output.max(1)

                val_targets.extend(target.cpu().numpy())
                val_predictions.extend(predicted.cpu().numpy())

                val_acc += predicted.eq(target).sum().item() / len(target)

        if classification != "MultiLabel":
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            val_macro_f1 = f1_score(val_targets, val_predictions, average='macro')
            val_macro_recall = recall_score(val_targets, val_predictions, average='macro')
            val_kappa = cohen_kappa_score(val_targets,val_predictions, weights = 'quadratic')
            val_auc = 0#roc_auc_score(val_targets, val_predictions, multi_class='ovr', average = 'weighted')
        else:
            for task in range(len(task_targets)):
                task_outputs[task] = np.concatenate(task_outputs[task])
                task_targets[task] = np.concatenate(task_targets[task])


            val_loss /= len(val_loader)
            task_aucs = []
            for task in range(len(task_targets)):
                if len(np.unique(task_targets[task]))> 1:
                    task_auc = roc_auc_score(task_targets[task], task_outputs[task])
                    #print(task, task_auc)
                    task_aucs.append(task_auc)
                else:
                    task_aucs.append(np.nan)

            task_aucs = np.asarray(task_aucs)
            val_auc = np.mean(task_aucs[~np.isnan(task_aucs)])
            val_acc, val_macro_f1, val_macro_recall, val_kappa = 0,0,0,0


    return val_loss, val_acc, val_macro_f1, val_macro_recall, val_kappa, val_auc

def trainer(
        model: torch.nn.Module,
        train_loader,
        val_loader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler,
        lr_scheduler_name: str,
        device: torch.device,
        epochs: int,
        save_dir: str,
        early_stopper=None,
        linear_probing_epochs=None,
        start_epoch = 1,
        dataset = 'CIFAR'
):
    """
    Train and evaluate model.

    Args:
        model: PyTorch model to train.
        train_loader: PyTorch dataloader for training data.
        val_loader: PyTorch dataloader for val data.
        loss_fn: PyTorch loss function.
        optimizer: PyTorch optimizer.
        lr_scheduler: PyTorch learning rate scheduler.
        device: PyTorch device to use for training.
        epochs: Number of epochs to train the model for.

    Returns:
        Average loss and accuracy for the val set.
    """

    results = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1":[],
        "val_f1":[],
        "train_recall":[],
        "val_recall":[],
        "train_kappa":[],
        "val_kappa":[],
        "train_auc":[],
        "val_auc":[],
    }
    best_val_loss = 1e10

    if dataset == "MIMIC" or dataset == "CHEXPERT":
        classification = "MultiLabel"
    else:
        classification = "MultiClass"
    print(classification)
    for epoch in range(start_epoch, epochs + 1):

        # if linear_probing_epochs is not None:
        #     if epoch == linear_probing_epochs:
        #         for param in model.parameters():
        #             param.requires_grad = True
        print(f"Epoch {epoch}:")
        train_loss, train_acc, train_macro_f1, train_macro_recall, train_kappa, train_auc = train_step(model, train_loader, loss_fn, optimizer, device, classification)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_macro_f1:.4f}, Train recall: {train_macro_recall:.4f}, Train Kappa: {train_kappa:.4f}, Train AUC: {train_auc:.4f}")

        

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_f1"].append(train_macro_f1)
        results["train_recall"].append(train_macro_recall)
        results["train_kappa"].append(train_kappa)
        results["train_auc"].append(train_auc)

        val_loss, val_acc, val_f1, val_recall, val_kappa, val_auc = val_step(model, val_loader, train_loader, loss_fn, device, classification)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val recall: {val_recall:.4f}, Val Kappa: {val_kappa:.4f}, Val AUC: {val_auc:.4f}")
        print()

        if lr_scheduler_name == "ReduceLROnPlateau":
            lr_scheduler.step(val_loss)
        elif lr_scheduler_name != "None":
            lr_scheduler.step()
        
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["val_f1"].append(val_f1)
        results["val_recall"].append(val_recall)
        results["val_kappa"].append(val_kappa)
        results["val_auc"].append(val_auc)
        
        
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc, "train_f1": train_macro_f1, "train_recall": train_macro_recall, "val_f1": val_f1, "val_recall": val_recall,  "train_kappa": train_kappa, "val_kappa": val_kappa, "trian_auc": train_auc, "val_auc": val_auc})
        

        if lr_scheduler_name=="CyclicLR":
            checkpoint = { 
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': lr_scheduler.state_dict()}
        else:
            checkpoint = { 
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': lr_scheduler}
            
                    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(save_dir, "best_checkpoint.pth"))

        torch.save(checkpoint, os.path.join(save_dir, "last_checkpoint.pth"))

        if early_stopper is not None:
            if early_stopper.early_stop(val_loss):
                print("Early stopping")
                break

    return results













##########REGRESSION#############


import torch

from tqdm import tqdm
import os
import wandb
import json
import numpy as np
from sklearn.metrics import f1_score, recall_score, cohen_kappa_score

def reg_train_step(
        model: torch.nn.Module,
        train_loader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        classification=None,
):
    """
    Train model for one epoch.

    Args:
        model: PyTorch model to train.
        train_loader: PyTorch dataloader for training data.
        loss_fn: PyTorch loss function.
        optimizer: PyTorch optimizer.
        device: PyTorch device to use for training.

    Returns:
        Average loss, accuracy, macro F1 score, and macro recall for the epoch.
    """

    model.train()
    train_loss = 0.0
    train_acc = 0.0
    train_targets = []
    train_predictions = []

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output.float(), target.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        predicted = torch.clip(torch.round(output), 0, 4)
        # print(output.detach().cpu().numpy().reshape(1, -1))
        # print("\n Loss", loss.item())
        # print('predddddddd',predicted.reshape(1,-1))
        # print('targetttttt', target)

        # exit(0)
        train_targets.extend(target.cpu().numpy())
        train_predictions.extend(predicted.detach().cpu().numpy())
        
        
        train_acc += predicted.detach().eq(target).sum().item() / len(target)
    
    train_kappa = cohen_kappa_score(train_targets, train_predictions, weights = 'quadratic')
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    train_macro_f1 = f1_score(train_targets, train_predictions, average='macro')
    train_macro_recall = recall_score(train_targets, train_predictions, average='macro')

    return train_loss, train_acc, train_macro_f1, train_macro_recall, train_kappa

def reg_val_step(
        model: torch.nn.Module,
        val_loader,
        train_loader,
        loss_fn: torch.nn.Module,
        device: torch.device,
        classification=None,
):
    """
    Evaluate model on val data.

    Args:
        model: PyTorch model to evaluate.
        val_loader: PyTorch dataloader for val data.
        loss_fn: PyTorch loss function.
        device: PyTorch device to use for evaluation.

    Returns:
        Average loss, accuracy, macro F1 score, and macro recall for the validation set.
    """

    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    val_targets = []
    val_predictions = []

    with torch.no_grad():
        for data, target in tqdm(val_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            val_loss += loss_fn(output.float(), target.float()).item()


            predicted = torch.clip(torch.round(output), 0, 4)

            val_targets.extend(target.cpu().numpy())
            val_predictions.extend(predicted.detach().cpu().numpy())
        
            val_acc += predicted.detach().eq(target).sum().item() / len(target)

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    val_macro_f1 = f1_score(val_targets, val_predictions, average='macro')
    val_macro_recall = recall_score(val_targets, val_predictions, average='macro')
    val_kappa = cohen_kappa_score(val_targets,val_predictions, weights = 'quadratic')
    val_auc = 0

    return val_loss, val_acc, val_macro_f1, val_macro_recall, val_kappa, val_auc

def reg_trainer(
        model: torch.nn.Module,
        train_loader,
        val_loader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler,
        lr_scheduler_name: str,
        device: torch.device,
        epochs: int,
        save_dir: str,
        early_stopper=None,
        linear_probing_epochs=None,
        start_epoch = 1,
        classification=None,
):
    """
    Train and evaluate model.

    Args:
        model: PyTorch model to train.
        train_loader: PyTorch dataloader for training data.
        val_loader: PyTorch dataloader for val data.
        loss_fn: PyTorch loss function.
        optimizer: PyTorch optimizer.
        lr_scheduler: PyTorch learning rate scheduler.
        device: PyTorch device to use for training.
        epochs: Number of epochs to train the model for.

    Returns:
        Average loss and accuracy for the val set.
    """

    results = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1":[],
        "val_f1":[],
        "train_recall":[],
        "val_recall":[],
        "train_kappa":[],
        "val_kappa":[]
    }
    best_val_loss = 1e10

    for epoch in range(start_epoch, epochs + 1):

        # if linear_probing_epochs is not None:
        #     if epoch == linear_probing_epochs:
        #         for param in model.parameters():
        #             param.requires_grad = True

        print(f"Epoch {epoch}:")
        train_loss, train_acc, train_macro_f1, train_macro_recall, train_kappa = reg_train_step(model, train_loader, loss_fn, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_macro_f1:.4f}, Train recall: {train_macro_recall:.4f}, Train Kappa: {train_kappa:.4f}")

        

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_f1"].append(train_macro_f1)
        results["train_recall"].append(train_macro_recall)
        results["train_kappa"].append(train_kappa)


        val_loss, val_acc, val_f1, val_recall, val_kappa = reg_val_step(model, val_loader, loss_fn, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}, Val recall: {val_recall:.4f}, Val Kappa: {val_kappa:.4f}")
        print()

        if lr_scheduler_name == "ReduceLROnPlateau":
            lr_scheduler.step(val_loss)
        elif lr_scheduler_name != "None":
            lr_scheduler.step()
        
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["val_f1"].append(val_f1)
        results["val_recall"].append(val_recall)
        results["val_kappa"].append(val_kappa)
        
        
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc, "train_f1": train_macro_f1, "train_recall": train_macro_recall, "val_f1": val_f1, "val_recall": val_recall,  "train_kappa": train_kappa, "val_kappa": val_kappa})
        

        if lr_scheduler_name=="CyclicLR":
            checkpoint = { 
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': lr_scheduler.state_dict()}
        else:
            checkpoint = { 
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': lr_scheduler}
            
                    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(save_dir, "best_checkpoint.pth"))

        torch.save(checkpoint, os.path.join(save_dir, "last_checkpoint.pth"))

        if early_stopper is not None:
            if early_stopper.early_stop(val_loss):
                print("Early stopping")
                break

    return results


# import torch

# from tqdm import tqdm

# from utils import save_model
# import os
# import wandb



# def train_step(
#         model: torch.nn.Module,
#         train_loader,
#         loss_fn: torch.nn.Module,
#         optimizer: torch.optim.Optimizer,
#         device: torch.device,
#         n_classes = 10,
#         mixup_alpha= None,
# ):
#     """
#     Train model for one epoch.

#     Args:
#         model: PyTorch model to train.
#         train_loader: PyTorch dataloader for training data.
#         loss_fn: PyTorch loss function.
#         optimizer: PyTorch optimizer.
#         device: PyTorch device to use for training.

#     Returns:
#         Average loss for the epoch.
#     """

#     model.train()
#     train_loss = 0.0
#     train_acc = 0.0
#     for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
#         if mixup_alpha!=None:
#             print('yesssssssss')
#             data, target = mixup(data, target, mixup_alpha, n_classes)
        
#         data, target = data.to(device), target.to(device)

#         optimizer.zero_grad()
#         output = model(data)

#         loss = loss_fn(output,target)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()

#         if mixup_alpha!=None:
#             _, target = target.max(dim=1)


#         _, predicted = output.max(1)



#         train_acc += predicted.eq(target).sum().item() / len(target)

#     train_loss /= len(train_loader)
#     train_acc /= len(train_loader)

#     return train_loss, train_acc

# def val_step(
#         model: torch.nn.Module,
#         val_loader,
#         loss_fn: torch.nn.Module,
#         device: torch.device,
# ):
#     """
#     Evaluate model on val data.

#     Args:
#         model: PyTorch model to evaluate.
#         val_loader: PyTorch dataloader for val data.
#         loss_fn: PyTorch loss function.
#         device: PyTorch device to use for evaluation.

#     Returns:
#         Average loss and accuracy for the val set.
#     """

#     model.eval()
#     val_loss = 0.0
#     val_acc = 0.0
#     with torch.no_grad():
#         for data, target in tqdm(val_loader):
#             data, target = data.to(device), target.to(device)

#             output = model(data)
#             val_loss += loss_fn(output, target).item()

#             _, predicted = output.max(1)
#             val_acc += predicted.eq(target).sum().item() / len(target)


#     val_loss /= len(val_loader)

#     val_acc /= len(val_loader)
#     return val_loss, val_acc


# def trainer(
#         model: torch.nn.Module,
#         train_loader,
#         val_loader,
#         loss_fn: torch.nn.Module,
#         optimizer: torch.optim.Optimizer,
#         lr_scheduler: torch.optim.lr_scheduler,
#         lr_scheduler_name: str,
#         device: torch.device,
#         epochs: int,
#         save_dir: str,
#         early_stopper=None,
#         linear_probing_epochs=None,
#         start_epoch = 1
# ):
#     """
#     Train and evaluate model.

#     Args:
#         model: PyTorch model to train.
#         train_loader: PyTorch dataloader for training data.
#         val_loader: PyTorch dataloader for val data.
#         loss_fn: PyTorch loss function.
#         optimizer: PyTorch optimizer.
#         lr_scheduler: PyTorch learning rate scheduler.
#         device: PyTorch device to use for training.
#         epochs: Number of epochs to train the model for.

#     Returns:
#         Average loss and accuracy for the val set.
#     """

#     results = {
#         "train_loss": [],
#         "val_loss": [],
#         "train_acc": [],
#         "val_acc": [],
#     }
#     best_val_loss = 1e10

#     for epoch in range(start_epoch, epochs + 1):

#         if linear_probing_epochs is not None:
#             if epoch == linear_probing_epochs:
#                 for param in model.parameters():
#                     param.requires_grad = True

#         print(f"Epoch {epoch}:")
#         train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, device)
#         print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        

#         results["train_loss"].append(train_loss)
#         results["train_acc"].append(train_acc)

#         val_loss, val_acc = val_step(model, val_loader, loss_fn, device)
#         print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
#         print()

#         if lr_scheduler_name == "ReduceLROnPlateau":
#             lr_scheduler.step(val_loss)
#         elif lr_scheduler_name != "None":
#             lr_scheduler.step()
        
#         results["val_loss"].append(val_loss)
#         results["val_acc"].append(val_acc)

#         wandb.log({"train_loss": train_loss, "val_loss": val_loss, "train_acc": train_acc, "val_acc": val_acc})

#         checkpoint = { 
#                 'epoch': epoch,
#                 'model': model.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'lr_sched': lr_scheduler}
        
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
            
#             torch.save(checkpoint, os.path.join(save_dir, "best_checkpoint.pth"))

#         torch.save(checkpoint, os.path.join(save_dir, "last_checkpoint.pth"))

#         if early_stopper is not None:
#             if early_stopper.early_stop(val_loss):
#                 print("Early stopping")
#                 break

#     return results
