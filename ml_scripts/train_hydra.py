#!/usr/bin/env python3

""" training script for kenya project 

This script defines a network, hyperparameters, optimizer,
scheduler, etc, and then performs training on a dataset of 
seismic traces (input) and animal species (output).

----
NOTE: Edits inputs in inputs/params_train.py before running the script.
----
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchvision.models as models
import torch.nn as nn
import os
import copy
import datetime
import hydra
from sklearn.metrics import confusion_matrix
from omegaconf import DictConfig, OmegaConf
from torch.optim import lr_scheduler
from .models_spectrograms import PretrainedCNN
from .models_seismograms import cnn1, cnn_simple, mlp, cnn_simple2
from .utilities.callbacks_hydra import update_best_model, save_best_model, EarlyStopping, Logger
from .datasets_hydra import SpectrogramDataset, SeismogramDataset


def train_step(model, optimizer, criterion, inputs, labels, device, train_conf_matrix):
    """ Trains on a batch of data.
    
    Performs SGD step on a batch of training data: 
    make predictions, compute gradients, update params.
    Store metrics for this step.

    Parameters:
    -----------
    model: torch.nn.Module object
        ML model in training.
    criterion: torch.nn.Loss object
        Loss used in training.
    optimizer: torch.optim.Optimizer object
        Optimizer used in training.
    inputs: torch.Tensor
        Batch of spectrograms.
    labels: torch.Tensor
        Batch of class labels.
    device: string
        Device to be used, "cuda" or "cpu".
    train_conf_matrix: pandas DataFrame
        confusion matrix for this epoch.
 
    Returns:
    -------
    train_loss: float
        loss for this batch of data.
    
    """
    
    optimizer.zero_grad()
    inputs = inputs.to(device)
    labels = labels.to(device)
    preds = model(inputs)
    loss = criterion(preds, labels)
    loss.backward()
    optimizer.step()

    class_preds = torch.argmax(preds, dim=1)

    conf_batch = confusion_matrix(labels.detach().cpu().numpy(), class_preds.detach().cpu().numpy(), labels=[0,1])
    train_conf_matrix[:][:] += conf_batch


    return loss.item()


def val_step(model, criterion, inputs, labels, device, val_conf_matrix):
    """ Validates on a batch of data.
    
    Performs validation on a batch of data: 
    make predictions, store metrics for this step.

    Parameters:
    -----------
    model: torch.nn.Module object
        ML model in training.
    criterion: torch.nn.Loss object
        Loss used in training.
    inputs: torch.Tensor
        Batch of spectrograms.
    labels: torch.Tensor
        Batch of class labels.
    device: string
        Device to be used, "cuda" or "cpu".
    val_conf_matrix: pandas DataFrame
        confusion matrix for this epoch.
    Returns:
    -------
    val_loss: float
        loss for this batch of data.
    
    """
    
    with torch.no_grad():
        inputs = inputs.to(device)
        labels = labels.to(device)
        preds = model(inputs)
        loss = criterion(preds, labels)

    class_preds = torch.argmax(preds, dim=1)

    conf_batch = confusion_matrix(labels.detach().cpu().numpy(), class_preds.detach().cpu().numpy(), labels=[0,1])
    val_conf_matrix[:][:] += conf_batch

    return loss.item()

@hydra.main(config_path="conf", config_name="config")
def setup_and_train(cfg):
    """Setup, train, and validate model."""

    torch.manual_seed(cfg.hp.rng_seed)
    np.random.seed(cfg.hp.rng_seed)

    # Dataset & Dataloader for train and validation
    if cfg.data.mode == "spectrograms":
        k_datasets = {x: SpectrogramDataset(cfg, phase=x) for x in ['train', 'val', 'test']}
    else:
        k_datasets = {x: SeismogramDataset(cfg, phase=x) for x in ['train', 'val', 'test']}
    

    k_dataloaders = {x: torch.utils.data.DataLoader(k_datasets[x], batch_size=cfg.hp.batch_size, shuffle=True) for x in ['train', 'val', 'test']}

    dataset_sizes = {x: len(k_datasets[x]) for x in ['train', 'val', 'test']}
    class_counts = k_datasets["train"].class_counts 
    tot_cls = np.sum(class_counts)  
    class_dict = k_datasets["train"].class_dict
    num_classes = len(class_counts)
    class_counts_val = k_datasets["val"].class_counts 
    tot_cls_val = np.sum(class_counts_val)  
    
    # Define net
    if cfg.data.mode == "spectrograms":
        model = PretrainedCNN(num_classes, class_counts, init_bias=False, arch=cfg.data.model, pretrained=cfg.data.pretrained)
    else: 
        if cfg.data.model == "cnn1":
            model = cnn1(num_classes)
        else:
            raise ValueError("Model not implemented")

    # Check device
    if cfg.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    model = model.to(device)
    
    # optimizer Adam with default params
    optimizer = optim.Adam(model.parameters(), lr = cfg.hp.lr, weight_decay=cfg.hp.l2reg)

    if cfg.hp.loss_weights:
        weights = torch.Tensor([tot_cls/class_count for class_count in class_counts]).to(device)
    else:
        weights = torch.Tensor([1., 1.]).to(device)

    criterion_train = torch.nn.CrossEntropyLoss(weight=weights)
    criterion_val = torch.nn.CrossEntropyLoss(reduction="sum")


    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    # scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.hp.scheduler.step_size, gamma=cfg.hp.scheduler.gamma)

    # Create logger and start saving info
    print(cfg, flush=True)
    logger = Logger(cfg)
    logger.log_weights(weights)
    logger.log_model(model, k_datasets["train"])

    # Start metrics
    train_loss = 0
    val_loss = 0
    species = [class_dict[i] for i in range(num_classes)]
    val_conf_matrix = pd.DataFrame(data = np.zeros((num_classes, num_classes)), columns = species)
    train_conf_matrix = pd.DataFrame(data = np.zeros((num_classes, num_classes)), columns = species)
    test_conf_matrix = pd.DataFrame(data = np.zeros((num_classes, num_classes)), columns = species)
    val_conf_matrix.index = val_conf_matrix.columns
    train_conf_matrix.index = train_conf_matrix.columns
    best_val_conf_matrix = copy.deepcopy(val_conf_matrix)
    best_model_weights = copy.deepcopy(model.state_dict())
    early_stopping = EarlyStopping(patience=cfg.hp.patience)

    # Output untrained model metrics
    model.eval()
    for inputs, labels, _ in k_dataloaders["train"]:
        # val_step is used because we just want metrics of random init model.
        batch_loss = val_step(model, criterion_train, inputs, labels, device, train_conf_matrix)
        train_loss += batch_loss * cfg.hp.batch_size
        
    for inputs, labels, _ in k_dataloaders["val"]:
        batch_loss = val_step(model, criterion_val, inputs, labels, device, val_conf_matrix)
        val_loss += batch_loss 

    train_loss = train_loss/dataset_sizes["train"]
    val_loss = val_loss/dataset_sizes["val"]
    logger.log(train_loss, val_loss, train_conf_matrix, val_conf_matrix, 0)

    print("Epoch {}/{}, train loss = {}, valloss = {}".format(0, cfg.hp.epochs, train_loss, val_loss), flush=True)
    print(val_conf_matrix, flush=True)
    print("\n", flush=True)
    
    # Reset metrics
    val_loss = 0
    train_loss = 0
    val_conf_matrix[:][:] = np.zeros((num_classes, num_classes))
    train_conf_matrix[:][:] = np.zeros((num_classes, num_classes))
    for epoch in range(1,cfg.hp.epochs+1):

        model.train()
        for inputs, labels, _ in k_dataloaders["train"]:
            batch_loss = train_step(model, optimizer, criterion_train, inputs, labels, device, train_conf_matrix)
            train_loss += batch_loss * cfg.hp.batch_size

        
        model.eval()
        for inputs, labels, _ in k_dataloaders["val"]:
            batch_loss = val_step(model, criterion_val, inputs, labels, device, val_conf_matrix)
            val_loss += batch_loss 

   
        # End of epoch callbacks
        scheduler.step()
        best_val_conf_matrix, best_model_weights = update_best_model(val_conf_matrix, best_val_conf_matrix, model, best_model_weights)
        early_stopping.check_early_stopping(val_conf_matrix)
        train_loss = train_loss/dataset_sizes["train"]
        val_loss = val_loss/dataset_sizes["val"]
        logger.log(train_loss, val_loss, train_conf_matrix, val_conf_matrix, epoch)

        print("Epoch {}/{}, train loss = {}, valloss = {}".format(epoch, cfg.hp.epochs, train_loss, val_loss), flush=True)
        print(val_conf_matrix, flush=True)
        print("\n", flush=True)
        
        # If early stopping triggered, end training
        if early_stopping.stop:
            break

        # Reset metrics
        val_loss = 0
        train_loss = 0
        val_conf_matrix[:][:] = np.zeros((num_classes, num_classes))
        train_conf_matrix[:][:] = np.zeros((num_classes, num_classes))


    # End of training callbacks
    print("Best validation confusion matrix: \n", flush=True)
    print(best_val_conf_matrix, flush=True)

    model.load_state_dict(best_model_weights) # load best weights before last metrics
    
    # Run on test set
    for inputs, labels, _ in k_dataloaders["test"]:
        
        _ = val_step(model, criterion, inputs, labels, device, test_conf_matrix)

    print("Test confusion matrix from best val model: \n", flush=True)
    print(test_conf_matrix, flush=True)

    #save_best_model(model, best_model_weights, cfg)    
    loader_val_noshuffle = torch.utils.data.DataLoader(k_datasets["val"], batch_size=cfg.hp.batch_size, shuffle=False) 
    loader_train_noshuffle = torch.utils.data.DataLoader(k_datasets["train"], batch_size=cfg.hp.batch_size, shuffle=False) 
    loader_test_noshuffle = torch.utils.data.DataLoader(k_datasets["test"], batch_size=cfg.hp.batch_size, shuffle=False) 
    _, _ = logger.get_all_labels_probas(model, loader_val_noshuffle, device, phase="val", save=True)
    _, _ = logger.get_all_labels_probas(model, loader_train_noshuffle, device, phase="train", save=True)
    _, _ = logger.get_all_labels_probas(model, loader_test_noshuffle, device, phase="test", save=True)

if __name__ == "__main__":
    setup_and_train()
