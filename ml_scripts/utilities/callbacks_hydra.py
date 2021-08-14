#!/usr/bin/env python3

"""Callbacks functions to be called at various points during training process."""

import os
import numpy as np
import pandas as pd
import copy
import torch
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils import tensorboard
from torchsummary import summary
from .utils import create_folder 

class Logger:
    """Logs different metrics into csv and/or tensorboard"""

    def __init__(self, cfg, log_csv=True, log_tboard=True,
                 metrics=["tp", "fp", "tn", "fn", "avg_acc", "acc", "mcc", "loss", "f1", "tpr", "fpr", "tnr", "precision"]):
        """Create logging objects and output directories."""

        self.log_csv = log_csv
        self.log_tboard = log_tboard
        self.metrics = metrics

        if log_csv:
            self.df_log_train = pd.DataFrame(data=np.zeros((cfg.hp.epochs+1, len(metrics))), columns=metrics)
            self.df_log_val = pd.DataFrame(data=np.zeros((cfg.hp.epochs+1, len(metrics))), columns=metrics)
        if log_tboard:
            train_log_dir = 'tboard/train'
            val_log_dir = 'tboard/val'
            self.train_summary_writer = tensorboard.SummaryWriter(train_log_dir)
            self.val_summary_writer = tensorboard.SummaryWriter(val_log_dir)   
        
        create_folder("figures")
        create_folder(os.path.join("figures", "tp"))
        create_folder(os.path.join("figures", "fp"))
 
    def log(self, train_loss, val_loss, train_conf_matrix, val_conf_matrix, epoch):
        """Given loss and confusion matrices for train and val sets, log metrics."""

        for metric in self.metrics:
            if self.log_csv:
                self.df_log_train[metric][epoch] = self.compute_metric(metric, train_loss, train_conf_matrix)
                self.df_log_val[metric][epoch] = self.compute_metric(metric, val_loss, val_conf_matrix)
            if self.log_tboard:
                self.train_summary_writer.add_scalar("train_"+metric,self.compute_metric(metric, train_loss, train_conf_matrix), epoch)
                self.val_summary_writer.add_scalar("val_"+metric,self.compute_metric(metric, val_loss, val_conf_matrix), epoch)


    def compute_metric(self, metric, loss, conf_matrix):
        """Return metric of interest."""

        tp = conf_matrix.iloc[0,0]
        fp = conf_matrix.iloc[1,0]
        tn = conf_matrix.iloc[1,1]
        fn = conf_matrix.iloc[0,1]
        tpr = tp/(tp+fn)
        tnr = tn/(tn+fp)
        fpr = fp/(fp+tn)
        precision = tp/(tp+fp)
        acc = (tp+tn)/(tp+tn+fp+fn)
        bal_acc = (tpr+tnr)/2
        f1 = 2*tp/(2*tp+fn+fp)
        
        try:
            mcc = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        except RuntimeWarning:
            mcc=0

        if metric=="tp":
            return tp
        elif metric=="fp":
            return fp
        elif metric=="tn":
            return tn
        elif metric=="fn":
            return fn
        elif metric=="tpr":
            return tpr
        elif metric=="tnr":
            return tnr
        elif metric=="fpr":
            return fpr
        elif metric=="precision":
            return precision
        elif metric=="acc":
            return acc
        elif metric=="avg_acc":
            return bal_acc
        elif metric=="f1":
            return f1
        elif metric=="mcc":
            return mcc
        elif metric=="loss":
            return loss
        else:
            raise ValueError("Metric not implemented.")
    
    def plot_roc_pr(self, model, val_dataloader, device):
        """Create roc and pr curves and write to tensorboard, also save arrays."""

        labels, preds = self.get_all_labels_probas(model, val_dataloader, device, phase="val", save=False)

        thresholds = np.linspace(0,1,128)
        tprs = []
        fprs = []
        precs = []
        for thr in thresholds:
            inds_el = np.where(preds[:,0]>thr)[0]
            class_preds = np.ones(len(labels))
            class_preds[inds_el] = 0

            conf = confusion_matrix(labels, class_preds, labels=[0,1])
            tp = conf[0,0]
            fp = conf[1,0]
            tn = conf[1,1]
            fn = conf[0,1]
            tpr = tp/(tp+fn)
            fpr = fp/(fp+tn)
            prec = tp/(tp+fp)
            tprs.append(tpr)
            fprs.append(fpr)
            precs.append(prec)
        
        fig, ax = plt.subplots(figsize = (10,10))
        ax.plot(fprs, tprs)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_title("ROC Curve", fontsize = 20)
        ax.set_xlabel("False Positive Rate", fontsize = 20)
        ax.tick_params(labelsize = 16)
        ax.set_ylabel("True Positive Rate", fontsize = 20)
        ax.grid()

        self.val_summary_writer.add_figure("val_roc_curve", fig)

        fig, ax = plt.subplots(figsize = (10,10))
        ax.plot(tprs, precs)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_title("PR Curve", fontsize = 20)
        ax.set_xlabel("True Positive Rate (Recall)", fontsize = 20)
        ax.tick_params(labelsize = 16)
        ax.set_ylabel("Precision", fontsize = 20)
        ax.grid()

        self.val_summary_writer.add_figure("val_pr_curve", fig)

    def plot_tp_fp(self, model, dataloader_val, device, cfg):
        """Loop through val set and plot some true and false positive data for inspection."""

        num_tp = 0
        num_fp = 0
        tot_images = 100

        model.eval()
        for inputs, labels, images in dataloader_val:
            with torch.no_grad():
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = model(inputs)
                class_preds = torch.argmax(preds, dim=1)
                diff = class_preds - labels
                inputs = inputs.detach().cpu().numpy()
                diff = diff.detach().cpu().numpy()
                inds_fp = np.where(diff != 0)[0]
                inds_tp = np.where(diff == 0)[0]
                
                if len(inds_fp) != 0 and num_fp<tot_images:
                    sample_inds_fp = np.random.choice(inds_fp, min(10, len(inds_fp)), replace=False)
                    
                    for ind in sample_inds_fp:

                        if cfg.data.mode == "spectrograms":
                            fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(20,10))
                            ax[0].imshow(images[ind])
                            ax[1].pcolormesh(inputs[ind, 0, :, :], shading='gouraud')
                            ax[1].set_title("True class: {}, predicted: {}".format(labels[ind], class_preds[ind]), fontsize=16)
                            ax[1].set_xlabel("Time (rescaled)", fontsize = 16)
                            ax[1].tick_params(labelsize = 12)
                            ax[1].set_ylabel("Mel bins", fontsize = 16)
                        else:
                            fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(21,7), gridspec_kw={'width_ratios': [1, 2]})
                            ax[0].imshow(images[ind])
                            ax[1].plot(inputs[ind, 0, :])
                            ax[1].grid()
                            ax[1].set_title("True class: {}, predicted: {}".format(labels[ind], class_preds[ind]), fontsize=16)
                            ax[1].set_xlabel("Time (samples)", fontsize = 16)
                            ax[1].tick_params(labelsize = 12)
                            ax[1].set_ylabel("Normalised amplitude", fontsize = 16)
           
                        plt.savefig(os.path.join("figures", "fp",
                                    "fp_{:03d}".format(num_fp)), bbox_inches="tight")
                        plt.close(fig)
                        num_fp += 1

                if len(inds_tp) != 0 and num_tp<tot_images:
                    sample_inds_tp = np.random.choice(inds_tp, min(3, len(inds_tp)), replace=False)
                    
                    for ind in sample_inds_tp:

                        if cfg.data.mode == "spectrograms":
                            fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(20,10))
                            ax[0].imshow(images[ind])
                            ax[1].pcolormesh(inputs[ind, 0, :, :], shading='gouraud')
                            ax[1].set_title("True class: {}, predicted: {}".format(labels[ind], class_preds[ind]), fontsize=16)
                            ax[1].set_xlabel("Time (rescaled)", fontsize = 16)
                            ax[1].tick_params(labelsize = 12)
                            ax[1].set_ylabel("Mel bins", fontsize = 16)
                        else:
                            fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(21,7), gridspec_kw={'width_ratios': [1, 2]})
                            ax[0].imshow(images[ind])
                            ax[1].plot(inputs[ind, 0, :])
                            ax[1].grid()
                            ax[1].set_title("True class: {}, predicted: {}".format(labels[ind], class_preds[ind]), fontsize=16)
                            ax[1].set_xlabel("Time (samples)", fontsize = 16)
                            ax[1].tick_params(labelsize = 12)
                            ax[1].set_ylabel("Normalised amplitude", fontsize = 16)
           
                        plt.savefig(os.path.join("figures", "tp",
                                    "tp_{:03d}".format(num_tp)), bbox_inches="tight")
                        plt.close(fig)
                        num_tp += 1


                if num_fp>=tot_images and num_tp>=tot_images:
                    return

    def log_weights(self,weights):
        """write class weights to info file"""

        with open("class_weights.txt", "w") as f:
            f.write("Class weights in training set:\n")
            f.write(str(weights.detach().cpu().numpy()))

    def log_model(self, model, dataset):
        """write model summary to file."""

        inp, lab, _ = dataset[0]
        with open("model_summary.txt", "w") as f:
            with redirect_stdout(f):
                print(model)
                print("\n")
                summary(model, inp.shape)

    def get_all_labels_probas(self, model, dataloader, device, phase="val", save=True):
        """loop through val set and return all labels and predictions probas."""

        model.eval()
        tot_labels = []
        tot_preds = []
        for inputs, labels, _ in dataloader:
            with torch.no_grad():
                inputs = inputs.to(device)
                labels = labels.to(device)
                preds = torch.nn.functional.softmax(model(inputs), dim=-1)
                tot_labels.append(labels.detach().cpu().numpy())
                tot_preds.append(preds.detach().cpu().numpy())

        tot_labels = np.concatenate(np.array(tot_labels), axis = 0)
        tot_preds = np.concatenate(np.array(tot_preds), axis = 0)

        if save:
            np.save("{}_all_labels.npy".format(phase), tot_labels)
            np.save("{}_all_probas.npy".format(phase), tot_preds)
        
        return tot_labels, tot_preds

       


    def __del__(self):
        """Save csv file and close tensorboard summaries in destructor"""

        if self.log_csv:
            self.df_log_train.to_csv("train_metrics_log.csv")
            self.df_log_val.to_csv("val_metrics_log.csv")
        if self.log_tboard:
            self.train_summary_writer.close()
            self.val_summary_writer.close()

class EarlyStopping:
    """Early stopping of model if accuracy does not improve."""

    def __init__(self, patience):
        self.stop = False
        self.patience = max(1, patience)
        self.counter = 0
        self.best_accuracy = 0.
    
    def check_early_stopping(self,conf_matrix):
        """Check if accuracy improves."""

        avg_accuracy = np.mean(np.diag(conf_matrix)/np.sum(conf_matrix, axis=1))
        
        if avg_accuracy > self.best_accuracy:
            self.best_accuracy = avg_accuracy
            self.counter = 0
        else:
            self.counter += 1

        if self.counter == self.patience:
            self.stop = True
        
def update_best_model(conf_matrix, best_conf_matrix, model, best_model_weights):
    """If confusion matrix improves, we update the best model weights.
    
    Parameters:
    -----------
    conf_matrix: pandas DataFrame
        current epoch confusion matrix on validation set.
    best_conf_matrix: pandas DataFrame
        best confusion matrix so far.
    model: torch.nn.Module
        current model.
    best_model_weights: dict
        dict with weights of best model so far.

    Returns:
    -------
    None
    
    """

    # Compute accuracy averaged over all classes
    avg_accuracy = np.mean(np.diag(conf_matrix)/np.sum(conf_matrix, axis=1))
    
    
    if np.sum(np.sum(best_conf_matrix)) == 0:
        best_avg_accuracy = 0
    else:    
        best_avg_accuracy = np.mean(np.diag(best_conf_matrix)/np.sum(best_conf_matrix, axis=1))
    
    if avg_accuracy>best_avg_accuracy:
        best_model_weights = copy.deepcopy(model.state_dict())
        best_conf_matrix = copy.deepcopy(conf_matrix)

    return best_conf_matrix, best_model_weights

def save_best_model(model, best_model_weights, cfg):
    """Save best performing model at the end of training.
    
    Parameters:
    -----------
    model: torch.nn.Module
        Current model.
    best_model_weights: dict
        dict with best weights.
    cfg: Config Dict
        Hydra config dict to get output directories.
    Returns:
    -------
    None
    """

    model.load_state_dict(best_model_weights)
    torch.save(model, "best_model.pt")


