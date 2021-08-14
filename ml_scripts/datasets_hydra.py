#!/usr/bin/env python3

"""Module implementing datasets of spectrograms and seismograms for Kenya project."""

import librosa
import numpy as np
from PIL import Image
import netCDF4 as nc
import os
import torch
import scipy
from torch.utils.data import Dataset

class SpectrogramDataset(Dataset):
    """Pytorch implementation of kenya spectrogram dataset."""

    def __init__(self, cfg, phase):
        """Loads dataset, indices, and normalisation. 
        
        Parameters:
        ----------
        cfg: ConfigDict
            Hydra config dictionary for this run.
        phase: str
            "train" or "val" phase
        """
        self.load_images = cfg.load_images
        # Open netcdf dataset.
        self.dset = nc.Dataset(os.path.join(cfg.data.data_dir, cfg.data.dset_base, cfg.data.dset_name), 'r')

        # Load indices corresponding to the correct phase.
        self.inds = np.load(os.path.join(cfg.data.data_dir, cfg.data.dset_base, "inds_%s_%s.npy"%(phase, cfg.data.dset_split)))

        # Load normalisation.
        self.means = np.load(os.path.join(cfg.data.data_dir, cfg.data.dset_base, "mean_spectr_{}.npy".format(cfg.data.dset_split)))
        self.stds = np.load(os.path.join(cfg.data.data_dir, cfg.data.dset_base, "std_spectr_{}.npy".format(cfg.data.dset_split)))

        # Class characteristics
        _, self.class_counts_all = np.unique(self.dset["class"][self.inds], return_counts=True)
        self.binary = cfg.data.binary

        # Make a class -> species mapping
        self.class_dict_all = {}
        for species in self.dset.groups["class_dict"].ncattrs():
            self.class_dict_all[self.dset.groups["class_dict"].getncattr(species)] = species

        # Create real class counts and mapping depending on binary clf or all species
        self.class_counts = []
        self.class_dict = {}
        if self.binary:
            cls_el = self.dset.groups["class_dict"].getncattr("ELEPHANTIDAE")
            self.class_counts.append(self.class_counts_all[cls_el])
            non_el_class_counts = [count for i,count in enumerate(self.class_counts_all) if i!=cls_el]
            self.class_counts.append(np.sum(non_el_class_counts))
            self.class_dict[0] = "ELEPHANTIDAE"
            self.class_dict[1] = "OTHERS"
        else:
            self.class_counts = self.class_counts_all
            self.class_dict = self.class_dict_all


    def __len__(self):

        return len(self.inds)

    def __getitem__(self, index):

        ind = self.inds[index]
        
        cls = int(self.dset["class"][ind])
        spectr = (self.dset["mel_spectr"][ind] - self.means)/self.stds
        spectr = np.transpose(spectr, (2,0,1)) 
        

        if self.load_images:
            img = self.dset["image"][ind] 
        else:
            img = 0.

        # Elephant versus rest mode.
        if self.binary:
            if self.class_dict_all[cls] == "ELEPHANTIDAE":
                cls = 0
            else:
                cls = 1

        return torch.from_numpy(spectr), torch.from_numpy(np.array(cls)), np.array(img)

class SeismogramDataset(Dataset):
    """Dataset for seismograms."""

    def __init__(self, cfg, phase="train"):
        """ Loads dataset, indices, and normalisation.
        
        Parameters
        ----------
        cfg: ConfigDict
            Hydra config dictionary for this run.
        phase: str
            "train" or "val"
        Returns
        -------
        None
        """

        self.load_images = cfg.load_images

        # Load netcdf file
        self.dset = nc.Dataset(os.path.join(cfg.data.data_dir, cfg.data.dset_base, cfg.data.dset_name), 'r')
        
        # Load indices corresponding to the correct phase.
        self.inds = np.load(os.path.join(cfg.data.data_dir, cfg.data.dset_base, "inds_%s_%s.npy"%(phase, cfg.data.dset_split)))

        # Load normalisation.
        self.inmean = np.load(os.path.join(cfg.data.data_dir, cfg.data.dset_base, "mean_seis_{}.npy".format(cfg.data.dset_split)))
        self.instd = np.load(os.path.join(cfg.data.data_dir, cfg.data.dset_base, "std_seis_{}.npy".format(cfg.data.dset_split)))

        # Class characteristics
        _, self.class_counts_all = np.unique(self.dset["class"][self.inds], return_counts=True)
        self.binary = cfg.data.binary

        # Make a class -> species mapping
        self.class_dict_all = {}
        for species in self.dset.groups["class_dict"].ncattrs():
            self.class_dict_all[self.dset.groups["class_dict"].getncattr(species)] = species

        # Create real class counts and mapping depending on binary clf or all species
        self.class_counts = []
        self.class_dict = {}
        if self.binary:
            cls_el = self.dset.groups["class_dict"].getncattr("ELEPHANTIDAE")
            self.class_counts.append(self.class_counts_all[cls_el])
            non_el_class_counts = [count for i,count in enumerate(self.class_counts_all) if i!=cls_el]
            self.class_counts.append(np.sum(non_el_class_counts))
            self.class_dict[0] = "ELEPHANTIDAE"
            self.class_dict[1] = "OTHERS"
        else:
            self.class_counts = self.class_counts_all
            self.class_dict = self.class_dict_all

        # Data augmentation by stacking two events
        self.augment_events = cfg.hp.aug_stack
        cls_el = self.dset.groups["class_dict"].getncattr("ELEPHANTIDAE")
        self.inds_el = np.where(self.dset["class"]==cls_el)[0]
 
   
    def __len__(self):
        """ Gives size of dataset"""
        return len(self.inds)
    
    def __getitem__(self, index):
        """Returns a set of inputs and corresponding outputs"""

        ind = self.inds[index]
        cls = int(self.dset["class"][ind])
        traces = self.dset["chunk"][ind,:,:]
        traces[0,:] = (traces[0,:] - self.inmean[0])/self.instd[0]
        traces[1,:] = (traces[1,:] - self.inmean[1])/self.instd[1]
        traces[2,:] = (traces[2,:] - self.inmean[2])/self.instd[2] 


        if self.load_images:
            img = self.dset["image"][ind] 
        else:
            img = 0.


        # Elephant versus rest mode.
        if self.binary:
            if self.class_dict_all[cls] == "ELEPHANTIDAE":
                cls = 0
            else:
                cls = 1

        return torch.from_numpy(traces), torch.from_numpy(np.array(cls)), np.array(img)


