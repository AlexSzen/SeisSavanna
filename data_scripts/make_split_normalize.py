#!/usr/bin/env python3

""" split into train val test and compute normalisation on training set

This script produces numpy arrays with indices for train, validation
and test sets. It also produces numpy arrays with the mean and standard
deviation computed over the training set, to be used as normalisation
during training.

----
NOTE: Change inputs in inputs/inputs_split_norm.py
----

It contains the following functions:
    *split: produces numpy arrays with splits indices

    *normalize: produces numpy arrays with normalization quantities 
    over training set

"""

import sys
import numpy as np
from netCDF4 import Dataset
import os
import json
from .inputs.params_split_norm import InputParams
from .helpers import load_obj

def split_by_station_time():
    """ Create indices for train val test based on InputParams 
    
    Create train/val/test indices based on seismic stations and time. 
    This is same as normal time split but using only a subset of stations.
    We try this because some of the stations (east and west triangles)
    generate a lot of duplicated examples.
    """

    dbase = Dataset(os.path.join(InputParams["DATA_PATH"], InputParams["DBASE_NAME"]), 'r')
    dbase_size = dbase.dimensions["traces"].size
    indices = np.asarray(range(dbase_size))
    
    path = os.path.join(InputParams["DATA_PATH"], "seis_dict.pkl")
    seis_dict = load_obj(path)
    
    inds_test = []
    inds_val = []
    inds_train = []
    
    seis_subset = [seis_dict[seis] for seis in InputParams["stations_subset"]]
    
    inds_seis = np.array([ind for ind in indices if dbase["seis"][ind] in seis_subset])
    sub_dbase = dbase["class"][inds_seis]
    classes = np.unique(sub_dbase)
    inds_test = []
    inds_val = []
    inds_train = []

    end_test = int(InputParams["TEST_SIZE"]*len(inds_seis))
    end_val = end_test + int(InputParams["VAL_SIZE"]*len(inds_seis))


    test = inds_seis[:end_test]
    val = inds_seis[end_test:end_val]
    train = inds_seis[end_val:]

    if 'spect' in InputParams['DBASE_NAME']:
        # Compute mean and stds of spectrograms
        means = np.mean(dbase["mel_spectr"][train], axis=(0, 1, 2))
        stds = np.std(dbase["mel_spectr"][train], axis=(0, 1, 2))

        print("Means: " + str(means))
        print("Stds: " + str(stds))
        np.save(os.path.join(InputParams["DATA_PATH"], "mean_spectr_station_time.npy"), means.data)
        np.save(os.path.join(InputParams["DATA_PATH"], "std_spectr_station_time.npy"), stds.data)

    else:
        # Compute means and std of seismograms
        means = np.mean(dbase["chunk"][train], axis=(0,2))
        stds = np.std(dbase["chunk"][train], axis=(0,2))

        print("Means: " +str(means))
        print("Stds: "+ str(stds))

        np.save(os.path.join(InputParams["DATA_PATH"], "mean_seis_station_time.npy"), means.data)
        np.save(os.path.join(InputParams["DATA_PATH"], "std_seis_station_time.npy"), stds.data)

    np.save(os.path.join(InputParams["DATA_PATH"], "inds_train_station_time.npy"), train)
    np.save(os.path.join(InputParams["DATA_PATH"], "inds_val_station_time.npy"), val)
    np.save(os.path.join(InputParams["DATA_PATH"], "inds_test_station_time.npy"), test)

if __name__=="__main__":

    print("Start computing splits and normalization...\n")

    split_by_station_time()

    # dump split info 
    with open(os.path.join(InputParams["DATA_PATH"], "split_info.txt"), "w") as f:
        f.write(json.dumps(InputParams))

    print("Finished.")


