#!/usr/bin/env python3

""" Script containing inputs for make_split_normalize.py """

InputParams = {}

### Dataset directory
InputParams['DATA_PATH'] = '/data/dset_allspec_40'
### Dataset to use
InputParams["DBASE_NAME"] = "dset_allspec_40_spectrograms.nc"

### Size of test and validation set.
InputParams["VAL_SIZE"] = 0.15
InputParams["TEST_SIZE"] = 0.15


### Split subset of stations by time with splits given by VAL_SIZE and TEST_SIZE
InputParams["stations_subset"] = ["ETA00", "NNL62", "NTA02", "STA02", "NWP05", "WTA00"]


