#!/usr/bin/env python3

""" Inputs for generate_spectrograms.py

This script contains all the necessary inputs 
for generate_spectrograms.py, such as data_directories,
spectrograms parameters, etc.

"""


InputParams = {}


### Data directory
InputParams["DSET_DIR"] = "/data"

### Name for this dataset
InputParams["DSET_NAME"] = "dset_allspec_200_master"

### Params for mel-spectrograms
### We generate three mel-spectrograms
### for 3 RGB channels of imagenet models.
### Hence 3 values for parameters.

# Sampling frequency, has to match the one used
# to generate chunks.
InputParams["fs"] = 200.

# Sizes of windows for STFT
fs = InputParams["fs"]
InputParams["windows"] = [int(fs*0.2), int(fs*0.2), int(fs*0.2)]

# Padding for STFT
InputParams["nfft"] = 400

# Overlaps between windows.
InputParams["overlaps"] = [InputParams["windows"][0]-1, InputParams["windows"][1]-1, InputParams["windows"][2]-1]

InputParams["get_images"] = True
