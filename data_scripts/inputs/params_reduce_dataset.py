#!/usr/bin/env python3

""" Script containing inputs for reduce_dataset.py"""

import numpy as np


InputParams = {}

### Dataset directories
InputParams["LONG_DIR"] = "/data/dset_allspec_40"
InputParams["NEW_DIR"] = "/data/dset_allspec_20"

### Which nc files to go through
InputParams["DSET_NAMES"] = ["chunks_clean.nc","spectrograms.nc"]

### Max distance for new dataset
InputParams["DISTANCE"] = 20

