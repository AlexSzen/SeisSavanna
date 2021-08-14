#!/usr/bin/env python3

"""Various utility functions for species classification"""

import os
import numpy as np
from scipy.optimize import fsolve

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


