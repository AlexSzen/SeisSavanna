#!/usr/bin/env python3

""" Contains helper functions

This script contains miscellaneous helper functions
such as to create directories.

The following functions can be imported from this package:
    * createDir - creates a directory if it doesn't exist yet.

"""

import os
import pickle

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def save_obj(obj, path ):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
