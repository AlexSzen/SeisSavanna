#!usr/bin/env python3

""" Generate spectrograms from netcdf dataset of chunks.

    In generate_dbase.py, we create a dataset of seismic
    chunks corresponding to animal sightings. In this
    script, we generate the corresponding (mel) spectrograms
    to cast the problem as computer vision. In environmental
    sound classification, using mel-spectrograms with models
    pretrained on ImageNet achieves SOTA, so maybe it can
    work on seismic data too.

---
NOTE: Before running the script, edit inputs/params_generate_spectrogram.py
---

This script contains the following functions:
    * create_nc_dataset: create an empty netcdf file
    and defines all the required variables.
    * fill_nc_dataset: fills the previously defined
    dataset with spectrograms.
    * compute_spectrogram: compute seismograms of a 
    seismic chunk.
 
"""

import numpy as np
import netCDF4 as nc
import os
import sys
import scipy
import librosa
from PIL import Image
from .inputs.params_generate_spectrograms import InputParams
from .helpers import load_obj

def create_nc_dataset():
    """Create dataset which will contain spectrograms.
        
    Create new spectrogram netcdf dataset using the
    dataset with seismic chunks.

    Parameters
    ----------
    None

    Returns
    -------
    None
    
    """

    print("Create dataset file. \n")

    # Load dataset with seismic chunks.
    nc_dset_name = InputParams["DSET_NAME"]+"_chunks_clean.nc"
    chunk_dset = nc.Dataset(os.path.join(InputParams["DSET_DIR"], InputParams["DSET_NAME"], nc_dset_name), 'r')
    
    # Create new dataset for spectrograms.
    nc_dset_name = InputParams["DSET_NAME"]+"_spectrograms.nc"
    dset = nc.Dataset(os.path.join(InputParams["DSET_DIR"], InputParams["DSET_NAME"], nc_dset_name), 'w')
    
    path = os.path.join(InputParams["DSET_DIR"], InputParams["DSET_NAME"], "class_dict.pkl")
    class_dict = load_obj(path)
    class_group = dset.createGroup('class_dict')
    for k,v in class_dict.items():
        setattr(class_group, k, v)
    

    # Define dimensions. For now dimensions of image are fixed
    # to 128x128, but can make it more flexible later.
    dim1 = dset.createDimension('side', 128)
    dim2 = dset.createDimension('traces', chunk_dset["chunk"].shape[0])
    # comp dimension is 3 to match imagenet models.
    dim3 = dset.createDimension("comp", 3)
    if InputParams["get_images"]:
        dim4 = dset.createDimension("im_side", chunk_dset["image"].shape[1])
    
    var1 = dset.createVariable('mel_spectr', 'f4', ("traces", "side", "side", "comp"))
    var2 = dset.createVariable('class', 'i4', ("traces"))
    var3 = dset.createVariable('distance', 'f4', ("traces"))
    var4 = dset.createVariable('seis', 'i8', ("traces"))
    if InputParams["get_images"]:
        var5 = dset.createVariable("image", 'i4', ("traces", "im_side", "im_side", "comp"))
    chunk_dset.close()
    dset.close()

    print("Done. \n")

def fill_nc_dataset():
    """Fill netcdf dataset with spectrograms.
    
    Compute spectrograms for each chunk and fill
    the new dataset with them.

    Parameters
    ----------
    None

    Returns
    -------
    None
    
    """
    
    print("Fill dataset with spectrograms. \n")

    # Load dataset with seismic chunks.
    nc_dset_name = InputParams["DSET_NAME"]+"_chunks_clean.nc"
    chunk_dset = nc.Dataset(os.path.join(InputParams["DSET_DIR"], InputParams["DSET_NAME"], nc_dset_name), 'r')
 
    # Fill new dataset with spectrograms.
    nc_dset_name = InputParams["DSET_NAME"]+"_spectrograms.nc"
    dset = nc.Dataset(os.path.join(InputParams["DSET_DIR"], InputParams["DSET_NAME"], nc_dset_name), 'a')
    dset["class"][:] = chunk_dset["class"][:]
    dset["distance"][:] = chunk_dset["distance"][:]
    dset["seis"][:] = chunk_dset["seis"][:]
    side = dset.dimensions["side"].size
    dset_size = len(dset["class"][:])


    for i_chunk, chunk in enumerate(chunk_dset["chunk"]):
        
        # Compute spectrograms on Z component only.
        ms1, ms2, ms3 = compute_spectrogram(chunk)
        
        # Apply log following papers and resize image.
        resized_log_ms1 = np.array(Image.fromarray(np.log(ms1)).resize(size=(side,side)))
        resized_log_ms2 = np.array(Image.fromarray(np.log(ms2)).resize(size=(side,side)))
        resized_log_ms3 = np.array(Image.fromarray(np.log(ms3)).resize(size=(side,side)))
        dset["mel_spectr"][i_chunk, :, :, 0] = resized_log_ms1
        dset["mel_spectr"][i_chunk, :, :, 1] = resized_log_ms2
        dset["mel_spectr"][i_chunk, :, :, 2] = resized_log_ms3
        
        if InputParams["get_images"]:
            dset["image"][i_chunk, :, :, :] = chunk_dset["image"][i_chunk, :, :, :] 
    
        if ((i_chunk+1) % int(dset_size/10)) == 0:
            print("{}% done".format(round((i_chunk+1)/dset_size*100)))

    
      
    chunk_dset.close()
    dset.close()      

    print("Done.")


def compute_spectrogram(chunk):
    """Compute mel-spectrogram for a seismic chunk.
    
    Given a seismic chunk, compute mel-spectrogram with
    parameters defined in params_generate_spectrograms.py.
    
    """

    f, t, s1 = scipy.signal.spectrogram(chunk[0], fs=InputParams["fs"], nperseg=InputParams["windows"][0],
            noverlap=InputParams["overlaps"][0], nfft=InputParams["nfft"])     
    f, t, s2 = scipy.signal.spectrogram(chunk[1], fs=InputParams["fs"], nperseg=InputParams["windows"][1],
            noverlap=InputParams["overlaps"][1], nfft=InputParams["nfft"])     
    f, t, s3 = scipy.signal.spectrogram(chunk[2], fs=InputParams["fs"], nperseg=InputParams["windows"][2],
            noverlap=InputParams["overlaps"][2], nfft=InputParams["nfft"])

    ms1 = librosa.feature.melspectrogram(sr=InputParams["fs"], S=s1)
    ms2 = librosa.feature.melspectrogram(sr=InputParams["fs"], S=s2)
    ms3 = librosa.feature.melspectrogram(sr=InputParams["fs"], S=s3)

    #return s1, s2, s3
    return ms1, ms2, ms3



if __name__ == "__main__":

    create_nc_dataset()

    fill_nc_dataset()

