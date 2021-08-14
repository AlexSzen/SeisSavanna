#/usr/bin/env python3

"""Create kenya dataset of shorter distance from dataset of larger distance."""

import os
import sys
import numpy as np
import pandas as pd
import netCDF4 as nc
from .inputs.params_reduce_dataset import InputParams
from .helpers import createFolder


def main():
    createFolder(InputParams["NEW_DIR"])
    for dset_name in InputParams["DSET_NAMES"]:

        dset_base = InputParams["LONG_DIR"].split("/")[-1]
        
        print("Processing {}".format(dset_base+"_"+dset_name))

        os.system("cp {}/*pkl {}".format(InputParams["LONG_DIR"], InputParams["NEW_DIR"]))
        
        long_dset = nc.Dataset(os.path.join(InputParams["LONG_DIR"], dset_base+"_"+dset_name), "r")
        inds_dist = np.where(long_dset["distance"][:]<InputParams["DISTANCE"])[0]
        new_len = len(inds_dist)
        
        dset_base = InputParams["NEW_DIR"].split("/")[-1]
        new_dset = nc.Dataset(os.path.join(InputParams["NEW_DIR"], dset_base+"_"+dset_name), "w")
        new_dset.setncatts(long_dset.__dict__)
       
        for group in long_dset.groups:
            new_group = new_dset.createGroup(group)
            for kattr in long_dset.groups[group].ncattrs():
                vattr = long_dset.groups[group].getncattr(kattr)
                setattr(new_group, kattr, vattr)

        for name, dim in long_dset.dimensions.items():
            new_dset.createDimension(name, len(dim) if not name=="traces" else new_len)

        for name, var in long_dset.variables.items():
            dims = tuple([new_dset.dimensions[dim].name for dim in var.dimensions])
            x = new_dset.createVariable(name, var.datatype, dims)
            new_dset[name].setncatts(long_dset[name].__dict__)
        
        for i, ind in enumerate(inds_dist): 
            for name, var in new_dset.variables.items():    
                new_dset[name][i] = long_dset[name][ind]
            
            if ((i+1)%(int(new_len/10)) == 0):
                print("{} % done.".format(round(100*(i+1)/new_len)))
                
        long_dset.close()
        new_dset.close()

if __name__ == "__main__":
    main()
