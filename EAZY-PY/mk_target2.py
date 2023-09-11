### Imports

# Printing the versions of packages
from importlib_metadata import version
for pkg in ['numpy', 'matplotlib', 'pandas']:
    print(pkg+": ver "+version(pkg))

# importing necessary modules
import numpy as np
import glob, os, copy
import pandas as pd
from astropy.io import fits
import pickle

import warnings
warnings.filterwarnings("ignore")

c = 2.99792e+5    # km/s


# # ----- Loading the photometry data ----- #
# dir_phot = "../Phot/"

# # load data
# with open(dir_phot+"phot_data.pickle", 'rb') as fr:
#     phot_data = pickle.load(fr)
 
 
# ----- Read the catalog ----- #
dir_output = "EAZY_OUTPUT/"
run_mode = "run5z"
dz, hz = fits.getdata(dir_output+run_mode+".eazypy.zout.fits", ext=1, header=True)


# ----- Reading the region file ----- #
with open("target2.reg", "r") as f:
    ll = f.readlines()
id2 = np.array([int(l.split()[-4].split(",")[0][8:]) for l in ll[3:]])


# ----- Writing results ----- #
with open("target2.csv", "w") as f:
    f.write("ID,z,z_flag,z_phot_chi2,logmass,restU,restB,restV,restJ\n")
    for i in range(len(id2)):
        id_cnd = (dz['id'] == id2[i])
        if (dz['z_spec'][id_cnd][0] > 0.):
            z_flag = "spec"
        else:
            z_flag = "phot"
        f.write(f"{dz['id'][id_cnd][0]:>7d},{dz['z_phot'][id_cnd][0]:>7.4f},"+z_flag+",")
        f.write(f"{dz['z_phot_chi2'][id_cnd][0]:>.3e},{np.log10(dz['mass'][id_cnd][0]):>7.4f},")
        f.write(f"{dz['restU'][id_cnd][0]:>7.4f},{dz['restB'][id_cnd][0]:>7.4f},")
        f.write(f"{dz['restV'][id_cnd][0]:>7.4f},{dz['restJ'][id_cnd][0]:>7.4f}\n")
