#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:29:04 2022
@author: jlee
"""


import time

import numpy as np
import glob, os, copy
from matplotlib import pyplot as plt
# import pandas as pd
from astropy.io import fits
from astropy.visualization import ZScaleInterval
# from astropy.nddata import Cutout2D
# from astropy import wcs

import warnings
warnings.filterwarnings("ignore")

# from scipy.special import gamma
# def bn(n):
#     return 1.9992*n - 0.3271
# def fn(n):
#     b = bn(n)
#     return (n*np.exp(b)/b**(2*n))*gamma(2*n)

import init_settings as init


# ----- Step 4. Making the GALFIT script ----- #
def run_galfit(galaxy_id, band, rth=40, mask_prefix="Mask",
               dir_input="input", dir_output="output1", edge_pix=5,
               n_sersic=1.5):
#     for i, img in enumerate(imglist):
#         d_sep = np.genfromtxt("SExtractor/"+num[i]+".cat", dtype=None, encoding='ascii',
#                              names=('x','y','num','mag_auto','e_mag_auto','kron','petro','bgr',
#                                     'ra','dec','cxx','cyy','cxy','a','b','theta','mu0',
#                                     'mut','flag','fwhm','flxrad','cl'))
#         obj_aper = d_sep['mag_auto'].argmin()
    d_sep = init.dat[band]

    if (dir_input[-1] == "/"):
        dir_input = dir_input[:-1]
    if (dir_output[-1] == "/"):
        dir_output = dir_output[:-1]
    dir_root = os.getcwd()

    f = open(dir_input+f"/{galaxy_id:05d}.feedme", "w")

    f.write(f"A) Input_{galaxy_id:05d}_"+band+".fits\n")  # Input data image
    f.write("B) "+dir_root+"/"+dir_output+f"/Block_{galaxy_id:05d}.fits\n")  # Output data image block
    f.write("C) none\n")  # Sigma image name
    f.write("D) PSF_"+band+".fits\n")  # Input PSF image
    f.write("E) 1\n")  # PSF find sampling factor relative to data
    f.write("F) "+mask_prefix+f"_{galaxy_id:05d}.fits\n")  # Bad pixel mask
    f.write("G) none\n")  # File with parameter constraints
    f.write(f"H) {edge_pix:d} {2*rth-edge_pix:d} {edge_pix:d} {2*rth-edge_pix:d}\n")  # Image region to fit
    f.write(f"I) {3*rth:d} {3*rth:d}\n")  # Size of the convolution box (x y)
    f.write(f"J) {init.magAB_zero:.4f}\n")  # Magnitude photometric zeropoint
    f.write(f"K) {init.pixel_scale:.3f} {init.pixel_scale:.3f}\n")  # Plate scale (dx dy) [arcsec per pixel]
    f.write("O) regular\n")  # Display type (regular, curses, both)
    f.write("P) 0\n\n")  # Options (0 = normal run; 1,2 = make model/imgblock & quit)

    f.write("0) sersic\n")  # Object type
    f.write(f"1) {init.rth:.2f} {init.rth:.2f} 1 1\n")  # Position (x, y)
    f.write(f"3) {d_sep.iloc[galaxy_id-1]['mag']:.3f} 1\n")  # Total magnitude
    f.write(f"4) {d_sep.iloc[galaxy_id-1]['flxrad']:.2f} 1\n")  # R_e
    f.write(f"5) {n_sersic:.1f} 1\n")  # Sersic exponent
    f.write(f"9) {d_sep.iloc[galaxy_id-1]['b']/d_sep.iloc[galaxy_id-1]['a']:.3f} 1\n")  # Axis ratio (b/a)
    f.write(f"10) {d_sep.iloc[galaxy_id-1]['theta']-90.0:.2f} 1\n")  # Position angle (PA)
    f.write("Z) 0\n\n")  # Skip this model in output image? (yes=1, no=0)

    f.write("0) sky\n")
    f.write("1) 0.00 1\n")  # Sky background
    f.write("2) 0.00 1\n")  # dsky/dx
    f.write("3) 0.00 1\n")  # dsky/dy
    f.write("Z) 0 1\n\n")  # Skip this model in output image?

    f.close()

    os.chdir(dir_input)
    os.system(f"galfit {galaxy_id:05d}.feedme")
    os.system("mv -v fit.log "+dir_root+"/"+dir_output+f"/Result_{galaxy_id:05d}.log")
    os.system("rm -rfv galfit.*")
    os.chdir(init.current_dir)


start_time = time.time()

nS = [1.5, 1.0, 0.5]
for n, dir_output in enumerate(["output1/", "output2/", "output3/"]):
    # dir_output = "../output2/"
    if (glob.glob(dir_output) == []):
        os.system("mkdir "+dir_output)

    for i, gid in enumerate(np.hstack([init.id_z1, init.id_z2, init.id_z3])):
        if (i < init.n_group_z1):
            j = 0
        elif ((i >= init.n_group_z1) & (i < init.n_group_z1+init.n_group_z2)):
            j = 1
        else:
            j = 2
    #     j = i // n_group
        run_galfit(gid, init.band[j], rth=init.rth, mask_prefix="Mask",
                   dir_input="input", dir_output=dir_output,
                   n_sersic=nS[n])

# Printing the running time
print(f"--- {time.time()-start_time:.4f} seconds ---")
