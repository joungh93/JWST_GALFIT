#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 09:56:36 2022
@author: jlee
"""


import numpy as np
import glob, os
from matplotlib import pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import time

import sys
assert sys.version[0] == '2'    # Assert that the python version is 2.

current_dir = os.getcwd()
dir_iraf = "/data/jlee/DATA/JWST/First/SMACS0723/Align/"    # where 'login.cl' file is located

os.chdir(dir_iraf)
from pyraf import iraf
os.chdir(current_dir)


# ----- Displaying the images & Imexamine tasks ----- #
imglist_hst = ["Ah435.fits", "Ah606.fits", "Ah814.fits",
               "Ah105.fits", "Ah125.fits", "Ah140.fits", "Ah160.fits"]
n_hst = len(imglist_hst)

imglist_jwst = ["Aj090.fits", "Aj150.fits", "Aj200.fits",
                "Aj277.fits", "Aj356.fits", "Aj444.fits"]
n_jwst = len(imglist_jwst)

imglist = imglist_hst + imglist_jwst
n_img = n_hst + n_jwst


# Sky, skysigma estimation
# fsky, ftime = 100., 1000.
sky_value, sky_sigma = [], []
for i, img in enumerate(imglist):
    if (i < n_hst):
        dat = fits.getdata(img, ext=0)
    else:
        dat = fits.getdata(img, ext=1)
    # dat2 = fsky +ftime*dat
    avg, med, std = sigma_clipped_stats(dat, sigma=3.0)
    sky_val = 3.0*med - 2.0*avg
    sky_value.append(sky_val)
    sky_sig = std
    sky_sigma.append(sky_sig)
    print(sky_val, sky_sig)
np.savetxt("tmp/sky.txt", [sky_value, sky_sigma], delimiter="  ", fmt="%.3e")


iraf.noao()
iraf.digiphot()
iraf.daophot()

# DAOFIND task
fwhms = np.loadtxt("tmp/fwhms.txt")
sky_value, sky_sigma = np.loadtxt("tmp/sky.txt")

start_time = time.time()
for i, img in enumerate(imglist):
    if (i < n_hst):
        dat = fits.getdata(img, ext=0)
        extn = "[0]"
    else:
        dat = fits.getdata(img, ext=1)
        extn = "[1]"
    coofile = "tmp/"+imglist[i].split(".fits")[0]+".coo"
    os.system("rm -rfv "+coofile)

    kwargs = {"fwhmpsf":fwhms[i], "sigma":sky_sigma[i], "threshold":2.0, "verify":False}
    iraf.daofind(image=imglist[i]+extn, output=coofile, **kwargs)
end_time = time.time()
print("\n--- DAOFIND tasks take {0:.4f} sec ---".format(end_time-start_time))



# PHOT task
start_time = time.time()
for i, img in enumerate(imglist):
    if (i < n_hst):
        dat = fits.getdata(img, ext=0)
        extn = "[0]"
    else:
        dat = fits.getdata(img, ext=1)
        extn = "[1]"
    coofile = "tmp/"+imglist[i].split(".fits")[0]+".coo"
    magfile = "tmp/"+imglist[i].split(".fits")[0]+".mag"
    datfile = "tmp/"+imglist[i].split(".fits")[0]+".dat"
    os.system("rm -rfv "+magfile)
    os.system("rm -rfv "+datfile)

    kwargs = {"fwhmpsf":fwhms[i], "sigma":sky_sigma[i],
              "datamax":1000.0, "readnoise":5.0, "epadu":2.0, "itime":1.0,
              "cbox":1.5*fwhms[i], "annulus":4*fwhms[i], "dannulus":2*fwhms[i],
              "skyvalue":sky_value[i], "apertures":1.0*fwhms[i], "zmag":25.0, "verify":False}
    iraf.phot(image=imglist[i]+extn, coords=coofile, output=magfile, **kwargs)
    iraf.txdump(magfile, "XINIT,YINIT,MAG,MERR,MSKY,STDEV,NSKY,SUM,AREA,FLUX","yes", Stdout=datfile)
end_time = time.time()
print("\n--- PHOT tasks take {0:.4f} sec ---".format(end_time-start_time))



# ap = pd.read_csv("tmp/Aj150.dat", sep='  ', header=None, engine='python', na_values='INDEF',
#                  names=('x','y','mag','merr','msky','stdev','nsky','sum','area','flux'))
# ap.head(10)
