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

region_name = "tmp/point_picked.reg"
# ds9_opt = "-scalemode zscale -scale lock yes -frame lock image -region "+region_name+" "
# for i in range(n_img):
# 	logname = "tmp/imexam_"+imglist[i].split('.fits')[0]+".log"
# 	os.system("ds9 "+imglist[i]+" "+ds9_opt+" &")
# 	iraf.sleep(5.0)

# 	os.system("rm -rfv "+logname)
# 	iraf.imexamine(logfile = logname, keeplog = "yes")
# 	yn = raw_input("Done? Please close the current DS9 window. (Y/N): ")
# 	if (yn == "Y"):
# 		continue
# 	else:
# 		raise Exception

pxs = 0.063    # arcsec/pixel
fwhms = []
os.system("rm -rfv tmp/fwhms.txt")
for i in range(n_img):
	logname = "tmp/imexam_"+imglist[i].split('.fits')[0]+".log"
	poi = np.genfromtxt(logname, usecols=(0,1,7,9,13), names=('X','Y','SKY','E','MOFFAT'))
	med_fwhm = np.median(poi['MOFFAT'])
	fwhms.append(med_fwhm)
	print("\n"+imglist[i])
	print("Using {0:d} sources".format(len(poi)))
	print("Median FWHMs (Moffat profile): {0:.2f} pix = {1:.3f} arcsec".format(med_fwhm, med_fwhm*pxs))
np.savetxt("tmp/fwhms.txt", fwhms, fmt="%.3f")

