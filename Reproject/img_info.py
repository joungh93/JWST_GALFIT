#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:45:04 2023

@author: jlee
"""


import time
start_time = time.time()

import numpy as np
import glob, os
from astropy.io import fits


# ----- Directories & Files ----- #
dir_root = os.path.abspath("../")
print(dir_root)
dir_Img = dir_root+"/Images/"
img_sci = np.array(sorted(glob.glob(dir_Img+"*_sci.fits")))


# ----- Wavelength sorting ----- #
wav_eff = []
for i in range(len(img_sci)):
    hh = fits.getheader(img_sci[i])
    wav_eff.append(hh['PHOTPLAM'])
idx_wav = np.argsort(np.array(wav_eff))


# ----- Writing infomation file ----- #
f = open("info.txt", "w")
for i in np.argsort(wav_eff):
    dd, hh = fits.getdata(img_sci[i], header=True)
    imgname = img_sci[i].split('/')[-1]
    try:
        pxs = 0.5 * (np.sqrt(hh['CD1_1']**2. + hh['CD1_2']**2.) + np.sqrt(hh['CD2_1']**2. + hh['CD2_2']**2.)) * 3600.
    except:
        pxs = 0.5 * (np.sqrt(hh['CD1_1']**2.) + np.sqrt(hh['CD2_2']**2.)) * 3600.
    if (hh['INSTRUME'] == 'ACS'):
        if (hh['FILTER1'][0] == 'F'):
            filt = hh['FILTER1']
        else:
            filt = hh['FILTER2']
    else:
        filt = hh['FILTER']
    info  = f"{imgname:<55s}  {hh['TELESCOP']:<5s}  {hh['INSTRUME']:<7s}  {hh['DETECTOR']:<9s}  "
    info += f"{filt:5s}  {hh['EXPTIME']:>8.0f} sec   {hh['NAXIS1']:5d} x {hh['NAXIS2']:5d}   "
    info += f"{pxs:.2f} arcsec/pixel   {np.sum(dd != 0.)*pxs**2. / 3600.:>5.2f} arcmin2"
    print(info)
    f.write(info+"\n")
f.close()


# Printing the running time
print('--- %.4f seconds ---' %(time.time()-start_time))
