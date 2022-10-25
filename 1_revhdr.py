#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:29:04 2022
@author: jlee
"""


# import time
# start_time = time.time()

import numpy as np
# import glob, os, copy
# from matplotlib import pyplot as plt
# import pandas as pd
from astropy.io import fits
# from astropy.visualization import ZScaleInterval
from astropy.nddata import Cutout2D
from astropy import wcs

import warnings
warnings.filterwarnings("ignore")

# from scipy.special import gamma
# def bn(n):
#     return 1.9992*n - 0.3271
# def fn(n):
#     b = bn(n)
#     return (n*np.exp(b)/b**(2*n))*gamma(2*n)

import init_settings as init


# ----- Step 1. Revising headers in FITS images ----- #

def create_fits(mother_image, segmentation_image, band, galaxy_id,
                rth=40, x0=0, y0=0):

    d, h = fits.getdata(mother_image, ext=1, header=True)
    w = wcs.WCS(h)
    
    ds = fits.getdata(segmentation_image, ext=0, header=False)
    
    cutFlags = {'size':(2*rth, 2*rth), 'wcs':w,
                'position':(round(x0-1), round(y0-1))}
    chdu = Cutout2D(data=d, **cutFlags)
    h = chdu.wcs.to_header()
    h['EXPTIME'] = '1.000'
    
    shdu = Cutout2D(data=ds, **cutFlags)

    fits.writeto(f"input/Input_{galaxy_id:05d}_"+band+".fits", chdu.data, h, overwrite=True)
    fits.writeto(f"input/Segm_{galaxy_id:05d}.fits", shdu.data, h, overwrite=True)

    
for i, gid in enumerate(np.hstack([init.id_z1, init.id_z2, init.id_z3])):
    if (i < init.n_group_z1):
        j = 0
    elif ((i >= init.n_group_z1) & (i < init.n_group_z1+init.n_group_z2)):
        j = 1
    else:
        j = 2
#     j = i // n_group
    xc = init.dat['f277w']['x'].iloc[gid-1]
    yc = init.dat['f277w']['y'].iloc[gid-1]
    create_fits(init.imglist[j], init.seglist[j], init.band[j], gid, rth=init.rth, x0=xc, y0=yc)
