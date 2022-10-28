#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:29:04 2022
@author: jlee
"""


# import time
# start_time = time.time()

import numpy as np
import glob, os, copy
# from matplotlib import pyplot as plt
# import pandas as pd
from astropy.io import fits
# from astropy.visualization import ZScaleInterval
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


# ----- Step 3. Making new segmentation images ----- #
def do_mask(input_image, segm_image, galaxy_id, dir_input='input', output_prefix='Mask', #edge_pix=5,
            n_circular_mask=0, x0=None, y0=None, rad0=None,
            n_box_mask=0, xl=None, xh=None, yl=None, yh=None):
#             rth=40, kron0=0, kfac=0.75, cxx0=0, cyy0=0, cxy0=0):

    if (dir_input[-1] == "/"):
        dir_input = dir_input[:-1]
   
    img = fits.getdata(input_image)
    seg = fits.getdata(segm_image)
    segnew = copy.deepcopy(seg)
    
    segnew[segnew == galaxy_id] = 0
    
    xsz, ysz = seg.shape[1], seg.shape[0]
    x, y = np.arange(xsz), np.arange(ysz)
    xx, yy = np.meshgrid(x, y, sparse=True)
    
#     z = (xx-rth)**2 + (yy-rth)**2 - 2.0**2
#     segnew[z <= 0.0] = 1
#     z = cxx0*(xx-rth)**2.0 + cyy0*(yy-rth)**2.0 + cxy0*(xx-rth)*(yy-rth) - (kfac*kron0)**2.0
#     segnew[z <= 0.0] = 0
    
    if (n_circular_mask > 0):
        for nn in np.arange(n_circular_mask):
            z = (xx-x0[nn])**2 + (yy-y0[nn])**2 - rad0[nn]**2
            segnew[z <= 0.0] = 1
    
    if (n_box_mask > 0):
        for nn in np.arange(n_box_mask):
            segnew[int(yl[nn]):int(yh[nn]), int(xl[nn]):int(xh[nn])] = 1
    
#     segnew2 = segnew[edge_pix:-edge_pix+1, edge_pix:-edge_pix+1]

    fits.writeto(dir_input+"/"+output_prefix+f"_{galaxy_id:05d}.fits", segnew, overwrite=True)



for i, gid in enumerate(np.hstack([init.id_z1, init.id_z2, init.id_z3])):
    if (i < init.n_group_z1):
        j = 0
    elif ((i >= init.n_group_z1) & (i < init.n_group_z1+init.n_group_z2)):
        j = 1
    else:
        j = 2
#     j = i // n_group
    xc, yc, kr = init.dat['f277w'][['x','y','kron']].iloc[gid-1].values
    cxx, cyy, cxy = init.dat['f277w'][['cxx','cyy','cxy']].iloc[gid-1].values
    do_mask(f"input/Input_{gid:05d}_"+init.band[j]+".fits", f"input/Segm_{gid:05d}.fits", gid,
            n_circular_mask=0, x0=[init.rth], y0=[init.rth], rad0=[2.0])
    do_mask(f"input/Input_{gid:05d}_"+init.band[j]+".fits", f"input/Segm_{gid:05d}.fits", gid, output_prefix='Mask2',
            n_circular_mask=1, x0=[init.rth], y0=[init.rth], rad0=[2.0])
