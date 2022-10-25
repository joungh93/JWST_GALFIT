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


# ----- Step 2. Creating PSF images ----- #
import webbpsf
interval = ZScaleInterval()
magtot_psf = []
fig, axs = plt.subplots(1, len(init.band), figsize=(8,3))
for i, b in enumerate(init.band):
    ax = axs[i]
    nc = webbpsf.NIRCam()
    nc.pixelscale = init.pixel_scale
    nc.filter = b.upper()
    psf = nc.calc_psf()
    psfdata = psf[1].data
    ax.imshow(psfdata, cmap='gray_r', origin='lower',
              vmin=interval.get_limits(psfdata)[0] / 10,
              vmax=interval.get_limits(psfdata)[1] * 10)
    ax.tick_params(labelleft=False, labelbottom=False)
    ax.tick_params(axis='both', length=0)
    ax.text(0.50, 0.95, b.upper(), fontsize=12.0, fontweight='bold',
            transform=ax.transAxes, ha='center', va='top')
    psfflux = psfdata.sum()
    psfmag  = init.magAB_zero - 2.5*np.log10(psfflux)
    ax.text(0.05, 0.05, f"{psfflux:.4f} count = {psfmag:.3f} mag",
            fontsize=10.0, transform=ax.transAxes, ha='left', va='bottom')
    magtot_psf.append(psfmag)
    fits.writeto("input/PSF_"+b+".fits", psfdata, overwrite=True)
plt.tight_layout()
plt.savefig("PSFs.png", dpi=300)
