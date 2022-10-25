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
import pandas as pd
# from astropy.io import fits
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


# ----- Directories ----- #
current_dir = os.getcwd()
dir_img = "/data/jlee/DATA/JWST/First/SMACS0723/MAST/Reproject/"
imglist = [dir_img+"f090w.fits", dir_img+"f277w.fits", dir_img+"f356w.fits"]


# ----- Reading photometric catalogs ----- #
dir_phot = "/data/jlee/DATA/JWST/First/SMACS0723/MAST/Phot/"
band = ['f090w', 'f277w', 'f356w']
cat_name = [dir_phot+b[1:-1]+".cat" for b in band]
cate_name = [dir_phot+b[1:-1]+"_var.cat" for b in band]
# colnames = ['x','y','num','flux','e_flux','kron','backgr','ra','dec',
#             'a','b','theta','flag','fwhm','flxrad','cl']
colnames = ['x','y','num','flux','e_flux','kron','backgr','ra','dec',
            'a','b','theta','flag','fwhm','flxrad','cl','cxx','cyy','cxy']
pixel_scale = 0.063  # arcsec/pixel
magAB_zero = -2.5*np.log10(23.504*pixel_scale*pixel_scale) + 23.90  # magnitude
seglist = [dir_phot+"090_segm.fits", dir_phot+"277_segm.fits", dir_phot+"356_segm.fits"]


# ----- Photometric data dictionaries ----- #
dat = {}
for i in np.arange(len(band)):
    dat[band[i]] = np.genfromtxt(cat_name[i], dtype=None, encoding='ascii', names=colnames)
    e_dat = np.genfromtxt(cate_name[i], dtype=None, encoding='ascii', names=colnames)
    dat[band[i]]['e_flux'] = np.sqrt(np.maximum(0., e_dat['flux']) + e_dat['backgr'] * \
                                     np.pi*dat[band[i]]['a']*dat[band[i]]['b']*dat[band[i]]['kron']**2.)
    dat[band[i]] = pd.DataFrame(dat[band[i]])
    dat[band[i]]['mag'] = -2.5*np.log10(23.504 * pixel_scale**2 * dat[band[i]]['flux']) + 23.90
    dat[band[i]]['e_mag'] = (2.5 / np.log(10)) * (dat[band[i]]['e_flux'] / dat[band[i]]['flux'])
    negative_flux = np.isnan(dat[band[i]]['mag'])
    dat[band[i]]['mag'] = np.where(negative_flux, 99.00, dat[band[i]]['mag'])
    dat[band[i]]['e_mag'] = np.where(negative_flux, 99.00, dat[band[i]]['e_mag'])


# ----- Reading the region files (Ferreira+22) ----- #
dir_reg = "/data/jlee/DATA/JWST/First/SMACS0723/Ferreira+22/Regions/"
coo1 = pd.DataFrame(np.loadtxt(dir_reg+"photz_range1.reg")).rename(columns={0:'x', 1:'y'})
coo2 = pd.DataFrame(np.loadtxt(dir_reg+"photz_range2.reg")).rename(columns={0:'x', 1:'y'})
coo3 = pd.DataFrame(np.loadtxt(dir_reg+"photz_range3.reg")).rename(columns={0:'x', 1:'y'})

print(f"Group 1 (z = 1.5-3.0): {len(coo1):d} objects")
print(f"Group 2 (z = 3.0-4.0): {len(coo2):d} objects")
print(f"Group 3 (z = 4.0-6.0): {len(coo3):d} objects")

assert coo1.merge(dat['f090w'], on=['x', 'y'], how="left")['num'].isna().sum() == 0
num_z1 = coo1.merge(dat['f090w'], on=['x', 'y'], how="left").sort_values('flux', ascending=False)['num'].values
print(f"{len(np.unique(num_z1)):d} / {len(coo1):d}")

assert coo2.merge(dat['f090w'], on=['x', 'y'], how="left")['num'].isna().sum() == 0
num_z2 = coo2.merge(dat['f090w'], on=['x', 'y'], how="left").sort_values('flux', ascending=False)['num'].values
print(f"{len(np.unique(num_z2)):d} / {len(coo2):d}")

assert coo3.merge(dat['f090w'], on=['x', 'y'], how="left")['num'].isna().sum() == 0
num_z3 = coo3.merge(dat['f090w'], on=['x', 'y'], how="left").sort_values('flux', ascending=False)['num'].values
print(f"{len(np.unique(num_z3)):d} / {len(coo3):d}")



# n_group = 40
id_z1 = num_z1#[:n_group]
id_z2 = num_z2#[:n_group]
id_z3 = num_z3#[:n_group]

n_group_z1 = len(num_z1)
n_group_z2 = len(num_z2)
n_group_z3 = len(num_z3)


rth = 40    # Thumbnail radius
