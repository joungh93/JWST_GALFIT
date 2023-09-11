#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 22:49:12 2022

@author: jlee
"""


import time
start_time = time.time()

import numpy as np
import glob, os
from astropy.io import fits
from astropy import wcs
from reproject import reproject_interp

import warnings
warnings.filterwarnings("ignore")


# ----- Directories & Files ----- #
dt = np.genfromtxt("info.txt", dtype=None, encoding='ascii', usecols=(0,1,2,3,4,5,7,9,10,12),
                   names=('name', 'tel', 'det', 'inst', 'flt', 'texp', 'nx', 'ny', 'pxs', 'area'))

dir_root = os.path.abspath("../")
print(dir_root)
dir_Img = dir_root+"/Images/"
imgs_sci = np.array([dir_Img + img for img in dt['name']])

jwst_swc = ((dt['det'] == 'NIRCAM') & (dt['inst'] == 'NRCA1'))
imgs_jwst_swc = imgs_sci[jwst_swc]


# ----- Reading FITS files ----- #
idx_ref = np.flatnonzero(['f444w' in img for img in imgs_sci])[0]
h_ref = fits.getheader(imgs_sci[idx_ref], ext=0)
shape_ref = (h_ref['NAXIS2'], h_ref['NAXIS1'])
w_ref = wcs.WCS(h_ref)

os.system("rm -rfv ./*.fits")
for i in range(len(imgs_sci)):
    # fits.open(imgs_sci[i]).info()
    filt = dt['flt'][i].lower()
    d1, h1 = fits.getdata(imgs_sci[i], header=True, ext=0)    # image, header
    
    if ((dt['tel'][i] == 'JWST') & (dt['det'][i] != 'NIRCAM') & (dt['inst'][i] == 'IR')):
        os.system("ln -s "+imgs_sci[i]+" ./nir_detect.fits")
    else:
        print("... Reprojecting '"+filt+"' band ...")
        
        if ((dt['nx'][i] == shape_ref[1]) & (dt['ny'][i] == shape_ref[0])):
            os.system("ln -s "+imgs_sci[i]+" ./"+filt+".fits")
        else:
            w1 = wcs.WCS(h1)
            array, footprint = reproject_interp((d1, w1), w_ref, shape_out=shape_ref,
                                                order='nearest-neighbor')
            for keys in ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
                         'CD1_1' , 'CD1_2',  'CD2_1',  'CD2_2']:
                h1[keys] = h_ref[keys]
            # fits.writeto(filt+".fits", array, h1, overwrite=True)
            fits.writeto(filt+".fits", (dt['pxs'][idx_ref]/dt['pxs'][i])**2. * array, h1, overwrite=True)


# # ----- Creating the NIR detection image ----- #
# d277 = fits.getdata("f277w.fits")
# d356 = fits.getdata("f356w.fits")
# d444 = fits.getdata("f444w.fits")

# dnir = (d277 + d356 + d444) / 3.0

# fits.writeto("nir_detect.fits", dnir, h_ref, overwrite=True)



# Printing the running time
print('--- %.4f seconds ---' %(time.time()-start_time))
