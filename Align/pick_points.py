#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 7 14:38:06 2022
@author: jlee
"""


import numpy as np
import glob, os
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
from astropy import units as u
from apply_sep1 import sep0


# ----- JWST images ----- #
img_jwst = sorted(glob.glob("Aj*.fits"))


# ----- SEP photometry for JWST image ----- #
s1 = sep0(img_jwst[2], img_extn=1, hdr_extn=1)
df1 = s1.AutoPhot()
m1_hi, m1_lo = np.percentile(df1['mag'].values[~np.isnan(df1['mag'].values)], [10.0, 40.0])
print(f"JWST magnitude cut: {m1_hi:.2f} - {m1_lo:.2f} mag")
df1_poi = s1.Pick_PointSrc(mag_high=m1_hi, mag_low=m1_lo, size_low=1.0, size_high=5.0, ar_cut=0.8)
print(f"{len(df1_poi):d} point source candidates are selected.\n")


# ----- Mathing point sources ----- #
n_poi = len(df1_poi)
src1 = SkyCoord(ra=df1_poi['ra'].values*u.deg, dec=df1_poi['dec'].values*u.deg)

g = open('tmp/point_picked.reg','w')
g.write('global color=magenta dashlist=8 3 width=2 font="helvetica 10 normal roman" ')
g.write('select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \n')
g.write('fk5 \n')
for k in np.arange(n_poi):
    g.write(f"circle({src1.ra.value[k]:.5f},{src1.dec.value[k]:.5f},0.5"+'")\n')
g.close()

