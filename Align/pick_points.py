#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 7 14:38:06 2022
@author: jlee
"""


import numpy as np
import glob, os
from matplotlib import pyplot as plt
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

fig, ax = plt.subplots(figsize=(5,4))
ax.plot(df1['mag'], df1['flxrad'], '.', ms=3.0, color='gray', alpha=0.7)
ax.tick_params(axis='both', labelsize=12.0)
ax.set_xlabel("(Uncalibrated) magnitude", fontsize=12.0)
ax.set_ylabel("FLUX_RADIUS [pix]")
ax.set_ylim([-2.5, 32.5])


# ----- Selecting PSF stars ----- #
m1_hi, m1_lo = np.percentile(df1['mag'].values[~np.isnan(df1['mag'].values)], [15.0, 35.0])
sz_hi, sz_lo = 2.0, 0.5
print(f"JWST magnitude cut: {m1_hi:.2f} - {m1_lo:.2f} mag")
df1_poi = s1.Pick_PointSrc(mag_high=m1_hi, mag_low=m1_lo, size_low=sz_lo, size_high=sz_hi, ar_cut=0.8)
print(f"{len(df1_poi):d} point source candidates are selected.\n")

ax.plot([m1_hi, m1_lo], [sz_lo, sz_lo], color='b', ls='-', lw=1.25, alpha=0.7)
ax.plot([m1_hi, m1_lo], [sz_hi, sz_hi], color='b', ls='-', lw=1.25, alpha=0.7)
ax.plot([m1_hi, m1_hi], [sz_lo, sz_hi], color='b', ls='-', lw=1.25, alpha=0.7)
ax.plot([m1_lo, m1_lo], [sz_lo, sz_hi], color='b', ls='-', lw=1.25, alpha=0.7)

plt.tight_layout()
plt.savefig("sep_j200.png", dpi=300)
plt.close()


# ----- Writing point source region file ----- #
n_poi = len(df1_poi)
src1 = SkyCoord(ra=df1_poi['ra'].values*u.deg, dec=df1_poi['dec'].values*u.deg)
g = open('tmp/point_picked.reg','w')
g.write('global color=magenta dashlist=8 3 width=2 font="helvetica 10 normal roman" ')
g.write('select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \n')
g.write('fk5 \n')

poi2 = ((df1_poi['x'].values >= 100.) & (df1_poi['x'].values <= s1.dat.shape[1]-100.) & \
        (df1_poi['y'].values >= 100.) & (df1_poi['y'].values <= s1.dat.shape[0]-100.) & \
        ((df1_poi['x'].values <= 2250-100.) | (df1_poi['x'].values >= 2800+100.)))
n_poi2 = np.sum(poi2)
for k in np.arange(n_poi2):
    g.write(f"circle({src1.ra.value[poi2][k]:.5f},{src1.dec.value[poi2][k]:.5f},1.0"+'")\n')
g.close()
print(f"{n_poi2:d} point source candidates are saved.\n")


# ----- Plotting the PSF stars ----- #
from astropy.table import Table
from astropy.nddata import NDData
from astropy.visualization import simple_norm
from astropy.stats import SigmaClip

from photutils.psf import extract_stars
from photutils.psf import EPSFBuilder

stars_tbl = Table()
stars_tbl['x'] = df1_poi['x'].values[poi2]
stars_tbl['y'] = df1_poi['y'].values[poi2]

nddata = NDData(data=s1.dat-s1.backgr)
stars = extract_stars(nddata, stars_tbl, size=200-1)

ncols = 5
nrows = 1 + len(stars) // ncols

fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20), squeeze=True)
ax = ax.ravel()
for i in range(nrows * ncols):
    if (i < len(stars)):
        norm = simple_norm(stars[i], 'log', percent=99.0)
        ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='gray_r')
        ax[i].tick_params(axis='both', labelbottom=False, labelleft=False)
plt.tight_layout()
plt.savefig("point_thumb.png", dpi=300)
plt.close()


# ----- Creating the PSF models ----- #
sc_Flag = {'sigma':5000, 'maxiters':1,
           'cenfunc':'median', 'stdfunc':'std', 'grow':False}
epsf_builder = EPSFBuilder(oversampling=1, smoothing_kernel=None, maxiters=10,
                           recentering_maxiters=3, norm_radius=50.,
                           sigma_clip=SigmaClip(**sc_Flag), progress_bar=True)
epsf, fitted_stars = epsf_builder(stars)

fig, ax = plt.subplots(figsize=(4,4))

norm = simple_norm(epsf.data, 'log', percent=99.0)
im = ax.imshow(epsf.data, norm=norm, origin='lower', cmap='gray_r')
plt.colorbar(im)
plt.savefig("PSF_model.png", dpi=300)
plt.close()

