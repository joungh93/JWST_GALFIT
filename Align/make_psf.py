#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 09:56:36 2022
@author: jlee
"""

import numpy as np
import glob, os, copy
from matplotlib import pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve

import time
import warnings
warnings.filterwarnings("ignore")

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


iraf.noao()
iraf.digiphot()
iraf.daophot()

# p0_x, p0_y = np.loadtxt("tmp/point_picked_xy.reg").T


# ----- Writing the initial PSF star files ----- #
tol = 0.5 / 3600
for i in np.arange(n_img):
    if (i < n_hst):
        h = fits.getheader(imglist[i], ext=0)
    else:
        h = fits.getheader(imglist[i], ext=1)
    w = wcs.WCS(h)
    
    logname = "tmp/imexam_"+imglist[i].split('.fits')[0]+".log"
    poi = np.genfromtxt(logname, usecols=(0,1,7,9,13), names=('X','Y','SKY','E','MOFFAT'))
    ra_poi, dec_poi = w.wcs_pix2world(poi['X'], poi['Y'], 1)
    coord_regr = SkyCoord(ra=ra_poi*u.deg, dec=dec_poi*u.deg)

    magfile = "tmp/"+imglist[i].split(".fits")[0]+".mag"
    iraf.txdump(magfile, "ID,XCENTER,YCENTER,MAG,MSKY", "yes",
                Stdout="tmp/tmp")
    dat_phot = np.genfromtxt("tmp/tmp", dtype='float', encoding="ascii",
                             names=("ID","XCENTER","YCENTER","MAG","MSKY"))
    ra_phot, dec_phot = w.wcs_pix2world(dat_phot['XCENTER'], dat_phot['YCENTER'], 1)
    coord_phot = SkyCoord(ra=ra_phot*u.deg, dec=dec_phot*u.deg)
    idx, d2d, d3d = coord_regr.match_to_catalog_sky(coord_phot)
    matched = d2d.value < tol
    n_mch = np.sum(matched)
    midx1 = np.where(matched)[0]
    midx0 = idx[matched]
    print(imglist[i].split(".fits")[0]+": {0:d}/{1:d} point sources matched.".format(n_mch, len(idx)))

    f = open("PSF/"+imglist[i].split('.fits')[0]+".pst.0", "w")
    f.write("#N ID    XCENTER   YCENTER   MAG         MSKY \n")
    f.write("#U ##    pixels    pixels    magnitudes  counts \n")
    f.write("#F %-9d  %-10.3f   %-10.3f   %-12.3f     %-15.7g\n")
    f.write("#\n")
    for j in midx0:
        f.write("{0:<9d} {1:<10.3f} {2:<10.3f} {3:<12.3f} {4:<15.7g}\n".format(int(dat_phot['ID'][j]),
                                                                               dat_phot['XCENTER'][j],
                                                                               dat_phot['YCENTER'][j],
                                                                               dat_phot['MAG'][j],
                                                                               dat_phot['MSKY'][j]))
    f.close()


# ----- Reading FWHMs and sky values ----- #
fwhms = np.loadtxt("tmp/fwhms.txt")
print(fwhms)

sky_value, sky_sigma = np.loadtxt("tmp/sky.txt")
print(sky_value, sky_sigma)


# ----- Masking tasks ----- #
for i in np.arange(n_img):
    if (i < n_hst):
        d, h = fits.getdata(imglist[i], ext=0, header=True)
    else:
        d, h = fits.getdata(imglist[i], ext=1, header=True)

    imgname = imglist[i].split(".fits")[0]

    logname = "tmp/imexam_"+imgname+".log"
    poi = np.genfromtxt(logname, usecols=(0,1,7,9,13), names=('X','Y','SKY','E','MOFFAT'))
    ra_poi, dec_poi = w.wcs_pix2world(poi['X'], poi['Y'], 1)
    coord_regr = SkyCoord(ra=ra_poi*u.deg, dec=dec_poi*u.deg)

    x, y = np.arange(d.shape[1]), np.arange(d.shape[0])
    xx, yy = np.meshgrid(x, y)

    msk = (d > sky_value[i] + sky_sigma[i])
    for j in range(len(poi)):
        reg_20sig = ((xx-poi['X'][j]+1)**2 + (yy-poi['Y'][j]+1)**2 <= (20*fwhms[i])**2)
        reg_10sig = (reg_20sig & ((xx-poi['X'][j]+1)**2 + (yy-poi['Y'][j]+1)**2 >= (10*fwhms[i])**2))
        msk[reg_20sig] = False
        if (i < n_hst):
            msk[reg_10sig & (d > sky_value[i] + 2.0*sky_sigma[i])] = True
    d[msk] = sky_value[i]

    msk2 = np.zeros_like(msk, dtype=bool)
    msk3 = copy.deepcopy(msk2)
    d2 = copy.deepcopy(d)
    for j in range(len(poi)):
        reg_03sig = (((xx-poi['X'][j]+1)**2 + (yy-poi['Y'][j]+1)**2 <= (10*fwhms[i])**2) & \
                     ((xx-poi['X'][j]+1)**2 + (yy-poi['Y'][j]+1)**2 >= (3.5*fwhms[i])**2))
        reg_05sig = (((xx-poi['X'][j]+1)**2 + (yy-poi['Y'][j]+1)**2 <= (10*fwhms[i])**2) & \
                     ((xx-poi['X'][j]+1)**2 + (yy-poi['Y'][j]+1)**2 >= (5*fwhms[i])**2))
        reg_10sig = (((xx-poi['X'][j]+1)**2 + (yy-poi['Y'][j]+1)**2 <= (20*fwhms[i])**2) & \
                     ((xx-poi['X'][j]+1)**2 + (yy-poi['Y'][j]+1)**2 >= (10*fwhms[i])**2))
        
        # HST/ACS (F435W, F606W, F814W)
        if (i < 3):
            msk2[reg_05sig & (d > sky_value[i] + 4.0*sky_sigma[i])] = True
            msk3[msk2 | ((xx-poi['X'][j]+1)**2 + (yy-poi['Y'][j]+1)**2 <= (5*fwhms[i])**2)] = True
            ker_std = 2
        # HST/WFC3-IR (F105W, F125W, F140W, F160W)
        elif (i < n_hst):
            msk2[reg_03sig & (d > sky_value[i] + 4.0*sky_sigma[i])] = True
            msk3[msk2 | ((xx-poi['X'][j]+1)**2 + (yy-poi['Y'][j]+1)**2 <= (5*fwhms[i])**2)] = True
            ker_std = 4
        # JWST/NIRCam (F090W, F150W, F200W) + F277W
        elif (i < n_hst+4):
            msk2[reg_10sig & (d > sky_value[i] + 4.0*sky_sigma[i])] = True
            msk3[msk2 | ((xx-poi['X'][j]+1)**2 + (yy-poi['Y'][j]+1)**2 <= (10*fwhms[i])**2)] = True
            ker_std = 2
        # JWST/NIRCam (F277W, F356W, F444W) - F277W
        else:
            msk2[(reg_05sig | reg_10sig) & (d > sky_value[i] + 1.0*sky_sigma[i])] = True
            msk3[msk2 | ((xx-poi['X'][j]+1)**2 + (yy-poi['Y'][j]+1)**2 <= (10*fwhms[i])**2)] = True
            ker_std = 2
    
    kernel = Gaussian2DKernel(stddev=ker_std)        
    d2[msk3] = np.nan
    dconv = convolve(d2, kernel)
    d[msk2] = dconv[msk2]

    fits.writeto("PSF/"+imgname+"_masked.fits", d, h, overwrite=True)
    print("'"+imgname+"_masked.fits' has been written.")


# ----- PSF task (non-interative) ----- #
start_time = time.time()

for i in np.arange(n_img):
    iraf.unlearn("psf")
    iraf.unlearn("seepsf")
    iraf.unlearn("allstar")
    iraf.unlearn("substar")

    imgname = imglist[i].split(".fits")[0]
    dat = fits.getdata("PSF/"+imgname+"_masked.fits", ext=0)

    print("\n----- Starting PSF task for "+imglist[i]+" -----\n")

    os.system("rm -rfv PSF/"+imgname+".als.* PSF/"+imgname+"*.arj.* PSF/"+imgname+"*.psg.*")
    os.system("rm -rfv PSF/"+imgname+".psf.*fits PSF/"+imgname+".spsf.*fits PSF/"+imgname+".sub.*fits ")
    for j, pst in enumerate(sorted(glob.glob("PSF/"+imgname+".pst.*"))):
        if (j > 0):
            os.system("rm -rfv "+pst)

    magfile = "tmp/"+imgname+".mag"
    kwargs = {"fwhmpsf":fwhms[i], "datamax":1000.0, 
              "readnoise":5.0, "epadu":2.0, "itime":1.0,
              "psfrad":10.0*fwhms[i],
              "fitrad":4.0*fwhms[i],
              # "fitsky":False,
              "verify":False}

    n_iter = 1
    image = "PSF/"+imgname+"_masked.fits"
    func = "moffat15"#"auto"
    # std = sky_sigma[i]
    dat = fits.getdata(image)
    avg, med, std = sigma_clipped_stats(dat, sigma=3.0)

    while True:
        psffile = "PSF/"+imgname+".psf."+str(n_iter)+".fits"
        seepsffile = "PSF/"+imgname+".spsf."+str(n_iter)+".fits"
        pstfile = "PSF/"+imgname+".pst."+str(n_iter)
        psgfile = "PSF/"+imgname+".psg."+str(n_iter)
        os.system("rm -rfv "+psffile+" "+seepsffile+" "+pstfile+" "+psgfile)

        iraf.psf(image=image, photfile=magfile, pstfile="PSF/"+imgname+".pst."+str(n_iter-1),
                 psfimage=psffile, opstfile=pstfile, groupfile=psgfile, 
                 function=func, varorder=0, saturated=True, 
                 interactive=False, sigma=std, **kwargs)
        iraf.seepsf(psffile, seepsffile)
        # ds9_command = "ds9 -scalemode zscale "+seepsffile+" &"
        # os.system(ds9_command)

        # response = raw_input("Are you satisfied? (Y/N) ")
        # if (response == "Y"):
        if (n_iter > 2):
            os.system("rm -rfv "+imgname+".als.* "+imgname+".arj.* "+imgname+".psg.*"+imgname+".sub.*.fits ")
            break
        # elif (response == "N"):
        else:
            alsfile = "PSF/"+imgname+".als."+str(n_iter)
            rejfile = "PSF/"+imgname+".arj."+str(n_iter)
            subimage = "PSF/"+imgname+".sub."+str(n_iter)+".fits"
            os.system("rm -rfv "+alsfile+" "+rejfile+" "+subimage)
            iraf.allstar(image=image, photfile=psgfile, psfimage=psffile, allstarfile=alsfile, 
                         rejfile=rejfile, subimage=subimage, sigma=std, **kwargs)
            
            os.system("rm -rfv "+subimage)
            iraf.substar(image=image, photfile=alsfile, exfile=pstfile, psfimage=psffile,
                         subimage=subimage, sigma=std, **kwargs)
            
            # ds9_command = "ds9 -scalemode zscale -scale lock yes -frame lock image "+ \
            #               imglist[i]+" "+subimage+" &"
            # os.system(ds9_command)
            
            try:
                dsub = fits.getdata(subimage)
                if (np.nansum(np.abs(dat - dsub)) > 0.1*np.abs(sky_value[i]*len(dat.flatten()))):
                    pass
                else:
                    image = subimage
            except:
                pass
            # image = subimage
            # hh = fits.getheader(seepsffile)
            # func = hh['FUNCTION']
            n_iter += 1
        # else:
        #     raise TypeError
        #     break
    
end_time = time.time()
print("\n--- PSF tasks take {0:.4f} sec ---".format(end_time-start_time))
