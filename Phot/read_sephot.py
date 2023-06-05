# Printing the versions of packages
from importlib_metadata import version
for pkg in ['numpy', 'matplotlib', 'astropy', 'pandas']:
    print(pkg+": ver "+version(pkg))


# importing necessary modules
import numpy as np
import glob, os
from pathlib import Path
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy import wcs
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


# ----- Directories ----- #
def get_dirs():
    dir_root = str(Path("../").resolve())
    dir_img  = str(Path(dir_root) / Path("Reproject"))
    dir_phot = str(Path(dir_root) / Path("Phot"))
    return [dir_root, dir_img, dir_phot]

dir_root, dir_img, dir_phot = get_dirs()


# ----- Bands (HST + JWST) ----- #
dt = np.genfromtxt(dir_img+"/"+"info.txt", dtype=None, encoding='ascii', usecols=(0,1,2,3,4,5,7,9,10,12),
                   names=('name', 'tel', 'det', 'inst', 'flt', 'texp', 'nx', 'ny', 'pxs', 'area'))

def get_bands(dir_img, filters_hst, filters_jwst):
    band_hst = filters_hst.split()
    imglist_hst = [dir_img+"/"+b+".fits" for b in band_hst]
    for img in imglist_hst:
        assert(glob.glob(img) != [])

    band_jwst = filters_jwst.split()
    imglist_jwst = [dir_img+"/"+b+".fits" for b in band_jwst]
    for img in imglist_jwst:
        assert(glob.glob(img) != [])

    return [band_hst, imglist_hst, band_jwst, imglist_jwst]

hst  = ' '.join(dt['flt'][dt['tel'] == 'HST'].tolist()).lower()
jwst = ' '.join(dt['flt'][(dt['tel'] == 'JWST') & (dt['det'] == 'NIRCAM')].tolist()).lower()
band_hst, imglist_hst, band_jwst, imglist_jwst = get_bands(dir_img, hst, jwst)


# ----- Photometric data dictionaries ----- #
bands = ['nir_detect'] + band_hst + band_jwst
imglists = [dir_img+"/"+"nir_detect.fits"] + imglist_hst + imglist_jwst
colnames = ['x','y','num','flux_aper','e_flux_aper','flux_auto','e_flux_auto','kron','petro',
            'backgr','ra','dec','a','b','theta','mu0','flag','fwhm','flxrad','cl','cxx','cyy','cxy']

phot_data = {}

# HST & JWST
cat_name = [img.split('/')[-1].split('.fits')[0]+".pickle" for img in imglists]
# cath_name = [img.split('/')[-1].split('.fits')[0]+"_half.cat" for img in imglist_hst]

for i in np.arange(len(imglists)):
    hdr = fits.getheader(imglists[i])
    w = wcs.WCS(hdr)
    if (i == 0):
        magzero = 28.90
    else:
        magzero = -2.5*np.log10(hdr['PHOTFLAM'])-5.0*np.log10(hdr['PHOTPLAM'])-2.408
    
    # phot_data[bands[i]] = np.genfromtxt(cat_name[i], dtype=None, encoding='ascii', names=colnames)
    # phot_data[bands[i]] = pd.DataFrame(phot_data[bands[i]])
    phot_data[bands[i]] = pd.read_pickle(cat_name[i])
    phot_data[bands[i]]['mag_aper'] = magzero - 2.5*np.log10(phot_data[bands[i]]['flux_aper'])
    phot_data[bands[i]]['e_mag_aper'] = (2.5 / np.log(10)) * (phot_data[bands[i]]['e_flux_aper'] / \
                                                              phot_data[bands[i]]['flux_aper'])
    phot_data[bands[i]]['mag_auto'] = magzero - 2.5*np.log10(phot_data[bands[i]]['flux_auto'])
    phot_data[bands[i]]['e_mag_auto'] = (2.5 / np.log(10)) * \
                                        (phot_data[bands[i]]['e_flux_auto'] / phot_data[bands[i]]['flux_auto'])

    if (i == 0):
        apcorr = phot_data[bands[i]]['mag_auto'] - phot_data[bands[i]]['mag_aper']
        e_apcorr = phot_data[bands[i]]['e_mag_auto']

    phot_data[bands[i]]['mag_corr'] = phot_data[bands[i]]['mag_aper'] + apcorr
    phot_data[bands[i]]['e_mag_corr'] = np.sqrt(phot_data[bands[i]]['e_mag_aper']**2.0 + \
                                                e_apcorr**2.0)

    negative_flux = np.isnan(phot_data[bands[i]]['mag_aper'])
    infinite_flux = np.isinf(phot_data[bands[i]]['mag_aper'])
    phot_data[bands[i]]['mag_aper'] = np.where(negative_flux | infinite_flux,
                                              99.00, phot_data[bands[i]]['mag_aper'])
    phot_data[bands[i]]['e_mag_aper'] = np.where(negative_flux | infinite_flux,
                                                99.00, phot_data[bands[i]]['e_mag_aper'])
    phot_data[bands[i]]['mag_auto'] = np.where(negative_flux | infinite_flux,
                                               99.00, phot_data[bands[i]]['mag_auto'])
    phot_data[bands[i]]['e_mag_auto'] = np.where(negative_flux | infinite_flux,
                                                 99.00, phot_data[bands[i]]['e_mag_auto'])
    phot_data[bands[i]]['mag_corr'] = np.where(negative_flux | infinite_flux,
                                               99.00, phot_data[bands[i]]['mag_corr'])
    phot_data[bands[i]]['e_mag_corr'] = np.where(negative_flux | infinite_flux,
                                                 99.00, phot_data[bands[i]]['e_mag_corr'])
#     phot_data[band_hst[i]]['mag_auto_1/2'] = np.where(negative_flux | infinite_flux,
#                                                       99.00, phot_data[band_hst[i]]['mag_auto_1/2'])
#     phot_data[band_hst[i]]['e_mag_auto_1/2'] = np.where(negative_flux | infinite_flux,
#                                                         99.00, phot_data[band_hst[i]]['e_mag_auto_1/2'])
    phot_data[bands[i]]['ra'], phot_data[bands[i]]['dec'] = \
        w.wcs_pix2world(phot_data[bands[i]]['x'].values, phot_data[bands[i]]['y'].values, 1)


# Save data
import pickle
with open("phot_data.pickle","wb") as fw:
    pickle.dump(phot_data, fw)
