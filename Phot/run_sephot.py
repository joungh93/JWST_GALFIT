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


# ----- Parameter file ----- #
os.system("sex -dp > default.param")    # default parameter file (400 output parameters)

def get_params(param_name):
    f = open(param_name, "w")
    f.write("X_IMAGE\n")    # Object position along x [pixel]
    f.write("Y_IMAGE\n")    # Object position along y [pixel]
    f.write("NUMBER\n")    # Object number
    f.write("FLUX_APER(1)\n")    # Kron-like elliptical aperture magnitude [mag]
    f.write("FLUXERR_APER(1)\n")    # RMS error for MAG_AUTO    [mag]
    f.write("FLUX_AUTO\n")    # Kron-like elliptical aperture magnitude [mag]
    f.write("FLUXERR_AUTO\n")    # RMS error for MAG_AUTO    [mag]
    f.write("KRON_RADIUS\n")    # Kron apertures in units of A or B
    f.write("PETRO_RADIUS\n")    # Petrosian apertures in units of A or B
    f.write("BACKGROUND\n")    # Background at centroid position
    f.write("ALPHA_J2000\n")    # Right ascension of object center (J2000)
    f.write("DELTA_J2000\n")    # Declination of object center (J2000)
    f.write("A_IMAGE\n")    # Along major axis
    f.write("B_IMAGE\n")    # Along minor axis
    f.write("THETA_IMAGE\n")    # Position angle
    f.write("MU_MAX\n")    # Peak surface brightness above background [mag * arcsec**(-2)]
    f.write("FLAGS\n")    # Extraction flags
    f.write("FWHM_IMAGE\n")    # FWHM assuming a gaussian core
    f.write("FLUX_RADIUS\n")    # Half-light radii
    f.write("CLASS_STAR\n")    # Star/Galaxy classifier output
    f.write("CXX_IMAGE\n")    # Cxx object ellipse parameter
    f.write("CYY_IMAGE\n")    # Cyy object ellipse parameter
    f.write("CXY_IMAGE\n")    # Cxy object ellipse parameter
    f.close()

param_name = "output.param"
get_params(param_name)


# ----- Configuration file ----- #
config_name = "config.txt"
os.system("sex -dd > "+config_name)

def get_configs(image_obj, image_ref, script_name,
                config_file="config.txt", param_file="output.param", #bands,
                conv_file="/data01/jhlee/Downloads/sextractor-2.25.0/config/gauss_2.0_3x3.conv",
                nnw_file="/data01/jhlee/Downloads/sextractor-2.25.0/config/default.nnw",
                detect_minarea=20, detect_thresh=1.0,
                deblend_nthresh=16, deblend_mincont=0.0001,
                saturated=100.0, phot_radius=5.0, pixel_scale=0.04, kron_fact=2.5, min_radius=5.0,
                magzero=[None], effective_gain=[None], fwhms=[None],
                back_size=32, back_flt=3, back_phot=24,
                mem_overstack=4000, mem_pixstack=400000, mem_bufsize=5000,
                check=True, mode="", weight=True, weight_image=None):

    f = open(script_name, "a")
    for i in range(len(image_obj)):
        image_str = image_obj[i].split('/')[-1].split('.fits')[0]
        if not (mode == ""):
            image_str += "_"+mode
        # if (image_obj[i] == image_ref):
        #     comm = "sex "+image_ref+" -c "+config_file
        # else:
        comm = "sex "+image_ref+","+image_obj[i]+" -c "+config_file
        comm +=  " -CATALOG_NAME "+image_str+".cat  -PARAMETERS_NAME "+param_file
        comm += f" -DETECT_MINAREA {detect_minarea:d} -DETECT_THRESH {detect_thresh:.1f} -ANALYSIS_THRESH {detect_thresh:.1f} "
        comm += f" -DEBLEND_NTHRESH {deblend_nthresh:d} -DEBLEND_MINCONT {deblend_mincont:.4f} "
        comm +=  " -FILTER_NAME "+conv_file+f" -SATUR_LEVEL {saturated:.1f} -STARNNW_NAME "+nnw_file
        comm += f" -PHOT_APERTURES {2.0*phot_radius:.2f} "
        comm += f" -PHOT_AUTOPARAMS {kron_fact:.1f},{min_radius:.1f} -PIXEL_SCALE {pixel_scale:.3f} "
        comm += f" -MAG_ZEROPOINT {mag0[i]:.3f} -GAIN {egain[i]:>9.1f} -SEEING_FWHM {fwhm[i]:.3f} "
        comm += f" -BACK_SIZE {back_size:d} -BACK_FILTERSIZE {back_flt:d} "
        comm += f" -BACKPHOTO_TYPE LOCAL -BACKPHOTO_THICK {back_phot:d} "
        comm += f" -MEMORY_OBJSTACK {mem_overstack:d} -MEMORY_PIXSTACK {mem_pixstack:d} -MEMORY_BUFSIZE {mem_bufsize:d} "
        if check:
            image_aper = image_str+"_aper.fits"
            image_segm = image_str+"_segm.fits"
            # image_objt = image_str+"_objt.fits"
            comm += " -CHECKIMAGE_TYPE APERTURES,SEGMENTATION"#,OBJECTS"
            comm += " -CHECKIMAGE_NAME "+image_aper+","+image_segm#+","+image_objt
        if weight:
            comm += " -WEIGHT_TYPE MAP_WEIGHT -WEIGHT_IMAGE "+weight_image

        f.write("nohup "+comm+" > run_sep-"+image_str+".out &\n")
    f.close()


refimg = dir_img+"/"+"nir_detect.fits"
h_ref  = fits.getheader(refimg, ext=0)
mag0   = [28.9]
egain  = [h_ref['EXPTIME']]

imglist_tot = [refimg] + imglist_hst + imglist_jwst
for img in imglist_hst+imglist_jwst:
    h = fits.getheader(img, ext=0)
    mag0.append(-2.5*np.log10(h['PHOTFLAM'])-5.0*np.log10(h['PHOTPLAM'])-2.408)
    egain.append(h['EXPTIME'])#h['CCDGAIN']*h['EXPTIME'])
mag0, egain = np.array(mag0), np.array(egain)
fwhm = np.array([0.1]*len(imglist_tot))


# # ----- SExtractor (field-mode?) ----- #
# scr_name = "sephot.sh"
# f = open(scr_name, "w")
# f.write("# SExtrator photometry\n")
# f.close()

# get_configs(imglist_tot, refimg, scr_name,
#             config_file="config.txt", param_file="output.param", #bands,
#             conv_file="/data01/jhlee/Downloads/sextractor-2.25.0/config/gauss_2.0_3x3.conv",
#             nnw_file="/data01/jhlee/Downloads/sextractor-2.25.0/config/default.nnw",
#             detect_minarea=20, detect_thresh=1.0,
#             deblend_nthresh=16, deblend_mincont=0.0001,
#             saturated=100.0, phot_radius=0.5 / 0.04, pixel_scale=0.04, kron_fact=2.5, min_radius=3.5,
#             magzero=mag0, effective_gain=egain, fwhms=fwhm,
#             back_size=16, back_flt=3, back_phot=24,
#             mem_overstack=4000, mem_pixstack=400000, mem_bufsize=5000,
#             check=True)


# # ----- Running SExtractor ----- #
# os.system("sh "+scr_name)

img_wht = dir_img+"/"+"nir_weight.fits"

# ----- SExtractor (cold-mode) ----- #
scr_name = "sephot_c.sh"
f = open(scr_name, "w")
f.write("# SExtrator photometry (cold-mode)\n")
f.close()

param_cold = {'config_file': "config.txt", 'param_file':"output.param",
              'conv_file':"/data01/jhlee/Downloads/sextractor-2.25.0/config/tophat_5.0_5x5.conv",
              'nnw_file':"/data01/jhlee/Downloads/sextractor-2.25.0/config/default.nnw",
              'detect_minarea':20, 'detect_thresh':1.0,
              'deblend_nthresh':32, 'deblend_mincont':0.01,
              'saturated':100.0, 'phot_radius':0.5/0.04,
              'pixel_scale':0.04, 'kron_fact':2.5, 'min_radius':3.5,
              'back_size':64, 'back_flt':5, 'back_phot':48,
              'mem_overstack':4000, 'mem_pixstack':400000, 'mem_bufsize':5000,
              'mode':"c", 'weight':True, 'weight_image':img_wht}

get_configs(imglist_tot[0:1], refimg, scr_name,
            magzero=mag0[0:1], effective_gain=egain[0:1], fwhms=fwhm[0:1], check=True,
            **param_cold)

get_configs(imglist_tot[1:], refimg, scr_name,
            magzero=mag0[1:], effective_gain=egain[1:], fwhms=fwhm[1:], check=False,
            **param_cold)

# # ----- Running SExtractor ----- #
# os.system("sh "+scr_name)


# ----- SExtractor (hot-mode) ----- #
scr_name = "sephot_h.sh"
f = open(scr_name, "w")
f.write("# SExtrator photometry (hot-mode)\n")
f.close()

param_hot  = {'config_file': "config.txt", 'param_file':"output.param",
              'conv_file':"/data01/jhlee/Downloads/sextractor-2.25.0/config/gauss_2.5_5x5.conv",
              'nnw_file':"/data01/jhlee/Downloads/sextractor-2.25.0/config/default.nnw",
              'detect_minarea':20, 'detect_thresh':1.0,
              'deblend_nthresh':16, 'deblend_mincont':0.0001,
              'saturated':100.0, 'phot_radius':0.5/0.04,
              'pixel_scale':0.04, 'kron_fact':2.5, 'min_radius':3.5,
              'back_size':16, 'back_flt':3, 'back_phot':24,
              'mem_overstack':4000, 'mem_pixstack':400000, 'mem_bufsize':5000,
              'mode':"h", 'weight':True, 'weight_image':img_wht}

get_configs(imglist_tot[0:1], refimg, scr_name,
            magzero=mag0[0:1], effective_gain=egain[0:1], fwhms=fwhm[0:1], check=True,
            **param_hot)

get_configs(imglist_tot[1:], refimg, scr_name,
            magzero=mag0[1:], effective_gain=egain[1:], fwhms=fwhm[1:], check=False,
            **param_hot)

# # ----- Running SExtractor ----- #
# os.system("sh "+scr_name)

