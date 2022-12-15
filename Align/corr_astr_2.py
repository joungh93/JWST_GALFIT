#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 02 23:31:15 2022
@author: jlee
"""


import numpy as np
import glob, os
from astropy.io import fits
from astropy import wcs
from pathlib import Path
from reproject import reproject_interp
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
from astropy import units as u
from apply_sep1 import sep0
import astroalign as aa


# ----- Reference images ----- #
imgref_hst = "h814.fits"
imgref_jwst = "j200.fits"


# ----- SEP photometry for HST image ---- #
s0 = sep0(imgref_hst, img_extn=0, hdr_extn=0)
df0 = s0.AutoPhot()
m0_hi, m0_lo = np.percentile(df0['mag'].values[~np.isnan(df0['mag'].values)], [5.0, 50.0])
print(f"HST magnitude cut: {m0_hi:.2f} - {m0_lo:.2f} mag")
df0_poi = s0.Pick_PointSrc(mag_high=m0_hi, mag_low=m0_lo, size_low=1.0, size_high=6.0, ar_cut=0.75)
print(f"{len(df0_poi):d} point source candidates are selected.\n")


# ----- SEP photometry for JWST image ----- #
s1 = sep0(imgref_jwst, img_extn=1, hdr_extn=1)
df1 = s1.AutoPhot()
m1_hi, m1_lo = np.percentile(df1['mag'].values[~np.isnan(df1['mag'].values)], [5.0, 50.0])
print(f"JWST magnitude cut: {m1_hi:.2f} - {m1_lo:.2f} mag")
df1_poi = s1.Pick_PointSrc(mag_high=m1_hi, mag_low=m1_lo, size_low=1.0, size_high=25.0, ar_cut=0.75)
print(f"{len(df1_poi):d} point source candidates are selected.\n")


# ----- Mathing point sources ----- #
src0 = SkyCoord(ra=df0_poi['ra'].values*u.deg, dec=df0_poi['dec'].values*u.deg)
src1 = SkyCoord(ra=df1_poi['ra'].values*u.deg, dec=df1_poi['dec'].values*u.deg)

tol = 0.5 / 3600.0    # arcsec
idx, d2d, d3d = src1.match_to_catalog_sky(src0)
matched = d2d.value < tol
n_mch = np.sum(matched)
midx1 = np.where(matched)[0]
midx0 = idx[matched]
print(f"{n_mch:d} point sources matched.")

g = open('tmp/point_matched.reg','w')
g.write('global color=red dashlist=8 3 width=2 font="helvetica 10 normal roman" ')
g.write('select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1 \n')
g.write('fk5 \n')
for k in np.arange(n_mch):
    g.write(f"circle({src0.ra.value[midx0][k]:.5f},{src0.dec.value[midx0][k]:.5f},0.5"+'")\n')
g.close()


# ----- Alignment of HST & JWST images ----- #
xc0, yc0 = df0_poi.iloc[midx0]['x'].values, df0_poi.iloc[midx0]['y'].values
xc1, yc1 = df1_poi.iloc[midx1]['x'].values, df1_poi.iloc[midx1]['y'].values 
src = np.array([xc1, yc1]).T ; dst = np.array([xc0, yc0]).T
tform = aa.estimate_transform('affine', src, dst)
print("# ----- Parameters of the transformation ----- #")
print(f"Rotation: {tform.rotation*180.0/np.pi:.2f} degrees")
print("Scale factor: ({:.2f}, {:.2f})".format(*tform.scale))
print("Translation: (x, y) = ({:.2f}, {:.2f})".format(*tform.translation))
print(f"Tranformation matrix:\n{tform.params}")
print("\n")


# ----- Writing aligned images ----- #
imglist_hst = ["h435.fits", "h606.fits", "h814.fits",
               "h105.fits", "h125.fits", "h140.fits", "h160.fits"]
imglist_jwst = ["j090.fits", "j150.fits", "j200.fits",
                "j277.fits", "j356.fits", "j444.fits"]

prefix = "A"
for i in range(len(imglist_hst)):
    os.system("cp -rpv "+imglist_hst[i]+" "+prefix+imglist_hst[i])

for i in range(len(imglist_jwst)):
    fhd0 = fits.PrimaryHDU()
    fhd1 = fits.ImageHDU()
    fhd2 = fits.ImageHDU()

    h0 = fits.getheader(imglist_jwst[i], ext=0)    # Header
    d1, h1 = fits.getdata(imglist_jwst[i], header=True, ext=1)    # SCI
    d2, h2 = fits.getdata(imglist_jwst[i], header=True, ext=2)    # VAR

    aligned_sci, footprint = aa.apply_transform(tform, d1.byteswap().newbyteorder(), s0.dat, inter_order=0)
    aligned_var, footprint = aa.apply_transform(tform, d2.byteswap().newbyteorder(), s0.dat, inter_order=0)

    fhd1.data = aligned_sci
    fhd2.data = aligned_var

    fhd0.header = h0
    fhd1.header = h1
    h2['EXTNAME'] = 'VAR'
    fhd2.header = h2

    fcb_hdu = fits.HDUList([fhd0, fhd1, fhd2])
    fcb_hdu.writeto(prefix+imglist_jwst[i], overwrite=True)
    print("'"+prefix+imglist_jwst[i]+"' has been written.")


# # ----- Basic setting ----- #
# os.system("rm -rfv config.* *.fits *.cat *.sh *.head *.png *.xml default.param")

# dir_hb = "/data/jlee/DATA/HLA/SMACS0723/Img_total/pyDrizzlePac/Results/"
# hb_img = [dir_hb+"435_sci.fits", dir_hb+"606_sci.fits", dir_hb+"814_sci.fits",
#           dir_hb+"105_sci.fits", dir_hb+"125_sci.fits", dir_hb+"140_sci.fits", dir_hb+"160_sci.fits"]
# n_hst = len(hb_img)
# # for i in np.arange(n_hst):
# #     os.system("cp -rpv "+hb_img[i]+" ./h"+hb_img[i].split("/")[-1].split("_")[0]+".fits")

# dir_jw = "/data/jlee/DATA/JWST/First/SMACS0723/MAST/Reproject/"
# jw_img = [dir_jw+"f090w.fits", dir_jw+"f150w.fits", dir_jw+"f200w.fits",
#           dir_jw+"f277w.fits", dir_jw+"f356w.fits", dir_jw+"f444w.fits"]
# n_jwst = len(jw_img)
# # for i in np.arange(n_jwst):
# #     os.system("cp -rpv "+jw_img[i]+" ./j"+jw_img[i].split("/")[-1][1:4]+".fits")

# imglist = hb_img + jw_img
# n_img = len(imglist)


# idx_ref, ref_img, out_cat, out_img = 3, "ref277.fits", "ref277.cat", "ref277.fits"
# os.system("cp -rpv "+jw_img[3]+" "+ref_img)

# pxs = 0.063
# extn = 1
# ra0, dec0 = "07:23:01.733", "-73:28:03.40"
# img = fits.getdata(ref_img, header=False, ext=extn)
# xsz0, ysz0 = img.shape[1], img.shape[0]


# # ----- Writing .param file ----- #
# par_file = "default.param"
# with open(par_file, "w") as f:

#     # SExtractor for SCAMP
#     f.write("XWIN_IMAGE\nYWIN_IMAGE\n")
#     f.write("ERRAWIN_IMAGE\nERRBWIN_IMAGE\nERRTHETAWIN_IMAGE\n")
#     f.write("FLUX_AUTO\nFLUXERR_AUTO\n")
#     f.write("FLAGS\nFLAGS_WEIGHT\n")
#     f.write("FLUX_RADIUS\nELONGATION\n")


# # ----- Writing the SExtractor configuration file ----- #
# os.system("rm -rfv config.sex "+out_cat)
# cfg_file = "config.sex"
# os.system("sex -dd > "+cfg_file)

# sh_file = "sephot.sh"
# with open(sh_file, "w") as f:
    
#     # SExtractor scripts
#     str_run = "sex "+ref_img+f"[{extn:d}] -c config.sex "
#     str_param = "-DETECT_MINAREA 5 -DETECT_THRESH 4.0 -ANALYSIS_THRESH 4.0 "
#     str_param += "-DEBLEND_NTHRESH 32 -DEBLEND_MINCONT 0.005 "
#     str_param += "-FILTER_NAME /usr/share/sextractor/default.conv "
#     str_param += "-SATUR_LEVEL 100.0 -MAG_ZEROPOINT 25.0 "
#     str_param += f"-PIXEL_SCALE {pxs:.3f} -SEEING_FWHM 0.1 "
#     str_param += "-STARNNW_NAME /usr/share/sextractor/default.nnw "
#     str_param += "-BACK_SIZE 32 -BACK_FILTERSIZE 3 -BACKPHOTO_TYPE LOCAL"

#     # FITS_LDAC
#     f.write(str_run)
#     f.write("-CATALOG_NAME "+out_cat+" ")
#     f.write("-CATALOG_TYPE FITS_LDAC ")
#     f.write(str_param)
#     f.write("\n")

# os.system("sh "+sh_file)


# # ----- Writing the SCAMP configuration file ----- #
# os.system("rm -rfv config.scamp *.head")
# cfg_file = "config.scamp"
# os.system("scamp -dd > "+cfg_file)

# sh_file = "scamp.sh"
# with open(sh_file, "w") as f:

#     # SCAMP scripts
#     f.write("scamp -c config.scamp ")
#     f.write(out_cat+" ")
#     f.write("-REF_SERVER vizier.unistra.fr ")
#     f.write("-ASTREF_CATALOG GAIA-EDR3 -MAGZERO_OUT 25.0 ")
#     f.write("-SOLVE_ASTROM Y -SOLVE_PHOTOM N ")
#     f.write("-FWHM_THRESHOLDS 1.5,5.0 ")
#     f.write("\n")

# os.system("sh "+sh_file)


# # # ----- Making weight images ----- #
# # for i in np.arange(n_img):
# #     dat, hdr = fits.getdata(imglist[i], header=True)
# #     exptime = hdr['EXPTIME']
# #     wei = np.ones_like(dat) * exptime

# #     avg, med, std = sigma_clipped_stats(dat[dat > -3.2e+4], sigma=2.5, maxiters=5)
# #     msk = (dat < -3.2e+4)# | (dat > 1.0e+4))
# #     wei[msk] = 0.0

# #     print("Writing "+imglist[i].split(".fits")[0]+".weight.fits...")
# #     fits.writeto(imglist[i].split(".fits")[0]+".weight.fits", wei, hdr, overwrite=True)


# # ----- Writing the SWARP configuration file ----- #
# os.system("rm -rfv config.swarp")
# cfg_file = "config.swarp"
# os.system("swarp -dd > "+cfg_file)

# sh_file = "swarp.sh"
# with open(sh_file, "w") as f:

#     # SWARP scripts
#     # for i in np.arange(n_cat):
#     f.write("swarp "+ref_img+f"[{extn:d}] -c config.swarp ")
#     f.write("-IMAGEOUT_NAME "+out_img+" ")
#     f.write("-HEADER_ONLY N ")
#     # f.write("-WEIGHTOUT_NAME "+out_img.split(".fits")[0]+".weight.fits ")
#     f.write("-WEIGHT_TYPE NONE -BLANK_BADPIXELS N ")#MAP_WEIGHT ")
#     f.write(f"-PIXELSCALE_TYPE MANUAL -PIXEL_SCALE {pxs:.3f} ")
#     f.write("-CENTER_TYPE MANUAL -CENTER "+ra0+","+dec0+" ")
#     f.write(f"-IMAGE_SIZE {xsz0:d},{ysz0:d} ")
#     # f.write("-SATLEV_DEFAULT 100. ")
#     f.write("-SUBTRACT_BACK N ")#-BACK_SIZE 256 ")
#     f.write("\n")

# os.system("sh "+sh_file)


# # ----- Reproject the HST/JWST images ----- #
# h_ref = fits.getheader(out_img)
# shape_ref = (h_ref['NAXIS2'], h_ref['NAXIS1'])
# hdr0 = fits.PrimaryHDU()
# for keys in ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
#              'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
#              'NAXIS1', 'NAXIS2', 'CTYPE1', 'CTYPE2']:
#     hdr0.header[keys] = h_ref[keys]
# w_ref = wcs.WCS(hdr0)

# # For HST images
# for i in np.arange(n_hst):
#     filt = hb_img[i].split("/")[-1].split("_")[0]
#     print("... Reprojecting HST f"+filt+"w ...")

#     # For new FITS files
#     # fhd0 = fits.PrimaryHDU()
#     # fhd1 = fits.ImageHDU()

#     d0, h0 = fits.getdata(hb_img[i], ext=0, header=True)    # Data & Header
#     w0 = wcs.WCS(h0)
#     array, footprint = reproject_interp((d0, w0), w_ref, shape_out=shape_ref,
#                                         order='nearest-neighbor')
#     for keys in ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
#                  'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']:
#         h0[keys] = h_ref[keys]

#     fits.writeto("h"+filt+".fits", array, h0, overwrite=True)

#     # fhd0.header = h0
#     # fhd1.data = array
#     # fhd1.header = h0

#     # fcb_hdu = fits.HDUList([fhd0, fhd1])
#     # fcb_hdu.writeto("h"+filt+".fits", overwrite=True)

# # For JWST images
# for i in np.arange(n_jwst):
#     filt = jw_img[i].split("/")[-1].split(".fits")[0][1:4]
#     print("... Reprojecting JWST f"+filt+"w ...")

#     # For new FITS files
#     fhd0 = fits.PrimaryHDU()
#     fhd1 = fits.ImageHDU()
#     fhd2 = fits.ImageHDU()

#     h0 = fits.getheader(jw_img[i], ext=0)    # Header
#     d1, h1 = fits.getdata(jw_img[i], header=True, ext=1)    # SCI
#     d2, h2 = fits.getdata(jw_img[i], header=True, ext=2)    # VAR

#     w1 = wcs.WCS(h1)
#     array, footprint = reproject_interp((d1, w1), w_ref, shape_out=shape_ref,
#                                         order='nearest-neighbor')
#     for keys in ['CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
#                  'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']:
#         h0[keys] = h_ref[keys]
#         h1[keys] = h_ref[keys]
#         h2[keys] = h_ref[keys]
#     fhd1.data = array

#     d2[np.isnan(d2)] = 0.
#     array, footprint = reproject_interp((d2, w1), w_ref, shape_out=shape_ref,
#                                         order='nearest-neighbor') 
#     fhd2.data = array

#     fhd0.header = h0
#     fhd1.header = h1
#     h2['EXTNAME'] = 'VAR'
#     fhd2.header = h2

#     fcb_hdu = fits.HDUList([fhd0, fhd1, fhd2])
#     fcb_hdu.writeto("j"+filt+".fits", overwrite=True)


