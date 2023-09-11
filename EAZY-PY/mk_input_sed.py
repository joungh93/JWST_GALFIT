#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 23:37:43 2022

@author: jlee
"""

import numpy as np
import glob, os
import pandas as pd
import pickle
from pystilts import wcs_match1

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import extinction
from dustmaps.sfd import SFDQuery

from eazy import filters
dir_eazy = "/data01/jhlee/Downloads/eazy-photoz/"
res = filters.FilterFile(dir_eazy+"filters/FILTER.RES.latest")

import warnings
warnings.filterwarnings("ignore")


# ----- Loading the photometry data ----- #
dir_phot = os.path.abspath("../Phot/")+"/"
dir_eazy_input = "EAZY_INPUT/"
if (glob.glob(dir_eazy_input) == []):
    os.system("mkdir "+dir_eazy_input)
else:
    os.system("rm -rfv "+dir_eazy_input+"*")

### Load the photometric data
with open(dir_phot+"phot_data.pickle", 'rb') as fr:
    phot_data = pickle.load(fr)
n_cold = np.sum(phot_data['nir_detect']['detect_flag'].values == 'cold')
n_hot  = np.sum(phot_data['nir_detect']['detect_flag'].values == 'hot')

### Segmentation image data
seg_dict = {}
seg_dict['cold'] = fits.getdata(dir_phot+"nir_detect_c_segm.fits")
seg_dict['hot']  = fits.getdata(dir_phot+"nir_detect_h_segm.fits")


# ----- HST/JWST Bandpass (to be revised manually!) ----- #
bands_hst  = ['f435w', 'f606w', 'f775w', 'f814w', 'f850lp',
              'f105w', 'f125w', 'f160w']
bands_jwst = ['f115w', 'f150w', 'f200w',
              'f277w', 'f356w', 'f444w']
n_hst = len(bands_hst)
n_jwst = len(bands_jwst)

id_flt_hst  = [233, 236, 238, 239, 240, 202, 203, 205]    # from 'FILTER.RES.latest.info' file
id_flt_jwst = [364, 365, 366, 375, 376, 377]    # from 'FILTER.RES.latest.info' file

### From 'FILTER.RES.latest.info' file
def get_centwave(id_filters, res):
    wave = []
    for i in id_filters:
        str_res = str(res[i]).split()
        idx_lam = str_res.index('lambda_c=')+1
        wave.append(float(str_res[idx_lam]))
    return np.array(wave)

wave_hst  = get_centwave(id_flt_hst, res)
wave_jwst = get_centwave(id_flt_jwst, res)

### Selecting sources (extended + point)
mag_cnd  = np.ones_like(phot_data['nir_detect']['num'].values, dtype=bool)
merr_cnd = np.ones_like(phot_data['nir_detect']['num'].values, dtype=bool)
size_cnd = np.ones_like(phot_data['nir_detect']['num'].values, dtype=bool)
for band in ['f200w', 'f277w', 'f356w', 'f444w']:
    mag_cnd  = (mag_cnd  & (phot_data[band]['mag_aper'] < 30.0))
    merr_cnd = (merr_cnd & (phot_data[band]['e_mag_aper'] < 1.0))
    size_cnd = (size_cnd & (phot_data[band]['flxrad'] > 0.0))
eff_cnd = (mag_cnd & merr_cnd & size_cnd)
print(np.sum(eff_cnd))

col_cnd = ((phot_data['f200w']['mag_auto'] - phot_data['f277w']['mag_auto'] >= -1.5) & \
           (phot_data['f200w']['mag_auto'] - phot_data['f277w']['mag_auto'] <=  1.5) & \
           (phot_data['f277w']['mag_auto'] - phot_data['f356w']['mag_auto'] >= -1.5) & \
           (phot_data['f277w']['mag_auto'] - phot_data['f356w']['mag_auto'] <=  1.5))

### Selecting only extended sources
pxs = 0.04    # arcsec/pixel
IR_mag = phot_data['nir_detect']['mag_aper']
Size50 = pxs * phot_data['nir_detect']['flxrad']

poi_cnd = (mag_cnd & merr_cnd & size_cnd & \
           ((Size50 <= 0.09) | (Size50 <= ((0.09-0.11)/(25.0-22.5))*(IR_mag-25.0)+0.09)))
print(np.sum(poi_cnd))

gal_cnd = (mag_cnd & merr_cnd & size_cnd & col_cnd & \
           (phot_data['nir_detect']['flag'] <= 4) & \
           (~poi_cnd))
           # (phot_data['nir_detect']['flxrad'] >= 2.25))    # (phot_data['f200w']['cl'] < 0.4)
print(np.sum(gal_cnd))

### Filter information & extinction
dir_img = os.path.abspath("../Reproject/")+"/"
hdr = fits.getheader(dir_img+"f444w.fits")
xsz, ysz = hdr['NAXIS1'], hdr['NAXIS2']
ra , dec = hdr['CRVAL1'], hdr['CRVAL2']

### Image data
img_dict = {}
for b in bands_hst + bands_jwst:
    img_dict[b] = fits.getdata(dir_img+b+".fits")

sfd = SFDQuery()
coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
ebv_sfd = sfd(coords)
R_V = 3.1
A_V = R_V * ebv_sfd

Amags_hst  = extinction.fitzpatrick99(wave_hst, A_V, R_V, unit='aa')
print(Amags_hst)
Amags_jwst = extinction.fitzpatrick99(wave_jwst, A_V, R_V, unit='aa')
print(Amags_jwst)


# ----- Flux calculations ----- #
obj_cnd = gal_cnd  #(gal_cnd | poi_cnd)
idx_obj = phot_data['nir_detect'][obj_cnd]['num'].values-1
n_obj = np.sum(obj_cnd)
print(n_obj)

### HST
mag_AB_hst, e_mag_AB_hst = np.zeros((n_obj, n_hst)), np.zeros((n_obj, n_hst))
for i in range(n_hst):
    mag_AB_hst[:, i] = phot_data[bands_hst[i]].loc[obj_cnd]['mag_corr']   - Amags_hst[i]
    e_mag_AB_hst[:, i] = np.maximum(0.1, phot_data[bands_hst[i]].loc[obj_cnd]['e_mag_corr'])

### JWST
mag_AB_jwst, e_mag_AB_jwst = np.zeros((n_obj, n_jwst)), np.zeros((n_obj, n_jwst))
for i in range(n_jwst):
    mag_AB_jwst[:, i] = phot_data[bands_jwst[i]].loc[obj_cnd]['mag_corr'] - Amags_jwst[i]
    e_mag_AB_jwst[:, i] = np.maximum(0.1, phot_data[bands_jwst[i]].loc[obj_cnd]['e_mag_corr'])

### Flux to micro-Jansky
def mag_to_uJy(mag_AB, e_mag_AB, index, bands, phot_data, img_dict, seg_dict,
               magzero=23.90, maglim=30.0, telescope='hst', xsize=100, ysize=100):
    
    Fv_min = 10.0 ** ((magzero-maglim)/2.5)
    Fv = 10.0 ** ((magzero-mag_AB)/2.5)   # micro-Jansky
    e_Fv = Fv * (np.log(10.0)/2.5) * e_mag_AB
    e_Fv[np.isnan(e_Fv)] = Fv[np.isnan(e_Fv)] / 10.
    Fv[(Fv < 1.0e-10) | (e_Fv < 1.0e-10) | (Fv < Fv_min)] = np.nan
    e_Fv[(Fv < 1.0e-10) | (e_Fv < 1.0e-10) | (Fv < Fv_min)] = np.nan
    
    for i in range(Fv.shape[0]):
        Fv_med = np.nanmedian(Fv[i, :])
        if np.isnan(Fv_med):
            continue
        else:
            Fv[i, :][Fv[i, :] < Fv_med/100.] = np.nan
    
    for i in range(Fv.shape[0]):
        num = phot_data['nir_detect']['num'].values[index[i]]
        a = phot_data['nir_detect']['a'].values[index[i]]
        kr = phot_data['nir_detect']['kron'].values[index[i]]
        
        x0 = np.minimum(np.maximum(0, phot_data['nir_detect']['x'].values[index[i]]-1-1.5*a*kr), xsize-1)
        y0 = np.minimum(np.maximum(0, phot_data['nir_detect']['y'].values[index[i]]-1-1.5*a*kr), ysize-1)
        x1 = np.minimum(np.maximum(0, phot_data['nir_detect']['x'].values[index[i]]-1+1.5*a*kr), xsize-1)
        y1 = np.minimum(np.maximum(0, phot_data['nir_detect']['y'].values[index[i]]-1+1.5*a*kr), ysize-1)
        
        seg_data = seg_dict[phot_data['nir_detect']['detect_flag'].values[index[i]]]
        
        for j in range(Fv.shape[1]):
            img_data  = img_dict[bands[j]]
            img_split = img_data[int(y0):int(y1)+1, int(x0):int(x1)+1]
            seg_split = seg_data[int(y0):int(y1)+1, int(x0):int(x1)+1]
            npix_zero = np.sum(img_split[seg_split == num] == 0)
            if (npix_zero > 10):
                Fv[i, j] = np.nan
            if ((phot_data[bands[j]]['mag_aper'].values[index[i]] >= 30.0) | \
                (phot_data[bands[j]]['e_mag_aper'].values[index[i]] >= 1.0) | \
                (phot_data[bands[j]]['flxrad'].values[index[i]] <= 0.0)):
                Fv[i, j] = np.nan

    # for i in range(Fv.shape[0]):
    #     for j in range(Fv.shape[1]):
    #         if (telescope == 'hst'):
    #             if (j == 0):
    #                 continue
    #             else:
    #                 if (j == range(Fv.shape[1])[-1]):
    #                     Fv_nei = Fv[i, -2]
    #                 else:
    #                     Fv_nei = np.nansum(Fv[i, j-1] + Fv[i, j+1]) / 2.
    #                 if ((Fv_nei > 0.) & (~np.isnan(Fv_nei)) & (~np.isnan(Fv[i, j]))):
    #                     if (Fv[i, j] < Fv_nei / 2.5):
    #                         Fv[i, j] = np.nan
    #         if (telescope == 'jwst'):
    #             if (j == 0):
    #                 Fv_nei = Fv[i, 1]
    #             elif (j == range(Fv.shape[1])[-1]):
    #                 continue
    #             else:
    #                 Fv_nei = np.nansum(Fv[i, j-1] + Fv[i, j+1])/2.
    #             if ((Fv_nei > 0.) & (~np.isnan(Fv_nei)) & (~np.isnan(Fv[i, j]))):
    #                 if (Fv[i, j] < Fv_nei / 2.5):
    #                     Fv[i, j] = np.nan  

    Fv[(np.isnan(Fv)) | (np.isnan(e_Fv))] = -99.
    e_Fv[(np.isnan(Fv)) | (np.isnan(e_Fv))] = -99.

    return [Fv, e_Fv]

# def mag_to_uJy(mag_AB, e_mag_AB, index, bands, phot_data, img_dict, seg_dict,
#                magzero=23.90, maglim=30.0, telescope='hst', xsize=100, ysize=100):

Fv_hst , e_Fv_hst   = mag_to_uJy(mag_AB_hst, e_mag_AB_hst, idx_obj, bands_hst,
                                 phot_data, img_dict, seg_dict,
                                 magzero=23.90, maglim=30.0, telescope='hst',
                                 xsize=xsz, ysize=ysz)

Fv_jwst, e_Fv_jwst  = mag_to_uJy(mag_AB_jwst, e_mag_AB_jwst, idx_obj, bands_jwst,
                                 phot_data, img_dict, seg_dict,
                                 magzero=23.90, maglim=30.0, telescope='jwst',
                                 xsize=xsz, ysize=ysz)


# ----- Reading the spectroscopic catalogs ----- #
dir_ned = os.path.abspath("../Archive/")+"/"
df_ned = pd.read_csv(dir_ned+"NED_NGDEEP.csv")
# df_ned.head(6)

gal_ned = ~(df_ned['Type'].str.startswith('!').values | df_ned['Type'].str.endswith('*').values)
spz_ned = np.in1d(df_ned['Redshift Flag'].values,
              np.array(['S1L', 'SLS', 'SMU', 'SSN', 'SST']))
zeff = (gal_ned & spz_ned & ~np.isnan(df_ned['Redshift'].values))
z_spec = np.where(zeff, df_ned['Redshift'].values, -1.0)

maglim = 27.0  #25.0

### Matching
Mag200 = phot_data['f200w']['mag_aper'].values
obj_mat = (obj_cnd & (Mag200 < maglim))
print(np.sum(obj_mat), len(df_ned))

tol = 1.5   # arcsec
idx_matched, idx_spec, sepr = wcs_match1(phot_data['nir_detect'].loc[obj_mat]['ra'].values,
                                         phot_data['nir_detect'].loc[obj_mat]['dec'].values,
                                         df_ned['RA'].values, df_ned['DEC'].values, tol, ".")
print(len(idx_matched))


# ----- Write the region files ----- #
reg_files = ["ned_total.reg", "ned_specz.reg", "ned_matched.reg",
             "src_objtot.reg", "src_objmat.reg"]
reg_cols  = ["yellow", "red", "magenta",
             "cyan", "green"]
n_reg     = [len(df_ned), np.sum(zeff), len(idx_matched),
             np.sum(obj_cnd), np.sum(obj_mat)]
reg_RA    = [df_ned['RA'].values , df_ned['RA'].values[zeff] , df_ned['RA'].values[idx_spec] ,
             phot_data['nir_detect'].loc[obj_cnd]['ra'].values, phot_data['nir_detect'].loc[obj_mat]['ra'].values]
reg_DEC   = [df_ned['DEC'].values, df_ned['DEC'].values[zeff], df_ned['DEC'].values[idx_spec],
             phot_data['nir_detect'].loc[obj_cnd]['dec'].values, phot_data['nir_detect'].loc[obj_mat]['dec'].values]
reg_rad   = [1.0, 1.5, 2.0, 0.5, 0.75]

for i in range(len(reg_files)):
    with open(reg_files[i], "w") as f:
        f.write('global color='+reg_cols[i]+' font="helvetica 10 normal" ')
        f.write("select=1 edit=1 move=1 delete=1 include=1 fixed=0 source width=2\n")
        f.write("fk5\n")
        for j in range(n_reg[i]):
            f.write(f"circle({reg_RA[i][j]:.6f}, {reg_DEC[i][j]:.6f}, {reg_rad[i]:.1f}")
            f.write('")\n')  # text={'+f"{z_spec[idx_spec][i]:.4f}"+'}\n')


# ----- Writing input files ----- #
def write_input_file(filename, objid, z_spec, ra, dec, flux, err_flux, id_filter):
    f = open(filename, "w")
    columns = "# id z_spec ra dec "
    for i in range(len(id_filter)):
        columns += f"F{id_filter[i]:d} E{id_filter[i]:d} "
    f.write(columns+"\n")
    for j in range(len(objid)):
        txt = f"{objid[j]:d} {z_spec[j]:.4f} {ra[j]:.5f} {dec[j]:.5f} "
        for i in range(len(id_filter)):
            txt += f"{flux[j, i]:5.3e} {err_flux[j, i]:5.3e} "
        f.write(txt+"\n")
    f.close()

z_spec1 = -1.0 * np.ones_like(phot_data['nir_detect'].loc[obj_cnd]['num'].values)
idx_bri = np.arange(len(z_spec1))[Mag200[obj_cnd] < maglim]
idx_brm = idx_bri[idx_matched]
z_spec1[idx_brm] = z_spec[idx_spec]
print(np.sum(z_spec1 > 0.))

z_spec2 = -1.0 * np.ones_like(phot_data['nir_detect'].loc[obj_mat]['num'].values)
z_spec2[idx_matched] = z_spec[idx_spec]
print(np.sum(z_spec2 > 0.))


# Initial Run 1 (HST+JWST, All, w/ spec-z sources)
write_input_file(dir_eazy_input+"flux_EAZY_run1z.cat",
                 phot_data['nir_detect'].loc[obj_mat]['num'].values[z_spec2 > 0.],
                 z_spec2[z_spec2 > 0.],
                 phot_data['nir_detect'].loc[obj_mat]['ra'].values[z_spec2 > 0.],
                 phot_data['nir_detect'].loc[obj_mat]['dec'].values[z_spec2 > 0.],
                 np.column_stack([Fv_hst[Mag200[obj_cnd] < maglim][z_spec2 > 0.],
                                  Fv_jwst[Mag200[obj_cnd] < maglim][z_spec2 > 0.]]),
                 np.column_stack([e_Fv_hst[Mag200[obj_cnd] < maglim][z_spec2 > 0.],
                                  e_Fv_jwst[Mag200[obj_cnd] < maglim][z_spec2 > 0.]]),
                 id_flt_hst + id_flt_jwst)

# Initial Run 2 (HST+JWST, -Bluest, w/ spec-z sources)
write_input_file(dir_eazy_input+"flux_EAZY_run2z.cat",
                 phot_data['nir_detect'].loc[obj_mat]['num'].values[z_spec2 > 0.],
                 z_spec2[z_spec2 > 0.],
                 phot_data['nir_detect'].loc[obj_mat]['ra'].values[z_spec2 > 0.],
                 phot_data['nir_detect'].loc[obj_mat]['dec'].values[z_spec2 > 0.],
                 np.column_stack([Fv_hst[Mag200[obj_cnd] < maglim][z_spec2 > 0.][:, 1:],
                                  Fv_jwst[Mag200[obj_cnd] < maglim][z_spec2 > 0.]]),
                 np.column_stack([e_Fv_hst[Mag200[obj_cnd] < maglim][z_spec2 > 0.][:, 1:],
                                  e_Fv_jwst[Mag200[obj_cnd] < maglim][z_spec2 > 0.]]),
                 id_flt_hst[1:] + id_flt_jwst)

# Initial Run 3 (HST+JWST, All, w/ bright sources)
write_input_file(dir_eazy_input+"flux_EAZY_run3z.cat",
                 phot_data['nir_detect'].loc[obj_mat]['num'].values,
                 z_spec2,
                 phot_data['nir_detect'].loc[obj_mat]['ra'].values,
                 phot_data['nir_detect'].loc[obj_mat]['dec'].values,
                 np.column_stack([Fv_hst[Mag200[obj_cnd] < maglim],
                                  Fv_jwst[Mag200[obj_cnd] < maglim]]),
                 np.column_stack([e_Fv_hst[Mag200[obj_cnd] < maglim],
                                  e_Fv_jwst[Mag200[obj_cnd] < maglim]]),
                 id_flt_hst + id_flt_jwst)

# Initial Run 4 (HST+JWST, -Bluest, w/ bright sources)
write_input_file(dir_eazy_input+"flux_EAZY_run4z.cat",
                 phot_data['nir_detect'].loc[obj_mat]['num'].values,
                 z_spec2,
                 phot_data['nir_detect'].loc[obj_mat]['ra'].values,
                 phot_data['nir_detect'].loc[obj_mat]['dec'].values,
                 np.column_stack([Fv_hst[Mag200[obj_cnd] < maglim][:, 1:],
                                  Fv_jwst[Mag200[obj_cnd] < maglim]]),
                 np.column_stack([e_Fv_hst[Mag200[obj_cnd] < maglim][:, 1:],
                                  e_Fv_jwst[Mag200[obj_cnd] < maglim]]),
                 id_flt_hst[1:] + id_flt_jwst)

os.system("cp -rpv "+dir_eazy_input+"flux_EAZY_run1z.cat "+dir_eazy_input+"flux_EAZY_run1i.cat")
os.system("cp -rpv "+dir_eazy_input+"flux_EAZY_run2z.cat "+dir_eazy_input+"flux_EAZY_run2i.cat")
os.system("cp -rpv "+dir_eazy_input+"flux_EAZY_run1z.cat "+dir_eazy_input+"flux_EAZY_run1p.cat")
os.system("cp -rpv "+dir_eazy_input+"flux_EAZY_run2z.cat "+dir_eazy_input+"flux_EAZY_run2p.cat")
os.system("cp -rpv "+dir_eazy_input+"flux_EAZY_run3z.cat "+dir_eazy_input+"flux_EAZY_run3i.cat")
os.system("cp -rpv "+dir_eazy_input+"flux_EAZY_run4z.cat "+dir_eazy_input+"flux_EAZY_run4i.cat")

# Run 5 (HST+JWST, All, w/ all extended sources)
write_input_file(dir_eazy_input+"flux_EAZY_run5z.cat",
                 phot_data['nir_detect'].loc[obj_cnd]['num'].values,
                 z_spec1,
                 phot_data['nir_detect'].loc[obj_cnd]['ra'].values,
                 phot_data['nir_detect'].loc[obj_cnd]['dec'].values,
                 np.column_stack([Fv_hst, Fv_jwst]),
                 np.column_stack([e_Fv_hst, e_Fv_jwst]),
                 id_flt_hst + id_flt_jwst)

# Run 6 (HST+JWST, -Bluest, w/ all extended sources)
write_input_file(dir_eazy_input+"flux_EAZY_run6z.cat",
                 phot_data['nir_detect'].loc[obj_cnd]['num'].values,
                 z_spec1,
                 phot_data['nir_detect'].loc[obj_cnd]['ra'].values,
                 phot_data['nir_detect'].loc[obj_cnd]['dec'].values,
                 np.column_stack([Fv_hst[:, 1:], Fv_jwst]),
                 np.column_stack([e_Fv_hst[:, 1:], e_Fv_jwst]),
                 id_flt_hst[1:] + id_flt_jwst)

