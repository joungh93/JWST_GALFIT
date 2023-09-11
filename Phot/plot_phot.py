# Printing the versions of packages
from importlib_metadata import version
for pkg in ['numpy', 'matplotlib', 'astropy', 'pandas']:
    print(pkg+": ver "+version(pkg))


# importing necessary modules
import numpy as np
import glob, os, copy
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import pickle

import warnings
warnings.filterwarnings("ignore")


# Load the Data
with open('phot_data.pickle', 'rb') as fr:
    phot_data = pickle.load(fr)


# ----- Directories ----- #
def get_dirs():
    dir_root = str(Path("../").resolve())
    dir_img  = str(Path(dir_root) / Path("Reproject"))
    dir_phot = str(Path(dir_root) / Path("Phot"))
    return [dir_root, dir_img, dir_phot]

dir_root, dir_img, dir_phot = get_dirs()
dir_fig = "Figures/"
if not os.path.exists(dir_fig):
    os.system("mkdir "+dir_fig)


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


# ----- Figures ----- #

### Magnitude - Magnitude error diagram

# HST & JWST
xlab = [b.upper()+" mag (AUTO)" for b in band_hst+band_jwst]
ylab = [b.upper()+" mag error (AUTO)" for b in band_hst+band_jwst]

for i in range(len(band_hst)+len(band_jwst)):
    if (i < len(band_hst)):
        bands = band_hst
        telname = "HST"
        idx = i
    else:
        bands = band_jwst
        telname = "JWST"
        idx = i - len(band_hst)

    fig, ax = plt.subplots(figsize=(4,4))
    mag_range = (phot_data[bands[idx]]['mag_auto'] < 99.0)
    ax.plot(phot_data[bands[idx]]['mag_auto'][mag_range],
            phot_data[bands[idx]]['e_mag_auto'][mag_range],
            'o', ms=1.0, color='gray', alpha=0.7)
    ax.set_xlim([12.5, 32.5])
    ax.set_ylim([0.0, 1.2])
    ax.set_xlabel(xlab[i])
    ax.set_ylabel(ylab[i])

    plt.tight_layout()
    plt.savefig(dir_fig + "Fig1_"+telname+"-"+bands[idx]+"-merr_vs_mag.png", dpi=300)
    plt.close()


### Magnitude - Stellarity diagram

# HST & JWST
xlab = [b.upper()+" mag (AUTO)" for b in band_hst+band_jwst]
ylab = "Stellarity (CLASS_STAR)"

for i in range(len(band_hst)+len(band_jwst)):
    if (i < len(band_hst)):
        bands = band_hst
        telname = "HST"
        idx = i
    else:
        bands = band_jwst
        telname = "JWST"
        idx = i - len(band_hst)

    fig, ax = plt.subplots(figsize=(4,4))
    mag_range = (phot_data[bands[idx]]['mag_auto'] < 99.0)
    ax.plot(phot_data[bands[idx]]['mag_auto'][mag_range],
            phot_data[bands[idx]]['cl'][mag_range],
            'o', ms=1.0, color='gray', alpha=0.7)
    ax.axhline(0.4, 0, 1, color='red', ls='--', lw=1)
    ax.axhline(0.8, 0, 1, color='red', ls='--', lw=1)
    ax.set_xlim([12.5, 32.5])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel(xlab[i])
    ax.set_ylabel(ylab)

    plt.tight_layout()
    plt.savefig(dir_fig + "Fig1_"+telname+"-"+bands[idx]+"-class_vs_mag.png", dpi=300)
    plt.close()


### Magnitude - Size diagram
band_sel = copy.deepcopy(band_jwst)
#band_sel.remove('f090w')    # Revise this manually.

mag_cnd  = np.ones_like(phot_data['nir_detect']['num'].values, dtype=bool)
merr_cnd = np.ones_like(phot_data['nir_detect']['num'].values, dtype=bool)
size_cnd = np.ones_like(phot_data['nir_detect']['num'].values, dtype=bool)
for i in range(len(band_sel)):
    mag_cnd  = (mag_cnd  & (phot_data[band_sel[i]]['mag_auto'] < 30.0))
    merr_cnd = (merr_cnd & (phot_data[band_sel[i]]['e_mag_auto'] < 1.0))
    size_cnd = (size_cnd & (phot_data[band_sel[i]]['flxrad'] > 0.0))
print(np.sum(mag_cnd & merr_cnd & size_cnd))

pxs = 0.04    # arcsec/pixel
xlab = "IR magnitude"
ylab = "Half-light Radius [arcsec]"

fig, ax = plt.subplots(figsize=(4,4))
plt_range = (mag_cnd & merr_cnd & size_cnd & \
             (phot_data['nir_detect']['flag'] <= 4))
print(np.sum(plt_range))
ax.plot(phot_data['nir_detect']['mag_aper'][plt_range],
        pxs*phot_data['nir_detect']['flxrad'][plt_range], 'o', ms=2.0, mew=0.0, alpha=0.6)
ax.plot([15.0, 25.0], [0.17, 0.09], color='red', ls='--', lw=1, alpha=0.8)
#ax.plot([15.0, 22.5], [0.11, 0.11], color='red', ls='--', lw=1, alpha=0.8)
#ax.plot([22.5, 25.0], [0.11, 0.09], color='red', ls='--', lw=1, alpha=0.8)
ax.plot([25.0, 32.5], [0.09, 0.09], color='red', ls='--', lw=1, alpha=0.8)
# ax.axhline(0.09, 0, 1, color='red', ls='--', lw=1, alpha=0.8)
# ax.axhline(0.11, 0, 1, color='red', ls='--', lw=1, alpha=0.8)
ax.set_xlim([15.0, 32.5])
ax.set_ylim([0.05, 5.0])
ax.set_yscale('log')
ax.set_xlabel(xlab)
ax.set_ylabel(ylab)

plt.tight_layout()
plt.savefig(dir_fig + "Fig1_"+telname+"-size_vs_mag.png", dpi=300)
plt.close()


# ----- For check ----- #
IR_mag = phot_data['nir_detect']['mag_aper']
Size50 = pxs * phot_data['nir_detect']['flxrad']

poi1 = (plt_range & \
        (IR_mag >= 20.0) & (IR_mag <= 22.5) & \
        (Size50 >= 0.09) & (Size50 <= 0.11))
with open("check_point1.reg", "w") as f:
    for i in range(np.sum(poi1)):
        f.write(f"{phot_data['nir_detect']['x'].values[poi1][i]:.4f}  ")
        f.write(f"{phot_data['nir_detect']['y'].values[poi1][i]:.4f}\n")

poi2 = (plt_range & \
        (((IR_mag < 22.5) & (Size50 <= 0.11)) | \
        ((IR_mag >= 22.5) & (IR_mag < 25.0) & \
         (Size50 <= ((0.09-0.11)/(25.0-22.5))*(IR_mag-25.0)-0.09)) | \
        ((IR_mag >= 25.0) & (Size50 <= 0.09))))
with open("check_point2.reg", "w") as f:
    for i in range(np.sum(poi2)):
        f.write(f"{phot_data['nir_detect']['x'].values[poi2][i]:.4f}  ")
        f.write(f"{phot_data['nir_detect']['y'].values[poi2][i]:.4f}\n")

poi3 = (plt_range & \
        (IR_mag <= 23.0) & \
        ((Size50 <= 0.09) | (Size50 <= ((0.09-0.11)/(25.0-22.5))*(IR_mag-25.0)+0.09)))
with open("check_point3.reg", "w") as f:
    for i in range(np.sum(poi3)):
        f.write(f"{phot_data['nir_detect']['x'].values[poi3][i]:.4f}  ")
        f.write(f"{phot_data['nir_detect']['y'].values[poi3][i]:.4f}\n")
# --------------------- #


### Color-magnitude diagram (revise the bands manually.)
gal_cnd = (mag_cnd & merr_cnd & size_cnd & \
           (phot_data['nir_detect']['flag'] <= 4) & \
           (pxs * phot_data['nir_detect']['flxrad'] >  0.09))
print(np.sum(gal_cnd))

poi_cnd = (mag_cnd & merr_cnd & size_cnd & \
           (phot_data['nir_detect']['flag'] <= 4) & \
           (pxs * phot_data['nir_detect']['flxrad'] <= 0.09))
print(np.sum(poi_cnd)) 

# JWST
c1 = ["f115w", "f115w", "f150w",
      "f150w", "f200w", "f200w",
      "f277w", "f200w", "f356w"] 
c2 = ["f150w", "f200w", "f200w",
      "f277w", "f277w", "f356w",
      "f356w", "f444w", "f444w"]
xlab = [c1[i].upper()+"-"+c2[i].upper() for i in range(len(c1))]
ylab = ["F277W magnitude"]

n_rows, n_cols = 3, 3
fig, axs = plt.subplots(n_rows, n_cols, figsize=(8,8))
for i, ax in enumerate(axs.flatten()):
    ax.plot(phot_data[c1[i]]['mag_corr'][gal_cnd]-phot_data[c2[i]]['mag_corr'][gal_cnd],
            phot_data['f277w']['mag_corr'][gal_cnd], 'o', ms=1)
    ax.set_xlim([-2.5, 3.5])
    ax.set_ylim([30.0, 17.0])
    ax.set_xlabel(xlab[i])
    if (i % n_cols == 0):
        ax.set_ylabel(ylab[0])
plt.tight_layout()
plt.savefig(dir_fig + "Fig1_JWST-CMDs.png", dpi=300)
plt.close()


### Color-color diagram
fig, ax = plt.subplots(figsize=(4,4))

ax.plot(phot_data['f200w']['mag_corr'][gal_cnd] - phot_data['f277w']['mag_corr'][gal_cnd],
        phot_data['f277w']['mag_corr'][gal_cnd] - phot_data['f356w']['mag_corr'][gal_cnd], 'o', ms=1, alpha=0.6)
ax.set_xlim([-2.5, 2.5])
ax.set_ylim([-2.5, 2.5])
ax.set_xlabel("F200W-F277W")
ax.set_ylabel("F277W-F356W")

plt.tight_layout()
plt.savefig(dir_fig + "Fig1_JWST-CCD1.png", dpi=300)
plt.close()


