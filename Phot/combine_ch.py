# Printing the versions of packages
import time
start_time = time.time()

from importlib_metadata import version
for pkg in ['numpy', 'astropy', 'pandas']:
    print(pkg+": ver "+version(pkg))


# importing necessary modules
import numpy as np
import glob, os, copy
from pathlib import Path
from astropy.io import fits
from astropy import wcs
import tqdm
import pandas as pd

import warnings
warnings.filterwarnings(action='ignore')


# ----- Directories ----- #
def get_dirs():
    dir_root = str(Path("../").resolve())
    dir_img  = str(Path(dir_root) / Path("Reproject"))
    dir_phot = str(Path(dir_root) / Path("Phot"))
    return [dir_root, dir_img, dir_phot]

dir_root, dir_img, dir_phot = get_dirs()


# ----- Reading source catalogs from detection image ----- #
dat = fits.getdata(dir_img+"/"+"nir_detect.fits")
colnames = ['x','y','num','flux_aper','e_flux_aper','flux_auto','e_flux_auto','kron','petro',
            'backgr','ra','dec','a','b','theta','mu0','flag','fwhm','flxrad','cl','cxx','cyy','cxy']
phot_c = np.genfromtxt('nir_detect_c.cat', dtype=None, encoding='ascii', names=colnames)
phot_h = np.genfromtxt('nir_detect_h.cat', dtype=None, encoding='ascii', names=colnames)

xsz, ysz = dat.shape[1], dat.shape[0]
x  , y   = np.arange(xsz), np.arange(ysz)
xx , yy  = np.meshgrid(x, y, sparse=True)


# ----- Making Kron-aperture images ----- #

# apr1 = np.zeros_like(dat)
# for i in tqdm.trange(10):
#   xc, yc = phot_c['x'][i]-1, phot_c['y'][i]-1
#   cxx, cyy, cxy = phot_c['cxx'][i], phot_c['cyy'][i], phot_c['cxy'][i]
#   kron = phot_c['kron'][i]
#   z = cxx*(xx-xc)**2. + cyy*(yy-yc)**2. + cxy*(xx-xc)*(yy-yc) - kron**2.
#   apr1[z < 0.] = 1.
# fits.writeto("apr1.fits", apr1, overwrite=True)
# print(np.sum(apr1))

# mag_cnd  = (phot_c['mag_auto'] < 30.0)
# merr_cnd = (phot_c['e_mag_auto'] < 1.0)
# size_cnd = (phot_c['flxrad'] > 0.0)
# eff_cnd  = (size_cnd & \
#           (phot_c['flag'] <= 4) & \
#           (phot_c['kron'] > 0.))

apr2 = np.zeros_like(dat)
for i in tqdm.trange(len(phot_c)):
    a, b, kr, fl, cl = phot_c['a'][i], phot_c['b'][i], phot_c['kron'][i], phot_c['flag'][i], phot_c['cl'][i]
    if ((kr <= 0.) | (28.9-2.5*np.log10(phot_c['flux_aper'][i]) > 30.)):
        continue
    if (a*kr > 100):
        if ((fl <= 2) & (cl < 0.4)):
            pass
        else:
            continue
    x0 = np.minimum(np.maximum(0, phot_c['x'][i]-1-1.5*a*kr), xsz-1)
    y0 = np.minimum(np.maximum(0, phot_c['y'][i]-1-1.5*a*kr), ysz-1)
    x1 = np.minimum(np.maximum(0, phot_c['x'][i]-1+1.5*a*kr), xsz-1)
    y1 = np.minimum(np.maximum(0, phot_c['y'][i]-1+1.5*a*kr), ysz-1)
    xc, yc = phot_c['x'][i]-1-x0, phot_c['y'][i]-1-y0
    cxx, cyy, cxy = phot_c['cxx'][i], phot_c['cyy'][i], phot_c['cxy'][i]
    apr_split = copy.deepcopy(apr2[int(y0):int(y1)+1, int(x0):int(x1)+1])
    xxs, yys  = np.meshgrid(np.arange(apr_split.shape[1]), np.arange(apr_split.shape[0]), sparse=True)
    zz = cxx*(xxs-xc)**2. + cyy*(yys-yc)**2. + cxy*(xxs-xc)*(yys-yc) - kr**2.
    apr_split[zz <= 0.] = 1.
    apr2[int(y0):int(y1)+1, int(x0):int(x1)+1] = apr_split
fits.writeto("apr_cold.fits", apr2, overwrite=True)
print(np.sum(apr2))


# ----- Combining cold+hot detected sources ----- #
dfm = pd.DataFrame(phot_c)
dfh = pd.DataFrame(phot_h)

n_cold = len(dfm)
dfm['detect_flag'] = 'cold'
dfh['detect_flag'] = 'hot'

j = 0
num_hot = []
for i in tqdm.trange(len(phot_h)):
    if (apr2[round(dfh['y'].values[i])-1, round(dfh['x'].values[i])-1] == 1.):
        continue
    else:
        num_hot.append(dfh['num'].values[i])
        j += 1
        dfh.loc[[i], 'num'] = n_cold + j
        dfm = pd.concat([dfm, dfh.loc[[i], :]], ignore_index=True)
num_hot = np.array(num_hot)
idx_hot = num_hot-1


with open("detect_cold+hot.reg", "w") as f:
    for i in range(len(dfm)):
        f.write(f"circle({dfm['x'].values[i]:.4f}, {dfm['y'].values[i]:.4f}, 3.00)\n")
        # f.write(f"{dfm['x'].values[i]:.4f}  {dfm['y'].values[i]:.4f}\n")

dfm.to_pickle("nir_detect.pickle")


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


# ----- Reading & combining all band data ----- #
bands = band_hst + band_jwst
imglists = imglist_hst + imglist_jwst

for i in tqdm.trange(len(imglists)):
    phot_c = np.genfromtxt(bands[i]+'_c.cat', dtype=None, encoding='ascii', names=colnames)
    dfc = pd.DataFrame(phot_c)
    dfc['detect_flag'] = 'cold'

    phot_h = np.genfromtxt(bands[i]+'_h.cat', dtype=None, encoding='ascii', names=colnames)
    dfh = pd.DataFrame(phot_h)
    dfh['detect_flag'] = 'hot'

    dfh.loc[idx_hot, 'num'] = 1 + np.arange(len(dfc), len(dfm))
    dfc = pd.concat([dfc, dfh.loc[idx_hot, :]], ignore_index=True)
    dfc.to_pickle(bands[i]+".pickle")


# Printing the running time
print('--- %.4f seconds ---' %(time.time()-start_time))

