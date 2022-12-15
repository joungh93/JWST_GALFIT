#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:41:35 2020
@author: jlee
"""


import numpy as np
from astropy.io import fits
from astropy import wcs
import sep
import pandas as pd
import warnings
# import init_param as ip

class sep0:

    # ----- Initialize ----- #
    def __init__(self, input_image, img_extn=0, hdr_extn=0, tmp_dir="tmp",
                 bw=32, bh=32, fw=3, fh=3,
                 thresh=3.0, minarea=5, db_nth=32, db_cont=0.005,
                 output_segment=False):
        self.img = input_image
        # self.hdu = fits.open(input_image)[extn]
        dat = fits.getdata(input_image, ext=img_extn, header=False)
        hdr = fits.getheader(input_image, ext=hdr_extn)
        self.dat = dat.byteswap().newbyteorder()
        self.hdr = hdr
        # self.dat = self.hdu.data.byteswap().newbyteorder()
        # self.hdr = self.hdu.header
        self.wcs = wcs.WCS(self.hdr)
        self.zmag = 25.0
        # -2.5*np.log10(self.hdr['PHOTFLAM'])-5.0*np.log10(self.hdr['PHOTPLAM'])-2.408
        self.gain = 1.0#self.hdr['CCDGAIN']
        # self.exptime = self.hdr['EXPTIME']

        # Background estimation
        bkg = sep.Background(self.dat, mask=None, bw=bw, bh=bh, fw=fw, fh=fh)
        sky, sig = bkg.globalback, bkg.globalrms
        self.backgr = bkg.back()
        self.sky = sky
        self.skysigma = sig

        # Source detection
        dat_sub = self.dat - self.backgr
        src = sep.extract(dat_sub, thresh, err=self.skysigma,
                          minarea=minarea, deblend_nthresh=db_nth, deblend_cont=db_cont,
                          segmentation_map=output_segment)
        src['theta'][src['theta'] < -np.pi/2.] = -np.pi/2. + 1.0e-7
        src['theta'][src['theta'] > np.pi/2.] = np.pi/2. - 1.0e-7

        self.src = src
        self.nsrc = len(src)
        print(input_image+': {0:d} sources extracted.'.format(self.nsrc))

        if (tmp_dir[-1] != "/"):
            tmp_dir += "/"
        self.tmp_dir = tmp_dir

        warnings.filterwarnings("ignore")#, category=RuntimeWarning) 

    # ----- Aperture photometry: MAG_AUTO ----- #
    def AutoPhot(self, Kron_fact=2.5, min_diameter=3.5, write=True):
        kronrad, krflag = sep.kron_radius(self.dat, self.src['x'], self.src['y'],
                                          self.src['a'], self.src['b'], self.src['theta'], 6.)
        kronrad[np.isnan(kronrad) == True] = 0.
        flux, fluxerr, flag = sep.sum_ellipse(self.dat, self.src['x'], self.src['y'],
                                              self.src['a'], self.src['b'], self.src['theta'],
                                              Kron_fact*kronrad, err=self.skysigma, gain=self.gain, subpix=0)
        flag |= krflag    # Combining flags
        r_min = 0.5*min_diameter

        use_circle = kronrad * np.sqrt(self.src['a']*self.src['b']) < r_min
        cflux, cfluxerr, cflag = sep.sum_circle(self.dat, self.src['x'][use_circle], self.src['y'][use_circle],
                                                r_min, err=self.skysigma, gain=self.gain, subpix=0)
        flux[use_circle] = cflux
        fluxerr[use_circle] = cfluxerr
        flag[use_circle] = cflag

        mag = self.zmag - 2.5*np.log10(flux)
        magerr = (2.5/np.log(10.0)) * (fluxerr/flux)

        r, flag = sep.flux_radius(self.dat, self.src['x'], self.src['y'], 6.0*self.src['a'],
                                  0.5, normflux=flux, subpix=5)

        ra, dec = self.wcs.all_pix2world(self.src['x']+1, self.src['y']+1, 1)

        df = pd.DataFrame(data = {'x': self.src['x'], 'y': self.src['y'], 'ra': ra, 'dec': dec,
                                  'a': self.src['a'], 'b': self.src['b'], 'theta': self.src['theta'],
                                  'flux': flux, 'e_flux': fluxerr,
                                  'mag': mag, 'e_mag': magerr,
                                  'kronrad': kronrad, 'flxrad': r,
                                  'flag': flag})

        if write:
            df.to_csv(self.tmp_dir+'auto_'+self.img.split('.fits')[0]+'.csv')

            f = open(self.tmp_dir+'src_'+self.img.split('.fits')[0]+'.reg','w')
            for i in np.arange(self.nsrc):
                f.write('{0:.3f}  {1:.3f}\n'
                    .format(self.src['x'][i]+1, self.src['y'][i]+1))
            f.close()

        return df

    # ----- Aperture photometry: MAG_APER ----- #
    def AperPhot(self, r_aper=[1.5, 2.0, 4.0], write=True):

        ra, dec = self.wcs.all_pix2world(self.src['x']+1, self.src['y']+1, 1)

        df = pd.DataFrame(data = {'x': self.src['x'], 'y': self.src['y'], 'ra': ra, 'dec': dec,
                                  'a': self.src['a'], 'b': self.src['b'], 'theta': self.src['theta']})

        for i in np.arange(len(r_aper)):
            flux, fluxerr, flag = sep.sum_circle(self.dat, self.src['x'], self.src['y'], r_aper[i],
                                                 err=self.skysigma, gain=self.gain, subpix=0)
            mag = self.zmag - 2.5*np.log10(flux)
            magerr = (2.5/np.log(10.0)) * (fluxerr/flux)

            df['r'+f'{i+1:d}'] = r_aper[i]
            df['flux'+f'{i+1:d}'] = flux
            df['e_flux'+f'{i+1:d}'] = fluxerr
            df['mag'+f'{i+1:d}'] = mag
            df['e_mag'+f'{i+1:d}'] = magerr

        if write:
            df.to_csv(self.tmp_dir+'aper_'+self.img.split('.fits')[0]+'.csv')

            f = open(self.tmp_dir+'src_'+self.img.split('.fits')[0]+'.reg','w')
            for i in np.arange(self.nsrc):
                f.write('{0:.3f}  {1:.3f}\n'
                    .format(self.src['x'][i]+1, self.src['y'][i]+1))
            f.close()

        return df

    # ----- Point source selection (w/ AutoPhot) ----- #
    def Pick_PointSrc(self, mag_high=20.0, mag_low=26.0, size_low=1.0, size_high=2.0, ar_cut=0.75, write=True):
        df = self.AutoPhot()
        poi = ((df['mag'].values > mag_high) & (df['mag'].values < mag_low) & \
               (df['flxrad'].values > size_low) & (df['flxrad'].values < size_high) & \
               (df['b'].values/df['a'].values > ar_cut))
        npoi = np.sum(poi)

        if write:
            f = open(self.tmp_dir+'poi_'+self.img.split('.fits')[0]+'.reg','w')
            for i in np.arange(npoi):
                f.write('{0:.3f}  {1:.3f}\n'
                    .format(df['x'].values[poi][i]+1, df['y'].values[poi][i]+1))
            f.close()

        return df[poi]


# if (__name__ == '__main__'):

