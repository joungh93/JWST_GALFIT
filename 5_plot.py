#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 11:29:04 2022
@author: jlee
"""


import numpy as np
import glob, os, copy
from matplotlib import pyplot as plt
# import pandas as pd
from astropy.io import fits
from astropy.visualization import ZScaleInterval
# from astropy.nddata import Cutout2D
# from astropy import wcs
import tqdm

import warnings
warnings.filterwarnings("ignore")

from scipy.special import gamma
def bn(n):
    return 1.9992*n - 0.3271
def fn(n):
    b = bn(n)
    return (n*np.exp(b)/b**(2*n))*gamma(2*n)

import init_settings as init


# ----- Step 5. Making thumbnail images & printing results ----- #
def plot_results(galaxy_id, band, outfile, mask_prefix="Mask", edge_pix=5,
                 dir_input="input", dir_output="output1"):
    
    read_success, stable_, fit0_success = 0, 0, 0
#     fault_id = {'read':[], 'unstable':[], 'fit0':[]}

    if (dir_input[-1] == "/"):
        dir_input = dir_input[:-1]
    if (dir_output[-1] == "/"):
        dir_output = dir_output[:-1]

    # Reading log file
    gflog = dir_output+f"/Result_{galaxy_id:05d}"+".log"
    print("\nReading "+gflog.split("/")[-1]+" ...")
    try:
        res1 = np.genfromtxt(gflog, dtype=None, skip_header=7, max_rows=1, encoding='ascii',
                             usecols=(3,4,5,6,7,8,9),
                             names=('x','y','mu','r','n','ba','pa'))
        res2 = np.genfromtxt(gflog, dtype=None, skip_header=8, max_rows=1, encoding='ascii',
                             usecols=(1,2,3,4,5,6,7),
                             names=('xe','ye','mue','re','ne','bae','pae'))
        res3 = np.genfromtxt(gflog, dtype=None, skip_header=12, max_rows=1, encoding='ascii',
                             usecols=(2), names=('chi'))
        read_success += 1
    except:
        pass
    
    if read_success:
        sx, sy = res1['x'].item(0).replace(',',''), res1['y'].item(0).replace(')','')
        smu, sr, sn = str(res1['mu'].item(0)), str(res1['r'].item(0)), str(res1['n'].item(0))
        sba, spa = str(res1['ba'].item(0)), str(res1['pa'].item(0))

        e_sx, e_sy = res2['xe'].item(0).replace(',',''), res2['ye'].item(0).replace(')','')
        e_smu, e_sr, e_sn = str(res2['mue'].item(0)), str(res2['re'].item(0)), str(res2['ne'].item(0))
        e_sba, e_spa = str(res2['bae'].item(0)), str(res2['pae'].item(0))

        schi = str(res3['chi'].item(0))

        if (((sx[0] != '*') & (sy[0] != '*') & (smu[0] != '*') & (sr[0] != '*') & (sn[0] != '*') & \
             (sba[0] != '*') & (spa[0] != '*'))):
            sr = str(f"{float(sr)*init.pixel_scale:.2f}")
            e_sr = str(f"{float(e_sr)*init.pixel_scale:.2f}")
            stable_ += 1
        else:
            sx, sy, smu, sr, sn, sba, spa = "99.9", "99.9", "99.9", "99.9", "99.9", "99.9", "99.9"
            e_sx, e_sy, e_smu, e_sr, e_sn, e_sba, e_spa = "99.9", "99.9", "99.9", "99.9", "99.9", "99.9", "99.9"
    else:
        sx, sy, smu, sr, sn, sba, spa = "99.9", "99.9", "99.9", "99.9", "99.9", "99.9", "99.9"
        e_sx, e_sy, e_smu, e_sr, e_sn, e_sba, e_spa = "99.9", "99.9", "99.9", "99.9", "99.9", "99.9", "99.9"
        schi = "99.9"        

    # Reading images
    try:
        ori = fits.getdata(dir_output+f"/Block_{galaxy_id:05d}.fits", ext=1)
        mod = fits.getdata(dir_output+f"/Block_{galaxy_id:05d}.fits", ext=2)
        res = fits.getdata(dir_output+f"/Block_{galaxy_id:05d}.fits", ext=3)
        msk = fits.getdata(dir_input+"/"+mask_prefix+f"_{galaxy_id:05d}.fits")
        msk2 = msk[edge_pix:-edge_pix+1, edge_pix:-edge_pix+1]
        fit0_success += 1
        go_plot = True
    except:
        ori = np.zeros((10,10))
        mod = np.zeros((10,10))
        res = np.zeros((10,10))
        msk2 = np.zeros((10,10))
        go_plot = False
        pass
    
    # Calculating asymmetry
    seg = fits.getdata(dir_input+f"/Segm_{galaxy_id:05d}.fits")
    xsz, ysz = seg.shape[1], seg.shape[0]
    x, y = np.arange(xsz), np.arange(ysz)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = (xx-init.rth)**2 + (yy-init.rth)**2 - 2.0**2
    seg[z <= 0.0] = 0
    seg_0 = copy.deepcopy(seg)
    seg_0[seg != galaxy_id] = 0
    seg_180 = np.flip(np.flip(seg_0, 1), 0)
    asym = np.sum(np.abs(seg_0 - seg_180)) / (2*np.sum(np.abs(seg_0)))
        
    # Plotting
    fig, axs = plt.subplots(1, 4, figsize=(9,3))
    interval = ZScaleInterval()
    vmin, vmax = interval.get_limits(ori)
    plt_img = [ori, mod, res, msk2]
    for k, p in enumerate(plt_img):
        ax = axs[k]
        if (k == 2):
            ax.imshow(p, cmap='gray_r', origin='lower',
                      vmin=interval.get_limits(res)[0],
                      vmax=interval.get_limits(res)[1])
        else:
            ax.imshow(p, cmap='gray_r', origin='lower', vmin=vmin, vmax=vmax)
        ax.tick_params(axis='both', length=0.0, labelleft=False, labelbottom=False)
        if (k == 0):
            ax.text(0.50, 1.03, f"ID {galaxy_id:05d} ("+band.upper()+")", fontsize=12.0, fontweight='bold',
                    transform=ax.transAxes, ha='center', va='bottom')
        if ((k == 1) & (read_success + stable_ + fit0_success == 3)):
#             (galaxy_id not in fault_id['read']) & \
#             (galaxy_id not in fault_id['fit0']) & \
#             (galaxy_id not in fault)):
            ax.text(0.50, 1.03, f"A={asym:.3f}", fontsize=11.0, fontweight='bold',
                    transform=ax.transAxes, ha='center', va='bottom')
            ax.text(0.04, 0.96, r"$R_{\rm e}=$"+sr+" arcsec", fontsize=10.0,
                    transform=ax.transAxes, ha='left', va='top')
            ax.text(0.04, 0.88, r"$m_{\rm tot}=$"+smu+" mag", fontsize=10.0,
                    transform=ax.transAxes, ha='left', va='top')
            try:
                mu1 = float(smu) + 2.5*np.log10(2.*np.pi*float(sr)**2*fn(float(sn)))
#             float(smu)+((2.5*1.9992*float(sn)-0.3271)/np.log(10.0))*((1.0/float(sr))**(1./float(sn)) - 1.)
            except:
                mu1 = 99.9
            ax.text(0.04, 0.12, r"$\mu_{\rm e}=$"+f"{mu1:.1f} mag/arcsec2", fontsize=10.0,
                    transform=ax.transAxes, ha='left', va='bottom') 
            ax.text(0.04, 0.04, r"$n=$"+sn+" +/- "+e_sn, fontsize=10.0,
                    transform=ax.transAxes, ha='left', va='bottom')
            ax.text(0.04, -0.04, r"$\chi_{\nu}^{2}=$"+f"{float(schi):.4f}", fontsize=10.0,
                    transform=ax.transAxes, ha='left', va='top')   
    plt.tight_layout()
    plt.savefig(dir_output+f"/Plot_{galaxy_id:05d}.png", dpi=300)
#     else:
#         fig, axs = plt.subplots(1, 4, figsize=(9,3))
        
#         plt.tight_layout()
#         plt.savefig(dir_output+f"/Plot_{galaxy_id:05d}.png", dpi=300)
        
    if read_success:
        print(f"ID {galaxy_id:05d}: Successfully read")
    else:
        print(f"ID {galaxy_id:05d}: Not successfully read")
    if stable_:
        print(f"ID {galaxy_id:05d}: Stable solutions")
    else:
        print(f"ID {galaxy_id:05d}: Unstable solutions")
    if fit0_success:
        print(f"ID {galaxy_id:05d}: Successfully fitted\n")
    else:
        print(f"ID {galaxy_id:05d}: Not successfully fitted\n")
        
    f = open(outfile, "a")
    f.write(f"{galaxy_id:05d},"+band+","+sx+","+sy+","+smu+","+sr+","+sn+","+sba+","+spa+","+ \
            f"{asym:.3f}"+","+schi+"\n")
    f.close()
    
    return [galaxy_id, read_success, stable_, fit0_success]


# dir_output = "output2/"

for n, dir_output in enumerate(["output1/", "output2/", "output3/"]):
    f = open(dir_output+"output.csv", "w")
    f.write("id,band,x,y,mu,r_e,n,b/a,pa,asym,chi2nu\n")
    f.close()

    fault_id = {'read':[], 'unstable':[], 'fit0':[]}
    for i, gid in enumerate(np.hstack([init.id_z1, init.id_z2, init.id_z3])):
        if (i < init.n_group_z1):
            j = 0
        elif ((i >= init.n_group_z1) & (i < init.n_group_z1+init.n_group_z2)):
            j = 1
        else:
            j = 2
    #     j = i // n_group
        id_, read_, stab_, fit0_ = plot_results(gid, init.band[j], dir_output+"output.csv",
                                                dir_input="input", dir_output=dir_output)
        if (read_ == 0):
            fault_id['read'].append(id_)
        if (stab_ == 0):
            fault_id['unstable'].append(id_)
        if (fit0_ == 0):
            fault_id['fit0'].append(id_)
    print(fault_id)
# f.close()
