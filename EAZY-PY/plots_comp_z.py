### Imports

# Printing the versions of packages
from importlib_metadata import version
for pkg in ['numpy', 'matplotlib', 'pandas']:
    print(pkg+": ver "+version(pkg))
print("\n")

# importing necessary modules
import numpy as np
import glob, os, copy
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rc, rcParams
rc('axes', linewidth=2)
rc('font', weight='bold')
rcParams['text.latex.preamble'] = r'\usepackage{sfmath} \boldmath'
from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import truncate_colormap as tc
from astropy.io import fits
import pickle
from matplotlib.patches import Rectangle

import warnings
warnings.filterwarnings("ignore")


c = 2.99792e+5    # km/s


# ----- Loading the photometry data ----- #
dir_phot = "../Phot/"

# load data
with open(dir_phot+"phot_data.pickle", 'rb') as fr:
    phot_data = pickle.load(fr)
 
 
# ----- Read the catalog ----- #
dir_output = "EAZY_OUTPUT/"

dz1i, hz1i = fits.getdata(dir_output+"run1i.eazypy.zout.fits", ext=1, header=True)
dz2i, hz2i = fits.getdata(dir_output+"run2i.eazypy.zout.fits", ext=1, header=True)
dz3i, hz3i = fits.getdata(dir_output+"run3i.eazypy.zout.fits", ext=1, header=True)
dz4i, hz4i = fits.getdata(dir_output+"run4i.eazypy.zout.fits", ext=1, header=True)

dz1p, hz1p = fits.getdata(dir_output+"run1p.eazypy.zout.fits", ext=1, header=True)
dz1z, hz1z = fits.getdata(dir_output+"run1z.eazypy.zout.fits", ext=1, header=True)
dz5z, hz5z = fits.getdata(dir_output+"run5z.eazypy.zout.fits", ext=1, header=True)


# ----- Redshift comparison ----- #
def plot_comp(z_spec, z_phot, z_phot_chi2, out, ids,
              label_x='', label_y='', title='',
              z_clu=0.30, dv_mem=3000., xmin=0.03, xmax=30.0, slope=0.10,
              check_highz=True, colorbar=True):
    z_cnd = ((z_phot > 0.) & (z_spec > 0.) & (z_phot_chi2 > 0.))
    z_mem = (c*np.abs(z_spec-z_clu)/(1+z_clu) < dv_mem)
    print(f"Objects : {np.sum(z_cnd):d}")
    print(f"Members : {np.sum(z_cnd & z_mem):d}")
    
    dz = np.abs(z_spec-z_phot)/(1+z_spec)
    sigma = 1.48*np.median(np.abs(dz[z_cnd]-np.median(dz[z_cnd]))/(1+z_spec[z_cnd]))
    if slope is None:
        slope = 5.0*sigma
        print(f"Slope: {slope:.3f}")
    
    outlier = (z_cnd & (dz >= slope))
    print(f"Outliers: {np.sum(outlier):d}")
    print("\n")

    cmap = cm.get_cmap("jet")
    c_lo, c_hi = -1.5, 1.5
    norm = mpl.colors.Normalize(vmin=c_lo, vmax=c_hi)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    
    if colorbar:
        # c_idxs = []
        X, Y = z_spec[z_cnd], z_phot[z_cnd]
        Color = phot_data['f200w']['mag_auto'].values[ids[z_cnd]-1] - \
                phot_data['f356w']['mag_auto'].values[ids[z_cnd]-1]
        for i in range(np.sum(z_cnd)):
            # print(z_spec[z_cnd][i])
            c_idx = (Color[i]-c_lo) / (c_hi-c_lo)
            ax.plot(X[i], Y[i], 'o', ms=3.5, mew=0.5, color=cmap(c_idx), mec='k', alpha=0.7)
            # c_idxs.append(c_idx)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.08)
        cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cb.set_label("F200W-F356W", size=12.0, labelpad=10.0, fontweight='bold')
        cb.ax.tick_params(direction='in', width=1.5, labelsize=11.0)

    else:
        ax.plot(z_spec[z_cnd & ~z_mem], z_phot[z_cnd & ~z_mem], 'o', ms=3.0, mew=0.5,
                color='tab:blue', alpha=0.6)
        ax.plot(z_spec[z_cnd & z_mem],  z_phot[z_cnd & z_mem],  'o', ms=3.0, mew=0.5,
                color='tab:red', alpha=0.6)
        sym_1, = ax.plot(1.0e-10, 1.0e-10, 'o', ms=4.0, mew=0.8,
                         color='tab:blue', alpha=0.7, label="Spec-z sample")
        sym_2, = ax.plot(1.0e-10, 1.0e-10, 'o', ms=4.0, mew=0.8,
                         color='tab:red', alpha=0.7, label="Cluster member")
        ax.legend(handles=[sym_1, sym_2], fontsize=7.5, loc=(0.035, 0.035),
                  handlelength=0, frameon=True, borderpad=0.8, handletextpad=0.8,
                  framealpha=0.6, edgecolor='gray')
    ax.plot([xmin, xmax], [xmin, xmax], '-', lw=1.5, color='gray', alpha=0.75)
    # ax.plot([(xmin+slope)/(1.0-slope), xmax], [xmin, (1.0-slope)*xmax-slope],
    #       '--', lw=1.2, color='gray', alpha=0.7)
    # ax.plot([xmin, xmax], [(1.0+slope)*xmin+slope, (1.0+slope)*xmax+slope],
    #       '--', lw=1.2, color='gray', alpha=0.7)
    # ax.plot([0.1, 10.], [(1.0+slope)*0.1+slope, (1.0+slope)*10.+slope],
    #         '--', lw=1.2, color='k', alpha=0.7)
    xx = np.logspace(np.log10(xmin), np.log10(xmax), 1000)
    ax.plot(xx, (1.0-slope)*xx-slope, '--', lw=1.2, color='gray', alpha=0.7)
    ax.plot(xx, (1.0+slope)*xx+slope, '--', lw=1.2, color='gray', alpha=0.7)
    # ax.set_xticks([0., 2., 4., 6., 8.])
    # ax.set_yticks([0., 2., 4., 6., 8.])
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([xmin, xmax])
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(label_x, fontsize=12.0)
    ax.set_ylabel(label_y, fontsize=12.0)
    ax.tick_params(axis='both', labelsize=12.0)
    ax.tick_params(width=1.5, length=7.5)
    ax.tick_params(width=1.5, length=4.0,which='minor')
    # ax.text(0.05, 0.95, title,
    #         fontsize=12.0, fontweight="bold", color="black",
    #         ha="left", va="top", transform=ax.transAxes)
    ax.text(0.05, 0.93, r"$N$"+f" = {np.sum(z_cnd):d} ({np.sum(outlier):d},"+ \
            f" {100.*np.sum(outlier)/np.sum(z_cnd):.1f}%)",
            fontsize=9.0, color="black",
            ha="left", va="top", transform=ax.transAxes)
    ax.text(0.05, 0.88, r"$\sigma_{\rm NMAD}$"+ \
            f" = {sigma:.3f}",
            fontsize=9.0, color="black",
            ha="left", va="top", transform=ax.transAxes)

    if check_highz:

        ### zs > 1.5
        hzsp_cnd = (z_spec > 1.5)
        ax.axvline(1.5, 0.0, 1.0, ls='-.', lw=1.2, color='gray', alpha=0.5)
        ax.text(0.95, 0.11, r"$N(z_{\rm s}>1.5)$"+ \
                f" = {np.sum(z_cnd & hzsp_cnd):d} "+ \
                f"({np.sum(outlier & hzsp_cnd):d},"+ \
                f" {100.*np.sum(outlier & hzsp_cnd)/np.sum(z_cnd & hzsp_cnd):.1f}%)",
                fontsize=9.0, color="dimgray",
                ha="right", va="bottom", transform=ax.transAxes                )
        sigma_highz = 1.48*np.median(np.abs(dz[z_cnd & hzsp_cnd]- \
                      np.median(dz[z_cnd & hzsp_cnd]))/(1+z_spec[z_cnd & hzsp_cnd]))
        ax.text(0.95, 0.05, r"$\sigma_{\rm NMAD}~(z_{\rm s}>1.5)$"+ \
                f" = {sigma_highz:.3f}",
                fontsize=9.0, color="dimgray",
                ha="right", va="bottom", transform=ax.transAxes)

        ### zp > 1.5
        hzph_cnd = (z_phot > 1.5)
        ax.axhline(1.5, 0.0, 1.0, ls='-.', lw=1.2, color='gray', alpha=0.5)
        ax.text(0.95, 0.23, r"$N(z_{\rm p}>1.5)$"+ \
                f" = {np.sum(z_cnd & hzph_cnd):d} "+ \
                f"({np.sum(outlier & hzph_cnd):d},"+ \
                f" {100.*np.sum(outlier & hzph_cnd)/np.sum(z_cnd & hzph_cnd):.1f}%)",
                fontsize=9.0, color="dimgray",
                ha="right", va="bottom", transform=ax.transAxes                )
        sigma_highz = 1.48*np.median(np.abs(dz[z_cnd & hzph_cnd]- \
                      np.median(dz[z_cnd & hzph_cnd]))/(1+z_spec[z_cnd & hzph_cnd]))
        ax.text(0.95, 0.17, r"$\sigma_{\rm NMAD}~(z_{\rm p}>1.5)$"+ \
                f" = {sigma_highz:.3f}",
                fontsize=9.0, color="dimgray",
                ha="right", va="bottom", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    # plt.show(block=False)
    plt.close()

    return ids[outlier]


# Diagrams
dzs = [dz1i, dz2i, dz3i, dz4i, dz1p, dz1z, dz5z]
run_names = ["run1i", "run2i", "run3i", "run4i", "run1p", "run1z", "run5z"]

### z_spec vs. z_phot (maximum likelihood) from eazy-py
ids_out = []
for i in range(len(run_names)):
    print("--- ", run_names[i], " ---")
    id_out = plot_comp(dzs[i]['z_spec'], dzs[i]['z_phot'], dzs[i]['z_phot_chi2'],
                       "Fig1-comp_z_ml_eazypy_"+run_names[i]+".png", dzs[i]['id'],
                       label_x=r"$z_{\rm spec}$", label_y=r"$z_{\rm phot}$ ($z_{\rm ml}$)",
                       title=r"Phot-z from EAZY",
                       z_clu=0.308, dv_mem=3000., xmin=0.03, xmax=30.0, slope=0.15,
                       check_highz=True, colorbar=True)
    np.savetxt("ID_out-"+run_names[i]+".txt", id_out.T, fmt='%d')
    if (run_names[i][-1] == 'p'):
        zph_hi = (np.in1d(dzs[i]['id'], id_out) & \
                  (dzs[i]['z_phot'] >= 1.5) & (dzs[i]['z_spec'] <= 1.5))
        zph_lo = (np.in1d(dzs[i]['id'], id_out) & \
                  (dzs[i]['z_phot'] <= 1.5) & (dzs[i]['z_spec'] >= 1.5))
        np.savetxt("ID_zph-"+run_names[i]+".txt", dzs[i]['id'][zph_hi].T, fmt='%d')
        np.savetxt("ID_zpl-"+run_names[i]+".txt", dzs[i]['id'][zph_lo].T, fmt='%d')
    ids_out.append(id_out)

### z_spec vs. z_phot (chi2-minimization) from eazy-py
ids_out2 = []
for i in range(len(run_names)):
    print("--- ", run_names[i], " ---")
    id_out2 = plot_comp(dzs[i]['z_spec'], dzs[i]['z_raw_chi2'], dzs[i]['raw_chi2'],
                        "Fig1-comp_z_chi2_eazypy_"+run_names[i]+".png", dzs[i]['id'],
                        label_x=r"$z_{\rm spec}$", label_y=r"$z_{\rm phot}$ ($z_{\chi^{2}_{\rm min}}$)",
                        title=r"Phot-z from EAZY",
                        z_clu=0.308, dv_mem=3000., xmin=0.03, xmax=30.0, slope=0.15,
                        check_highz=True, colorbar=True)
    np.savetxt("ID_out2-"+run_names[i]+".txt", id_out.T, fmt='%d')
    if (run_names[i][-1] == 'p'):
        zph_hi = (np.in1d(dzs[i]['id'], id_out2) & \
                  (dzs[i]['z_raw_chi2'] >= 1.5) & (dzs[i]['z_spec'] <= 1.5))
        zph_lo = (np.in1d(dzs[i]['id'], id_out2) & \
                  (dzs[i]['z_raw_chi2'] <= 1.5) & (dzs[i]['z_spec'] >= 1.5))
        np.savetxt("ID_zph2-"+run_names[i]+".txt", dzs[i]['id'][zph_hi].T, fmt='%d')
        np.savetxt("ID_zpl2-"+run_names[i]+".txt", dzs[i]['id'][zph_lo].T, fmt='%d')
    ids_out2.append(id_out2)



'''
# ----- Write the region files ----- #
run_idx = 3
id_check1 = dzs[run_idx]['id'][np.in1d(dzs[run_idx]['id'], id_out) & \
                               (dz4i['z_spec'] < 1.5) & (dz4i['z_phot'] > 1.5)]

with open("check1_"+run_names[run_idx]+".reg", "w") as f:
    f.write('global color=blue font="helvetica 10 normal" ')
    f.write("select=1 edit=1 move=1 delete=1 include=1 fixed=0 source width=2\n")
    f.write("fk5\n")
    for j in range(len(id_check1)):
        reg_RA  = phot_data['nir_detect']['ra'].values[id_check1[j]-1]
        reg_DEC = phot_data['nir_detect']['dec'].values[id_check1[j]-1]
        reg_rad = 2.0
        f.write(f"circle({reg_RA:.6f}, {reg_DEC:.6f}, {reg_rad:.1f}")
        f.write('")\n')  # text={'+f"{z_spec[idx_spec][i]:.4f}"+'}\n')



# ----- Color-color diagrams ----- #
def plot_ccd2(z_spec, z_phot, out, ids, title='', cmd=False, size=False,
              col1=['f200w', 'f356w'], col2=['f150w','f277w'], mag2=None,
              plot_box=False, box_x0=[], box_y0=[], box_width=[], box_height=[]):
              # z_clu=0.30, dv_mem=3000., xmin=0.03, xmax=30.0, slope=0.10):
    z_cnd = ((z_phot > 0) & (z_spec > 0))
    # id_all      = ids[z_cnd]-1
    id_all      = ids-1

    c1 = col1#['f200w', 'f356w']
    color1 = phot_data[c1[0]]['mag_auto'].values[id_all] - \
             phot_data[c1[1]]['mag_auto'].values[id_all]
    if cmd:
        m2 = mag2
        magnitude2 = phot_data[m2]['mag_auto'].values[id_all]
        X, Y = color1, magnitude2
    elif size:
        m2 = mag2
        size2 = phot_data[m2]['flxrad'].values[id_all]
        X, Y = color1, size2        
    else:
        c2 = col2
        color2 = phot_data[c2[0]]['mag_auto'].values[id_all] - \
                 phot_data[c2[1]]['mag_auto'].values[id_all]
        X, Y = color1, color2

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    cmap = cm.get_cmap("jet")
    f_lo, f_hi, nn = 0.1, 0.9, 256
    tcmp = tc.truncate_colormap(cmap, f_lo, f_hi, nn)

    c_lo, c_hi = np.log10(0.3), np.log10(3.0)
    norm = mpl.colors.Normalize(vmin=c_lo, vmax=c_hi)

    c_idxs = []
    for i in range(np.sum(z_cnd)):
        # print(z_spec[z_cnd][i])
        c_idx = f_lo + (f_hi-f_lo)*(np.log10(z_spec[z_cnd][i])-c_lo) / (c_hi-c_lo)
        ax.plot(X[z_cnd][i], Y[z_cnd][i], 'o', ms=4.0, color=tcmp(c_idx), mew=0.5, mec='k', alpha=0.6)
        c_idxs.append(c_idx)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=tcmp), cax=cax)
    cb.set_label(r"log $z_{\rm spec}$", size=12.0, labelpad=10.0, fontweight='bold')
    cb.ax.tick_params(direction='in', width=1.5, labelsize=11.0)

    ax.text(0.05, 0.95, title,
            fontsize=12.0, fontweight="bold", color="black",
            ha="left", va="top", transform=ax.transAxes)
    ax.set_xlim([-1.9, 1.9])
    if cmd:
        ax.set_ylim([30., 15.])
    elif size:
        ax.set_ylim([0.0, 10.0])
    else:
        ax.set_ylim([-1.4, 1.4])
    ax.set_xlabel(c1[0].upper()+"-"+c1[1].upper(), fontweight='bold')
    if cmd:
        ax.set_ylabel(m2.upper(), fontweight='bold')
    elif size:
        ax.set_ylabel("Size", fontweight='bold')
    else:
        ax.set_ylabel(c2[0].upper()+"-"+c2[1].upper(), fontweight='bold')
    # plt.legend(handles=syms, fontsize=8.0, loc=(0.03, 0.03),
    #            handlelength=0, frameon=True, borderpad=0.8, handletextpad=0.8,
    #            framealpha=0.6, edgecolor='gray')
    ax.tick_params(width=1.5)

    if plot_box:
        id_box = []
        for i in range(len(box_x0)):
            box = Rectangle((box_x0[i], box_y0[i]),
                            width=box_width[i], height=box_height[i],
                            ls='-', lw=1.0, edgecolor='gray', fill=False, alpha=0.6)
            ax.add_artist(box)
            box_cnd = ((X >= box_x0[i]) & (X <= box_x0[i]+box_width[i]) & \
                       (Y >= box_y0[i]) & (Y <= box_y0[i]+box_height[i]))
            id_box.append(ids[z_cnd][box_cnd])
        return_array = [X[z_cnd], Y[z_cnd], id_box, c_idxs]
    else:
        return_array = [X[z_cnd], Y[z_cnd], c_idxs]

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    
    return return_array


colset = ['f200w', 'f356w']
magset = ['f277w']
fig_modes = ["cmd"]

run_mode = "run1"
dz = dzs[int(run_mode[-1])-1]    # Run 2
delta_z = np.abs(dz['z_spec']-dz['z_phot'])/(1.+dz['z_spec'])
plot_cnds = [np.ones_like(delta_z, dtype=bool), (delta_z <= 0.15), (delta_z > 0.15)]


cmd, plot_box = True, False
title = "Color-magnitude Diagram"

cidxs_arr = []
cols_arr = []
mags_arr = []
for k in range(len(plot_cnds)):
    if (k == 0):
        suffix = "all"
    if (k == 1):
        suffix = "mat"
    if (k == 2):
        suffix = "out"
    cols, mags, cidxs = plot_ccd2(dz['z_spec'][plot_cnds[k]], dz['z_phot'][plot_cnds[k]],
                                  "Fig3-"+fig_modes[0]+"_"+run_mode+"_"+suffix+".png",
                                  dz['id'][plot_cnds[k]], cmd=cmd,
                                  col1=colset, col2=colset, mag2=magset[0],
                                  title=title+" ("+suffix[0].upper()+suffix[1:]+")",
                                  plot_box=plot_box)#, **box_par)
    cidxs_arr.append(cidxs)
    cols_arr.append(cols)
    mags_arr.append(mags)
'''


# col1, col2 = plot_ccd2(dz['z_spec'][plot_cnds[0]], dz['z_phot'][plot_cnds[0]],
                       # "Fig3-"+fig_modes[0]+"_"+run_mode+"_"+f"set1"+"_all.png",
                       # dz['id'][plot_cnds[0]], cmd=False,
                       # col1=colset1[0], col2=colset2[0], mag2=magset0[0],
                       # title="Color-color Diagram (All)",
                       # plot_box=False)

# col3, mag3 = plot_ccd2(dz['z_spec'], dz['z_phot'],
                       # "Fig3-"+fig_modes[1]+"_"+run_mode+"_"+f"set2"+"_ALL.png",
                       # dz['id'], cmd=True,
                       # col1=colset1[1], col2=colset2[1], mag2=magset0[1],
                       # title="Color-magnitude Diagram (All)",
                       # plot_box=False)

'''
mat_cnd = (delta_z <= 0.15)
out_cnd = (delta_z > 0.15)

out_zcl = (np.in1d(dz['id'], id_box[0]) & out_cnd)
out_zhi = (np.in1d(dz['id'], id_box[1]) & out_cnd)


def scaler(X, Xmin=None, Xmax=None, min=0.0, max=1.0):
    X_std = np.minimum(1.0, np.maximum(0.0, (X - Xmin) / (Xmax-Xmin)))
    X_scl = min + (max-min)*X_std
    return X_scl

Xs = [col3[mat_cnd], mag3[mat_cnd]]
Xmins = [-1.2, 18.0]
Xmaxs = [ 1.2, 30.0]
Xs_scl = []
for i, X in enumerate(Xs):
    Xs_scl.append(scaler(X, Xmin=Xmins[i], Xmax=Xmaxs[i], min=0.0, max=1.0))


def pz(z, col, mag, z0, k=9):
    col_scl = scaler(col, Xmin=-1.2, Xmax=1.2,  min=0.0, max=1.0)
    mag_scl = scaler(mag, Xmin=18.0, Xmax=30.0, min=0.0, max=1.0)
    dist2 = np.sqrt((col_scl-Xs_scl[0])**2. + \
                    (mag_scl-Xs_scl[1])**2.)
    dist2 *= 1.0
    idx = np.argsort(dist2)
    if (dist2[idx[0]] <= 0.):
        ii = 1 + np.arange(k)
    else:
        ii = np.arange(k)
    gauss = 0.
    for i in ii:
        sigma = dist2[idx[i]]
        sigma = np.maximum(sigma, 0.15)
        gauss += (1./(np.sqrt(2.*np.pi)*sigma)) * \
                 np.exp(-0.5*((np.log(z)-np.log(z0[idx[i]]))/sigma)**2.)
        #(1./(np.sqrt(2.*np.pi)*dist2[idx[i]])) * \
        # gauss += np.exp((z-z0[idx[i]])/dist2[idx[i]])
    return gauss


# In [15]: dz['id'][out_zcl]
# Out[15]: array([31676])

# In [16]: dz['id'][out_zhi]
# Out[16]: array([17262, 28188, 28391, 29380, 29603, 30606, 31762])


zgrid = fits.getdata("output.eazypy.data.fits", ext=2)
zchi2 = fits.getdata("output.eazypy.data.fits", ext=3)


plt.close('all')

z_peak2 = copy.deepcopy(dz['z_phot'])

for test_id in np.hstack([id_box[0], id_box[1]]):

    # test_id = 29380
    # pz_ = np.genfromtxt(dir_output+prefix+f"_{test_id:d}.pz", dtype=None, encoding='ascii',
                        # usecols=(0,1), names=('z','chi2'))
    zz = zgrid

    idz = np.argwhere(dz['id'] == test_id)[0][0]
    # idz = dz.loc[dz['id'] == test_id].index[0]
    test_col, test_mag = col3[idz], mag3[idz]
    # test_col1, test_col2, test_mag2 = color1[idz], color2[idz], magnitude2[idz]

    pz_sum = np.trapz(pz(zz, test_col, test_mag, dz['z_spec'][mat_cnd], k=5), x=zz)
    pz1_ = pz(zz, test_col, test_mag, dz['z_spec'][mat_cnd], k=5) / pz_sum

    # fig, ax = plt.subplots()
    # ax.plot(zz, pz1_)
    # ax.set_xscale('log')
    # plt.tight_layout()
    # plt.show(block=False)

    pz_chi2 = zchi2[idz, :] / zchi2[idz, :].min()
    pz_sum = np.trapz(np.exp(-0.5*pz_chi2**2.), x=zz)
    pz2_ = np.exp(-0.5*pz_chi2**2.) / pz_sum

    # fig, ax = plt.subplots()
    # ax.plot(zz, pz2_)
    # ax.set_xscale('log')
    # plt.tight_layout()
    # plt.show(block=False)

    # fig, ax = plt.subplots()
    # ax.plot(zz, pz1_*pz2_ / np.trapz(pz1_*pz2_, x=zz))
    # ax.set_xscale('log')
    # plt.tight_layout()
    # plt.show(block=False)

    zidx = np.argmax(pz1_*pz2_ / np.trapz(pz1_*pz2_, x=zz))
    print(dz['z_spec'][idz], dz['z_phot'][idz], zz[zidx], np.abs(dz['z_spec'][idz]-zz[zidx]) / (1.+dz['z_spec'][idz]))
    z_peak2[idz] = zz[zidx]


plot_comp(dz['z_spec'], z_peak2, "Fig1-comp_z_total_eazypy_run2_test.png",
          label_x=r"$z_{\rm spec}$", label_y=r"$z_{\rm phot}$", title="Phot-z from EAZY (HST+JWST)",
          z_clu=0.308, dv_mem=3000., xmin=0.03, xmax=30.0, slope=0.15,
          check_highz=True)
'''


