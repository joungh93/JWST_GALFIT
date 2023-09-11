# Imports
import eazy

# Module versions
import importlib
import sys
import time
print(time.ctime() + '\n')

print(sys.version + '\n')

for module in ['numpy', 'scipy', 'matplotlib','astropy','eazy']:#, 'prospect']:
    #print(module)
    mod = importlib.import_module(module)
    print('{0:>20} : {1}'.format(module, mod.__version__))

import glob, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
rc('axes', linewidth=2)
rc('font', weight='bold')
rcParams['text.latex.preamble'] = r'\usepackage{sfmath} \boldmath'
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.visualization import ZScaleInterval
interval = ZScaleInterval()
from matplotlib.patches import Ellipse
import pickle
import warnings
from astropy.io import fits
import tqdm


c = 2.99792e+5    # km/s


def plot_seds(dir_eazyout, run_name, objids=None, save_data=False):
    
    if not (dir_eazyout[-1] == "/"):
        dir_eazyout += "/"

    dir_figs = dir_eazyout+run_name+".SED_EAZY/"
    if not os.path.exists(dir_figs):
        os.system("mkdir "+dir_figs)

    dz, hz = fits.getdata(dir_eazyout+run_name+".eazypy.zout.fits", ext=1, header=True)

    with open(dir_eazyout+run_name+".eazypy.zphot.pickle", "rb") as fr:
        pred = pickle.load(fr)

    if objids is None:
        objids = dz['id']
    n_obj = len(objids)

    for i in tqdm.trange(n_obj):
        objid = objids[i]
        fig, data = pred.show_fit(objid, id_is_idx=False, show_fnu=1,
                                  xlim=[0.3, 9], show_components=True)
        plt.savefig(dir_figs+f"ID-{objid:05d}.png", dpi=300)
        plt.close()

        if save_data:
            with open(dir_figs+f"ID-{objid:05d}.data", "wb") as fw:
                pickle.dump(data, fw)                    



dir_output = "EAZY_OUTPUT/"
run_name = ["run1z", "run1p"]

# idc = np.array([11043, 11860, 14564, 12869,
                 # 7339,  7525,  8613,  9074, 10624, 10901, 11201, 12298, 13565,
                # 13615, 14151, 14165, 14497, 10474])

plot_seds(dir_output, run_name[0], objids=None, save_data=True)
plot_seds(dir_output, run_name[1], objids=None, save_data=True)


'''
##########

# id_flt_hst  = [208, 233, 236, 239, 203, 204, 205]    # from 'FILTER.RES.latest.info' file
id_flt_hst  = [233, 236, 238, 239, 240, 202, 203, 205]    # from 'FILTER.RES.latest.info' file
id_flt_jwst = [364, 365, 366, 375, 376, 377]    # from 'FILTER.RES.latest.info' file
n_hst, n_jwst = len(id_flt_hst), len(id_flt_jwst)

##########


run_id   = "run1p"
dz, hz   = fits.getdata(dir_output+run_id+".eazypy.zout.fits", ext=1, header=True)
hdul     = fits.open(dir_output+run_id+".eazypy.data.fits")
zgrid    = hdul[2].data
chi2     = hdul[3].data

files_id = sorted(glob.glob("ID_zp*-"+run_id+".txt"))
id_zph   = np.loadtxt(files_id[0], dtype='int')
if (id_zph.size == 1):
    id_zph = np.array([id_zph])
id_zpl   = np.loadtxt(files_id[1], dtype='int')
if (id_zpl.size == 1):
    id_zpl = np.array([id_zpl])

def plot_sed2(dir_eazyout, run_name, objid, zgrid, chi2,
              z_phot=0.1, z_spec=-1.0, mstar=1.0e+9,
              n_inst=2, n_hst=None, n_jwst=None, color_inst=[], label_inst=[],
              cut_img=None, pixel_scale=None,
              f200w_mag=None, r_h=None, f200w_f277w=None, f277w_f356w=None,
              r_kron=None, axis_ratio=None, theta=None):

    # objid, prefix, out, z_phot=0.1, z_spec=-1.0, peak_prob=1.0,
    #               dir_output="FAST_INPUT", dir_fig="EAZY_Figure",
    #               n_inst=1, wav_eff=[], color_inst=[], label_inst=[],
    #               cut_img=None, pixel_scale=None,
    #               f200w_mag=None, r_h=None, color1=None, color2=None,
    #               r_kron=None, axis_ratio=None, theta=None,
    #               cut_img2=None):
    
    if not (dir_eazyout[-1] == "/"):
        dir_eazyout += "/"

    dir_figs = dir_eazyout+run_name+".SED_EAZY/"
    if not os.path.exists(dir_figs):
        os.system("mkdir "+dir_figs)

    with open(dir_figs+f"ID-{objid:05d}.data", "rb") as fr:
        data = pickle.load(fr)

    idx_val   = np.flatnonzero(data['valid'])
    lambda_c  = data['pivot'] * 1.0e-4    # Angstrom to micro-meter
    flx_cat   = data['fobs'] 
    e_flx_cat = data['efobs']
    wav_fit   = data['templz'] * 1.0e-4    # Angstrom to micro-meter
    flx_fit   = data['templf']

    # Establish bounds
    xmin, xmax = np.min(lambda_c)*0.3, np.max(lambda_c)/0.3
    idx_xmin, idx_xmax = np.abs(wav_fit-0.4).argmin()-1, np.abs(wav_fit-10.).argmin()+2
    nu_fit = 1.0e+9 * c / wav_fit    # to Hz
    fphot  = 1.0e+9 * c / lambda_c    # to Hz
    specE  = nu_fit * flx_fit * 1.0e-29
    # print(specE)
    ymin = np.maximum(3.0e-19, specE[idx_xmin:idx_xmax][specE[idx_xmin:idx_xmax] > 0.].min()*0.2)
    ymax = specE[idx_xmin:idx_xmax][specE[idx_xmin:idx_xmax] > 0.].max()/0.2

    if (n_inst > 1):
        n_tot    = n_hst + n_jwst
        idx_hst  = idx_val[idx_val <  n_hst]  #np.arange(n_hst)
        idx_jwst = idx_val[idx_val >= n_hst]  #np.arange(n_tot)[~np.in1d(np.arange(n_tot), np.arange(n_hst))]
        idx_inst = [idx_hst, idx_jwst]

    # --------------- #
    fig = plt.figure(1, figsize=(10,8))
    gs = GridSpec(2, 1, left=0.15, bottom=0.10, right=0.75, top=0.95,
                  height_ratios=[6,3], hspace=0.30)
    ax1 = fig.add_subplot(gs[0,0])
    ax = ax1
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=15.0, pad=8.0)
    ax.set_xticks([1.0e-1, 5.0e-1, 1.0e+0, 5.0e+0, 1.0e+1])
    ax.set_xticklabels(["0.1", "0.5", "1", "5", "10"])
    ax.set_xlabel(r"$\lambda_{\rm obs}~{\rm [\mu m]}$", fontsize=15.0, labelpad=7.0)
    ax.set_ylabel(r"$\nu F_{\nu}~{\rm [erg~s^{-1}~cm^{-2}]}$", fontsize=15.0, labelpad=7.0)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.tick_params(width=1.5, length=9.0)
    ax.tick_params(width=1.5, length=5.0, which="minor")
    for axis in ["top","bottom","left","right"]:
        ax.spines[axis].set_linewidth(1.5)
    # --------------- #

    # Plotting model + data
    ax.plot(wav_fit, nu_fit*flx_fit*1.0e-29,
            label="Model spectrum",
            lw=1.5, color="navy", alpha=0.5)
    if (n_inst == 1):
        ax.errorbar(lambda_c, fphot*flx_cat*1.0e-29,
                    yerr=fphot*e_flx_cat*1.0e-29,
                    label=label_inst[0],
                    marker="o", markersize=8, alpha=0.9, ls="", lw=2,
                    ecolor=color_inst[0], markerfacecolor="none", markeredgecolor=color_inst[0], 
                    markeredgewidth=2)
    else:
        for i in range(n_inst):
            ax.errorbar(lambda_c[idx_inst[i]], fphot[idx_inst[i]]*flx_cat[idx_inst[i]]*1.0e-29,
                        yerr=fphot[idx_inst[i]]*e_flx_cat[idx_inst[i]]*1.0e-29,
                        label=label_inst[i],
                        marker="o", markersize=8, alpha=0.9, ls="", lw=2,
                        ecolor=color_inst[i], markerfacecolor="none", markeredgecolor=color_inst[i], 
                        markeredgewidth=2)        

    # Figure texts
    ax.text(1.05, 0.95, f"ID-{objid:05d}", fontsize=18.0, fontweight="bold", color="black",
            ha="left", va="top", transform=ax.transAxes)
    ax.text(1.03, 0.78, r"$z_{\rm spec}=$"+f"{z_spec:.4f}",
            fontsize=14.0, color='red', ha="left", va="top", transform=ax.transAxes)
    ax.text(1.03, 0.70, r"$z_{\rm phot}=$"+f"{z_phot:.4f}",
            fontsize=14.0, color='blue', ha="left", va="top", transform=ax.transAxes)
    ax.text(1.03, 0.62, r"$\chi^{2}=$"+f"{data['chi2']:.4f}",
            fontsize=14.0, color='blue', ha="left", va="top", transform=ax.transAxes)
    ax.text(1.03, 0.54, r"${\rm log}~(M_{{\ast}}/M_{\odot})=$"+f"{np.log10(mstar):.2f}",
            fontsize=14.0, color='blue', ha="left", va="top", transform=ax.transAxes)
    ax.legend(loc="upper left", fontsize=12)

    # --------------- #
    dchi2 = chi2.max() - chi2.min()
    chi2min, chi2max = chi2.min()-0.2*dchi2, chi2.max()+0.2*dchi2

    ax2 = fig.add_subplot(gs[1,0])
    ax = ax2
    ax.set_xlabel(r"$z$", fontsize=15.0, labelpad=7.0)
    ax.set_ylabel(r"$\chi^{2}$", fontsize=15.0, labelpad=7.0)
    # ax.set_xlim([xmin, xmax])
    ax.set_ylim([chi2min, chi2max])
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.tick_params(axis="both", labelsize=15.0, pad=8.0)
    ax.tick_params(width=1.5, length=9.0)
    ax.tick_params(width=1.5, length=5.0, which="minor")
    for axis in ["top","bottom","left","right"]:
        ax.spines[axis].set_linewidth(1.5)
    # --------------- #
    ax.plot(zgrid, chi2, lw=2.0, color="dimgray", alpha=0.7)
    ax.axvline(z_phot, 0., 1.,
               # dz.loc[dz['id'] == objid]['z_peak'].values[0], 0., 1.,
               ls='--', lw=1.5, color='blue', alpha=0.7, label=r"$z_{\rm phot}$")
    ax.axvline(z_spec, 0., 1.,
               # dz.loc[dz['id'] == objid]['z_spec'].values[0], 0., 1.,
               ls='--', lw=1.5, color='red', alpha=0.7, label=r"$z_{\rm spec}$")
    ax.legend(loc="upper left", fontsize=12)

    if (cut_img is not None):
        axins = inset_axes(ax, width="100%", height="100%",
                           bbox_to_anchor=(1.00, 0.70, 0.35, 0.80),
                           bbox_transform=ax.transAxes, borderpad=0)
        axins.tick_params(left=False, right=False, labelleft=False, labelright=False,
                          top=False, bottom=False, labeltop=False, labelbottom=False)
        for axis in ['top','bottom','left','right']:
            axins.spines[axis].set_linewidth(0.8)

        vmin, vmax = interval.get_limits(cut_img)
        axins.imshow(cut_img, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)
        # fib = Circle((round(rth/pixel_scale), round(rth/pixel_scale)), radius=0.75/pixel_scale,
        #              linewidth=1.5, edgecolor='magenta', fill=False, alpha=0.9)
        # axins.add_artist(fib)

        e = Ellipse((cut_img.shape[1] // 2, cut_img.shape[0] // 2),
                    width=2*r_kron, height=2*r_kron*axis_ratio, angle=theta,
                    fill=False, color='magenta', linestyle='-', linewidth=2.0, zorder=10, alpha=0.8) 
        axins.add_patch(e)

    ax.text(1.02, 0.48, f"F200W={f200w_mag:.2f}",
            fontsize=12.5, color='dimgray', ha="left", va="bottom", transform=ax.transAxes)
    ax.text(1.02, 0.32, r"$r_{h}=$"+f"{r_h*pixel_scale:.2f}"+'"',
            fontsize=12.5, color='dimgray', ha="left", va="bottom", transform=ax.transAxes)
    ax.text(1.02, 0.16, f"F200W-F277W={f200w_f277w:.2f}",
            fontsize=12.5, color='dimgray', ha="left", va="bottom", transform=ax.transAxes)
    ax.text(1.02, 0.00, f"F277W-F356W={f277w_f356w:.2f}",
            fontsize=12.5, color='dimgray', ha="left", va="bottom", transform=ax.transAxes)

    # if (cut_img2 is not None):
    #     axins = inset_axes(ax, width="100%", height="100%",
    #                        bbox_to_anchor=(1.00, 1.60, 0.35, 0.80),
    #                        bbox_transform=ax.transAxes, borderpad=0)
    #     axins.tick_params(left=False, right=False, labelleft=False, labelright=False,
    #                       top=False, bottom=False, labeltop=False, labelbottom=False)
    #     for axis in ['top','bottom','left','right']:
    #         axins.spines[axis].set_linewidth(0.8)

    #     try:
    #         vmin, vmax = interval.get_limits(cut_img2)
    #         axins.imshow(cut_img2, origin='lower', cmap='gray_r', vmin=vmin, vmax=vmax)
    #         e = Ellipse((cut_img.shape[1] // 2, cut_img.shape[0] // 2),
    #                     width=2*r_kron, height=2*r_kron*axis_ratio, angle=theta,
    #                     fill=False, color='magenta', linestyle='-', linewidth=2.0, zorder=10, alpha=0.8) 
    #         axins.add_patch(e)
    #     except:
    #         axins.imshow(np.zeros_like(cut_img), origin='lower', cmap='gray_r')


    plt.savefig(dir_figs+f"SED-{objid:05d}.png", dpi=300)
    # plt.savefig("./fig_FAST.png", dpi=300)
    plt.close() 



# Image data
dir_img = "../Reproject/"
totimg = fits.getdata(dir_img+"nir_detect.fits", header=False)
rth = 100

# Photometric data
dir_phot = "../Phot/"
with open(dir_phot+"phot_data.pickle", 'rb') as fr:
    phot_data = pickle.load(fr)

# def plot_sed2(dir_eazyout, run_name, objid, zgrid, chi2,
#               z_phot=0.1, z_spec=-1.0, mstar=1.0e+9,
#               n_inst=2, n_hst=None, n_jwst=None, color_inst=[], label_inst=[],
#               cut_img=None, pixel_scale=None,
#               f200w_mag=None, r_h=None, f200w_f277w=None, f277w_f356w=None,
#               r_kron=None, axis_ratio=None, theta=None):

id_plots = [id_zpl, id_zph]
for id_arr in id_plots:
    for i in tqdm.trange(len(id_arr)):
        id_obj = id_arr[i]
        idz = np.flatnonzero(dz['id'] == id_obj)[0]
        
        x, y = phot_data['nir_detect']['x'].values[id_obj-1], phot_data['nir_detect']['y'].values[id_obj-1]
        mag_f200w = phot_data['f200w']['mag_auto'].values[id_obj-1]
        r_h = phot_data['nir_detect']['flxrad'].values[id_obj-1]
        a, b = phot_data['nir_detect']['a'].values[id_obj-1], phot_data['nir_detect']['b'].values[id_obj-1]
        theta = phot_data['nir_detect']['theta'].values[id_obj-1]
        r_kron = phot_data['nir_detect']['kron'].values[id_obj-1]*a
        color1 = phot_data['f200w']['mag_auto'].values[id_obj-1]-phot_data['f277w']['mag_auto'].values[id_obj-1]
        color2 = phot_data['f277w']['mag_auto'].values[id_obj-1]-phot_data['f356w']['mag_auto'].values[id_obj-1]

        plot_sed2(dir_output, run_id, id_obj, zgrid, chi2[idz, :],
                  z_phot=dz['z_phot'][idz], z_spec=dz['z_spec'][idz], mstar=dz['mass'][idz],
                  n_inst=2, n_hst=len(id_flt_hst), n_jwst=len(id_flt_jwst),
                  color_inst=["tomato", "darkorange"],
                  label_inst=["Observed HST photometry", "Observed JWST photometry"],
                  cut_img=totimg[round(y-1-rth):round(y-1+rth),
                                 round(x-1-rth):round(x-1+rth)],
                  pixel_scale=0.04, f200w_mag=mag_f200w,
                  r_h=r_h, f200w_f277w=color1, f277w_f356w=color2,
                  r_kron=r_kron, axis_ratio=b/a, theta=theta)
'''

