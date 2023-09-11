### Imports

# Printing the versions of packages
from importlib_metadata import version
for pkg in ['numpy', 'matplotlib', 'pandas']:
    print(pkg+": ver "+version(pkg))

# importing necessary modules
import numpy as np
import glob, os, copy
import pandas as pd
from matplotlib import rc, rcParams
rc('axes', linewidth=2)
rc('font', weight='bold')
rcParams['text.latex.preamble'] = r'\usepackage{sfmath} \boldmath'
from matplotlib import pyplot as plt
from matplotlib import ticker
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
run_mode = "run5z"
dz, hz = fits.getdata(dir_output+run_mode+".eazypy.zout.fits", ext=1, header=True)
eff = ((dz['z_phot'] > 0.) & (dz['z_phot_chi2'] > 0.))

print(f"Total      : {len(dz):d}")
print(f"Effective  : {np.sum(eff):d}")

UBVJp = ((dz['restU'] > 0.) & (dz['restB'] > 0.) & \
         (dz['restV'] > 0.) & (dz['restJ'] > 0.))
print(f"UBVJ > 0   : {np.sum(eff & UBVJp):d}")

Meff = (dz['mass'] > 0.)
print(f"M_star > 0 : {np.sum(eff & UBVJp & Meff):d}")

Mcut = ((dz['mass'] > 1.0e+8))# & (dz['Lv'] > 1.0e+9))
print(f"Mcut       : {np.sum(eff & UBVJp & Meff & Mcut):d}")

zcut = ((dz['z_phot'] > 0.5))# & (dz['z_spec'] < 0.0))
print(f"zcut       : {np.sum(eff & UBVJp & Meff & Mcut & zcut):d}")

chi2cut = ((dz['z_phot_chi2'] < 20.0))
print(f"chi2cut    : {np.sum(eff & UBVJp & Meff & Mcut & zcut & chi2cut):d}")

magcut = (phot_data['f200w']['mag_aper'].values[dz['id']-1] < 28.0)
print(f"magcut     : {np.sum(eff & UBVJp & Meff & Mcut & zcut & chi2cut & magcut):d}")


# # ----- Writing regions ----- #
# with open("zg0.reg", "w") as f:
#     f.write('global color=cyan font="helvetica 10 normal" ')
#     f.write("select=1 edit=1 move=1 delete=1 include=1 fixed=0 source width=2\n")
#     f.write("fk5\n")
#     for i in np.arange(np.sum(eff)):
#         idx = dz['id'][eff][i]-1
#         f.write(f"circle({phot_data['f200w']['ra'].values[idx]:.6f}, ")
#         f.write(f"{phot_data['f200w']['dec'].values[idx]:.6f}, ")
#         f.write('1.0")  '+'# text={'+ \
#                 f"{dz['z_spec'][eff][i]:.2f},{dz['z_phot'][eff][i]:.4f}"+'}\n')

# with open("zg2.reg", "w") as f:
#     f.write('global color=blue font="helvetica 10 normal" ')
#     f.write("select=1 edit=1 move=1 delete=1 include=1 fixed=0 source width=2\n")
#     f.write("fk5\n")
#     for i in np.arange(np.sum(dz['z_phot'] > 2.)):
#         idx = dz['id'][dz['z_phot'] > 2.][i]-1
#         f.write(f"circle({phot_data['f200w']['ra'].values[idx]:.6f}, ")
#         f.write(f"{phot_data['f200w']['dec'].values[idx]:.6f}, ")
#         f.write('1.0")  '+'# text={'+ \
#                 f"{dz['z_spec'][dz['z_phot'] > 2.][i]:.2f},{dz['z_phot'][dz['z_phot'] > 2.][i]:.4f}"+'}\n')

# with open("zl0.reg", "w") as f:
#     f.write('global color=magenta font="helvetica 10 normal" ')
#     f.write("select=1 edit=1 move=1 delete=1 include=1 fixed=0 source width=2\n")
#     f.write("fk5\n")
#     for i in np.arange(np.sum(~eff)):
#         idx = dz['id'][~eff][i]-1
#         f.write(f"circle({phot_data['f200w']['ra'].values[idx]:.6f}, ")
#         f.write(f"{phot_data['f200w']['dec'].values[idx]:.6f}, ")
#         f.write('1.0")  '+'# text={'+ \
#                 f"{dz['z_spec'][~eff][i]:.2f},{dz['z_phot'][~eff][i]:.4f}"+'}\n')


##### PLOTS #####
# plt.close('all')

# ----- Plot: z_phot histogram ----- #
log_zmin, log_zmax, nbin = -1.5, 1.1, 40
logbins = np.logspace(log_zmin, log_zmax, nbin)

fig, ax = plt.subplots(figsize=(6,4))
ax.hist(dz['z_phot'][eff], bins=logbins, color='dodgerblue', alpha=0.8)
ax.set_xscale('log')
ax.tick_params(axis='both', labelsize=12.0)
ax.set_xlabel(r"$z_{\rm phot}$", fontsize=12.0)
ax.set_ylabel(r"$N$", fontsize=12.0)
ax.set_xticks([0.1, 1.0, 10.0])
ax.set_xticklabels(["0.1", "1.0", "10.0"])
ax.tick_params(width=1.1, length=8.0)
ax.tick_params(width=1.1, length=5.0, which='minor')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.1)
ax.text(0.04, 0.94, f"All sources : {np.sum(eff):d}",
        fontsize=12.5, color="black", #fontweight="bold", 
        ha="left", va="top", transform=ax.transAxes)
# ax.text(0.04, 0.87, r"$z_{\rm phot} > 1.5$     : "+ \
#         f"{np.sum(eff & (dz['z_phot'] > 1.5)):d}",
#         fontsize=12.5, color="black", #fontweight="bold", 
#         ha="left", va="top", transform=ax.transAxes)
ax.axvline(1.5, 0.0, 1.0, ls='--', lw=1.25, color='gray', alpha=0.8)
plt.tight_layout()
# plt.show(block=False)
plt.savefig("Fig2-zhist.png", dpi=300)
plt.close()


# ----- Plot: z_spec histogram ----- #
log_zmin, log_zmax, nbin = -1.5, 1.1, 40
logbins = np.logspace(log_zmin, log_zmax, nbin)

fig, ax = plt.subplots(figsize=(6,4))
ax.hist(dz['z_spec'][dz['z_spec'] > 0.1], bins=logbins, color='dodgerblue', alpha=0.8)
ax.set_xscale('log')
ax.tick_params(axis='both', labelsize=12.0)
ax.set_xlabel(r"$z_{\rm phot}$", fontsize=12.0)
ax.set_ylabel(r"$N$", fontsize=12.0)
ax.set_xticks([0.1, 1.0, 10.0])
ax.set_xticklabels(["0.1", "1.0", "10.0"])
ax.tick_params(width=1.1, length=8.0)
ax.tick_params(width=1.1, length=5.0, which='minor')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.1)
ax.text(0.04, 0.94, f"Spec-z sources : {np.sum(dz['z_spec'] > 0.1):d}",
        fontsize=12.5, color="black", #fontweight="bold", 
        ha="left", va="top", transform=ax.transAxes)
# ax.text(0.04, 0.87, r"$z_{\rm phot} > 1.5$     : "+ \
#         f"{np.sum(eff & (dz['z_phot'] > 1.5)):d}",
#         fontsize=12.5, color="black", #fontweight="bold", 
#         ha="left", va="top", transform=ax.transAxes)
ax.axvline(1.5, 0.0, 1.0, ls='--', lw=1.25, color='gray', alpha=0.8)
plt.tight_layout()
# plt.show(block=False)
plt.savefig("Fig2-zshist.png", dpi=300)
plt.close()


# # ----- Plot: Color-color diagrams ----- #
cmap = cm.get_cmap("jet")
f_lo, f_hi, nn = 0.1, 0.9, 256
tcmp = tc.truncate_colormap(cmap, f_lo, f_hi, nn)
c_lo, c_hi = np.log10(0.3), np.log10(3.0)
norm = mpl.colors.Normalize(vmin=c_lo, vmax=c_hi)

# fig, ax = plt.subplots(figsize=(5.5, 4.5))
ids = dz['id'][eff]-1
# X = phot_data['f200w']['mag_corr'].values[ids] - phot_data['f277w']['mag_corr'].values[ids]
# Y = phot_data['f277w']['mag_corr'].values[ids] - phot_data['f356w']['mag_corr'].values[ids]

# c_idxs = []
# for i in range(len(ids)):
#     c_idx = f_lo + (f_hi-f_lo)*(np.log10(dz['z_phot'][eff][i])-c_lo) / (c_hi-c_lo)
#     ax.plot(X[i], Y[i], 'o', ms=4.0, color=tcmp(c_idx), mew=0.5, mec='k', alpha=0.6)
#     c_idxs.append(c_idx)

# ax.tick_params(axis='both', labelsize=12.0)
# ax.set_xlim([-1.2, 1.2])
# ax.set_ylim([-1.2, 1.2])
# ax.set_xlabel("F200W-F277W", fontsize=12.0)
# ax.set_ylabel("F277W-F356W", fontsize=12.0)
# ax.tick_params(width=1.1, length=8.0)
# ax.tick_params(width=1.1, length=5.0, which='minor')
# for axis in ['top','bottom','left','right']:
#     ax.spines[axis].set_linewidth(1.1)

# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="4%", pad=0.08)
# cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=tcmp), cax=cax)
# cb.set_label(r"log $z_{\rm phot}$", size=12.0, labelpad=10.0)
# cb.ax.tick_params(direction='in', labelsize=11.0)

# plt.tight_layout()
# plt.show(block=False)


# ----- Plot: Color-magnitude diagrams ----- #
fig, ax = plt.subplots(figsize=(5.5, 4.5))
X = phot_data['f200w']['mag_corr'].values[ids] - phot_data['f356w']['mag_corr'].values[ids]
Y = phot_data['f277w']['mag_corr'].values[ids]
ax.hexbin(X, Y, np.log10(dz['z_phot'][eff]), gridsize=50, cmap=tcmp, norm=norm,
          edgecolors='k', linewidths=0.3, alpha=0.7,
          extent=(-2.0, 2.0, 31.0, 17.0))

ax.tick_params(axis='both', labelsize=12.0)
ax.set_xlim([-2.0, 2.0])
ax.set_ylim([31.0, 17.0])
ax.set_xlabel("F200W-F356W", fontsize=12.0)
ax.set_ylabel("F277W", fontsize=12.0)
ax.tick_params(width=1.1, length=8.0)
ax.tick_params(width=1.1, length=5.0, which='minor')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.1)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.08)
cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=tcmp), cax=cax)
cb.set_label(r"log $z_{\rm phot}$", size=12.0, labelpad=10.0)
cb.ax.tick_params(direction='in', labelsize=11.0)

plt.tight_layout()
# plt.show(block=False)
plt.savefig("Fig2-cmd.png", dpi=300)
plt.close()


# ----- Plot: Size-magnitude diagrams ----- #
fig, ax = plt.subplots(figsize=(5.5, 4.5))
X = phot_data['f277w']['mag_corr'].values[ids]
Y = 0.04 * phot_data['nir_detect']['flxrad'].values[ids]
ax.hexbin(X, Y, np.log10(dz['z_phot'][eff]), gridsize=50, cmap=tcmp, norm=norm,
          edgecolors='k', linewidths=0.3, alpha=0.7,
          extent=(17.0, 31.0, -0.05, 1.0))

ax.tick_params(axis='both', labelsize=12.0)
ax.set_xlim([17.0, 31.0])
ax.set_ylim([-0.05, 1.0])
ax.set_xlabel("F277W", fontsize=12.0)
ax.set_ylabel("FLUX_RADIUS [arcsec]", fontsize=12.0)
ax.tick_params(width=1.1, length=8.0)
ax.tick_params(width=1.1, length=5.0, which='minor')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.1)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.08)
cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=tcmp), cax=cax)
cb.set_label(r"log $z_{\rm phot}$", size=12.0, labelpad=10.0)
cb.ax.tick_params(direction='in', labelsize=11.0)

plt.tight_layout()
# plt.show(block=False)
plt.savefig("Fig2-size.png", dpi=300)
plt.close()


# ----- Plot: magnitude-mu0 diagrams ----- #
fig, ax = plt.subplots(figsize=(5.5, 4.5))
X = phot_data['f277w']['mag_corr'].values[ids]
Y = phot_data['f277w']['mu0'].values[ids]
ax.hexbin(X, Y, np.log10(dz['z_phot'][eff]), gridsize=50, cmap=tcmp, norm=norm,
          edgecolors='k', linewidths=0.3, alpha=0.7,
          extent=(17.0, 31.0, 14.0, 28.0))

ax.tick_params(axis='both', labelsize=12.0)
ax.set_xlim([17.0, 31.0])
ax.set_ylim([14.0, 28.0])
ax.set_xlabel("F277W", fontsize=12.0)
ax.set_ylabel(r"$\mu_{\rm 0,F277W}~[{\rm mag~arcsec^{-2}}]$", fontsize=12.0)
ax.tick_params(width=1.1, length=8.0)
ax.tick_params(width=1.1, length=5.0, which='minor')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.1)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.08)
cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=tcmp), cax=cax)
cb.set_label(r"log $z_{\rm phot}$", size=12.0, labelpad=10.0)
cb.ax.tick_params(direction='in', labelsize=11.0)

plt.tight_layout()
# plt.show(block=False)
plt.savefig("Fig2-mu0.png", dpi=300)
plt.close()


# ----- Plot: UVJ diagrams ----- #
fig, ax = plt.subplots(figsize=(5.5, 4.5))
X = -2.5*np.log10(dz['restV'][eff & UBVJp]) - -2.5*np.log10(dz['restJ'][eff & UBVJp])
Y = -2.5*np.log10(dz['restU'][eff & UBVJp]) - -2.5*np.log10(dz['restV'][eff & UBVJp])

ax.hexbin(X, Y, np.log10(dz['z_phot'][eff & UBVJp]), gridsize=50, cmap=tcmp, norm=norm,
          edgecolors='k', linewidths=0.3, alpha=0.7,
          extent=(-1.5, 3.0, -1.0, 3.5))
# for i in range(len(ids)):
    # c_idx = f_lo + (f_hi-f_lo)*(np.log10(dz['z_phot'][eff][i])-c_lo) / (c_hi-c_lo)
    # ax.plot(X[i], Y[i], 'o', ms=4.0, color=tcmp(c_idx), mew=0.5, mec='k', alpha=0.6)

ax.tick_params(axis='both', labelsize=12.0)
ax.set_xlim([-1.5, 3.0])
ax.set_ylim([-1.0, 3.5])
ax.set_xlabel(r"$V-J$", fontsize=12.0)
ax.set_ylabel(r"$U-V$", fontsize=12.0)
ax.tick_params(width=1.1, length=8.0)
ax.tick_params(width=1.1, length=5.0, which='minor')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.1)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.08)
cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=tcmp), cax=cax)
cb.set_label(r"log $z_{\rm phot}$", size=12.0, labelpad=10.0)
cb.ax.tick_params(direction='in', labelsize=11.0)

plt.tight_layout()
# plt.show(block=False)
plt.savefig("Fig2-UVJ.png", dpi=300)
plt.close()


# ----- Plot: Stellar mass histogram ----- #
log_Mmin, log_Mmax, nbin = 3.0, 13.0, 40
logbins = np.logspace(log_Mmin, log_Mmax, nbin)

fig, ax = plt.subplots(figsize=(6,4))
ax.hist(dz['mass'][eff & UBVJp & Meff], bins=logbins, color='dodgerblue', alpha=0.8)
ax.set_xscale('log')
ax.tick_params(axis='both', labelsize=12.0)
ax.set_xlabel(r"$M_{\ast}/M_{\odot}$", fontsize=12.0)
ax.set_ylabel(r"$N$", fontsize=12.0)
ax.set_xlim([1.0e+3, 1.0e+13])
# ax.set_xticks([0.1, 1.0, 10.0])
# ax.set_xticklabels(["0.1", "1.0", "10.0"])
ax.tick_params(width=1.1, length=8.0)
ax.tick_params(width=1.1, length=5.0, which='minor')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.1)
ax.text(0.04, 0.94, f"All sources : {np.sum(eff):d}",
        fontsize=12.5, color="black", #fontweight="bold", 
        ha="left", va="top", transform=ax.transAxes)
ax.text(0.04, 0.87, r"${\rm log}~M_{\ast}/M_{\odot} > 9.0$ : "+ \
        f"{np.sum(eff & UBVJp & Meff & (np.log10(dz['mass']) > 9.0)):d}",
        fontsize=12.5, color="black", #fontweight="bold", 
        ha="left", va="top", transform=ax.transAxes)
ax.axvline(10.0**(9.5), 0.0, 1.0, ls='--', lw=1.25, color='gray', alpha=0.8)
plt.tight_layout()
# plt.show(block=False)
plt.savefig("Fig2-mhist.png", dpi=300)
plt.close()


# ----- Plot: Mass-magnitude diagrams ----- #
idm = dz['id'][eff & UBVJp & Meff]-1

fig, ax = plt.subplots(figsize=(5.5, 4.5))
X = np.log10(dz['mass'][eff & UBVJp & Meff])
Y = phot_data['f277w']['mu0'].values[idm]
ax.hexbin(X, Y, np.log10(dz['z_phot'][eff & UBVJp & Meff]), gridsize=50, cmap=tcmp, norm=norm,
          edgecolors='k', linewidths=0.3, alpha=0.7,
          extent=(3.0, 13.0, 28.0, 14.0))

ax.tick_params(axis='both', labelsize=12.0)
ax.set_xlim([3.0, 13.0])
ax.set_ylim([28.0, 14.0])
ax.set_xlabel(r"${\rm log}~M_{\ast}/M_{\odot}$", fontsize=12.0)
ax.set_ylabel("F277W [magnitude]", fontsize=12.0)
ax.tick_params(width=1.1, length=8.0)
ax.tick_params(width=1.1, length=5.0, which='minor')
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.1)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.08)
cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=tcmp), cax=cax)
cb.set_label(r"log $z_{\rm phot}$", size=12.0, labelpad=10.0)
cb.ax.tick_params(direction='in', labelsize=11.0)

plt.tight_layout()
# plt.show(block=False)
plt.savefig("Fig2-mass_mag.png", dpi=300)
plt.close()


# ----- Plot: chi-square histogram ----- #
chi2bins = np.linspace(0.0, 100.0, 40)

fig, ax = plt.subplots(figsize=(6,4))
ax.hist(dz['z_phot_chi2'][eff & UBVJp & Meff & Mcut & zcut],
        bins=chi2bins, color='dodgerblue', alpha=0.8)
# ax.set_xscale('log')
# ax.tick_params(axis='both', labelsize=12.0)
# ax.set_xlabel(r"$M_{\ast}/M_{\odot}$", fontsize=12.0)
# ax.set_ylabel(r"$N$", fontsize=12.0)
# ax.set_xlim([1.0e+3, 1.0e+13])
# # ax.set_xticks([0.1, 1.0, 10.0])
# # ax.set_xticklabels(["0.1", "1.0", "10.0"])
# ax.tick_params(width=1.1, length=8.0)
# ax.tick_params(width=1.1, length=5.0, which='minor')
# for axis in ['top','bottom','left','right']:
#     ax.spines[axis].set_linewidth(1.1)
# ax.text(0.04, 0.94, f"All sources : {np.sum(eff):d}",
#         fontsize=12.5, color="black", #fontweight="bold", 
#         ha="left", va="top", transform=ax.transAxes)
# ax.text(0.04, 0.87, r"${\rm log}~M_{\ast}/M_{\odot} > 9.0$ : "+ \
#         f"{np.sum(eff & UBVJp & Meff & (np.log10(dz['mass']) > 9.0)):d}",
#         fontsize=12.5, color="black", #fontweight="bold", 
#         ha="left", va="top", transform=ax.transAxes)
# ax.axvline(10.0**(9.5), 0.0, 1.0, ls='--', lw=1.25, color='gray', alpha=0.8)
plt.tight_layout()
# plt.show(block=False)
plt.savefig("Fig2-chi2hist.png", dpi=300)
plt.close()


# ----- Plot: chi-square vs. magnitude diagram ----- #
fig, ax = plt.subplots(figsize=(6,4))
idxs = dz['id'][eff & UBVJp & Meff & Mcut & zcut]-1
ax.plot(phot_data['f200w']['mag_aper'].values[idxs],
        dz['z_phot_chi2'][eff & UBVJp & Meff & Mcut & zcut],
        'o', ms=3.0, color='dodgerblue', alpha=0.8)
# ax.set_xscale('log')
# ax.tick_params(axis='both', labelsize=12.0)
# ax.set_xlabel(r"$M_{\ast}/M_{\odot}$", fontsize=12.0)
# ax.set_ylabel(r"$N$", fontsize=12.0)
# ax.set_xlim([1.0e+3, 1.0e+13])
# # ax.set_xticks([0.1, 1.0, 10.0])
# # ax.set_xticklabels(["0.1", "1.0", "10.0"])
# ax.tick_params(width=1.1, length=8.0)
# ax.tick_params(width=1.1, length=5.0, which='minor')
# for axis in ['top','bottom','left','right']:
#     ax.spines[axis].set_linewidth(1.1)
# ax.text(0.04, 0.94, f"All sources : {np.sum(eff):d}",
#         fontsize=12.5, color="black", #fontweight="bold", 
#         ha="left", va="top", transform=ax.transAxes)
# ax.text(0.04, 0.87, r"${\rm log}~M_{\ast}/M_{\odot} > 9.0$ : "+ \
#         f"{np.sum(eff & UBVJp & Meff & (np.log10(dz['mass']) > 9.0)):d}",
#         fontsize=12.5, color="black", #fontweight="bold", 
#         ha="left", va="top", transform=ax.transAxes)
# ax.axvline(10.0**(9.5), 0.0, 1.0, ls='--', lw=1.25, color='gray', alpha=0.8)
plt.tight_layout()
# plt.show(block=False)
plt.savefig("Fig2-chi2mag.png", dpi=300)
plt.close()



# ----- Plot: chi-square vs. z_phot diagram ----- #
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(dz['z_phot'][eff & UBVJp & Meff & Mcut & zcut],
        dz['z_phot_chi2'][eff & UBVJp & Meff & Mcut & zcut],
        'o', ms=3.0, color='dodgerblue', alpha=0.8)
# ax.set_xscale('log')
# ax.tick_params(axis='both', labelsize=12.0)
# ax.set_xlabel(r"$M_{\ast}/M_{\odot}$", fontsize=12.0)
# ax.set_ylabel(r"$N$", fontsize=12.0)
# ax.set_xlim([1.0e+3, 1.0e+13])
# # ax.set_xticks([0.1, 1.0, 10.0])
# # ax.set_xticklabels(["0.1", "1.0", "10.0"])
# ax.tick_params(width=1.1, length=8.0)
# ax.tick_params(width=1.1, length=5.0, which='minor')
# for axis in ['top','bottom','left','right']:
#     ax.spines[axis].set_linewidth(1.1)
# ax.text(0.04, 0.94, f"All sources : {np.sum(eff):d}",
#         fontsize=12.5, color="black", #fontweight="bold", 
#         ha="left", va="top", transform=ax.transAxes)
# ax.text(0.04, 0.87, r"${\rm log}~M_{\ast}/M_{\odot} > 9.0$ : "+ \
#         f"{np.sum(eff & UBVJp & Meff & (np.log10(dz['mass']) > 9.0)):d}",
#         fontsize=12.5, color="black", #fontweight="bold", 
#         ha="left", va="top", transform=ax.transAxes)
# ax.axvline(10.0**(9.5), 0.0, 1.0, ls='--', lw=1.25, color='gray', alpha=0.8)
plt.tight_layout()
# plt.show(block=False)
plt.savefig("Fig2-chi2zph.png", dpi=300)
plt.close()


# ----- Writing region files ----- #
cnd = (eff & UBVJp & Meff & Mcut & zcut & chi2cut & magcut)
with open("target1.reg", "w") as f:
    f.write('global color=magenta font="helvetica 10 normal" ')
    f.write("select=1 edit=1 move=1 delete=1 include=1 fixed=0 source width=2\n")
    f.write("fk5\n")
    for i in np.arange(np.sum(cnd)):
        idx = dz['id'][cnd][i]-1
        f.write(f"circle({phot_data['f200w']['ra'].values[idx]:.6f}, ")
        f.write(f"{phot_data['f200w']['dec'].values[idx]:.6f}, ")
        f.write('0.54")  '+'# text={'+ \
                f"ID{dz['id'][cnd][i]:d}, {dz['z_phot'][cnd][i]:.4f}, ")
        f.write(f"{phot_data['f200w']['mag_aper'].values[idx]:.2f}, ")
        f.write(f"{dz['z_phot_chi2'][cnd][i]:.1f}"+'}\n')


# ----- Writing results ----- #
with open("target1.csv", "w") as f:
    f.write("ID,z,z_flag,z_phot_chi2,logmass,restU,restB,restV,restJ\n")
    for i in np.arange(np.sum(cnd)):
        if (dz['z_spec'][cnd][i] > 0.):
            z_flag = "spec"
        else:
            z_flag = "phot"
        f.write(f"{dz['id'][cnd][i]:>7d},{dz['z_phot'][cnd][i]:>7.4f},"+z_flag+",")
        f.write(f"{dz['z_phot_chi2'][cnd][i]:>.3e},{np.log10(dz['mass'][cnd][i]):>7.4f},")
        f.write(f"{dz['restU'][cnd][i]:>7.4f},{dz['restB'][cnd][i]:>7.4f},")
        f.write(f"{dz['restV'][cnd][i]:>7.4f},{dz['restJ'][cnd][i]:>7.4f}\n")
