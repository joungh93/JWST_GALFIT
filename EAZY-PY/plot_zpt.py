### Imports

# Printing the versions of packages
from importlib_metadata import version
for pkg in ['numpy', 'matplotlib', 'pandas']:
    print(pkg+": ver "+version(pkg))

# importing necessary modules
import numpy as np
import glob, os, copy
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rc, rcParams
rc('axes', linewidth=2)
rc('font', weight='bold')
rcParams['text.latex.preamble'] = r'\usepackage{sfmath} \boldmath'

from eazy import filters
dir_eazy = "/data01/jhlee/Downloads/eazy-photoz/"
res = filters.FilterFile(dir_eazy+"filters/FILTER.RES.latest")

# from 'FILTER.RES.latest.info' file
def get_centwave(id_filters, res):
    wave = []
    for i in id_filters:
        str_res = str(res[i]).split()
        idx_lam = str_res.index('lambda_c=')+1
        wave.append(float(str_res[idx_lam]))
    return np.array(wave)


# ----- Reading the zeropoints and translates ----- #
dir_output = "EAZY_OUTPUT/"

run_names = ["run1i", "run2i", "run3i", "run4i"]
for i in range(len(run_names)):
    # trls = np.genfromtxt(dir_output+run_names[i]+".eazypy.zphot.translate",
    #                      dtype=None, encoding='ascii', names=('filter','number'))
    zpts = np.genfromtxt(dir_output+run_names[i]+".eazypy.zphot.zeropoint",
                         dtype=None, encoding='ascii', names=('number','corr'))
    fids = [int(nn[1:]) for nn in zpts['number']]
    
    wave = get_centwave(fids, res)
    
    # plt.close('all')
    fig, ax = plt.subplots(figsize=(8,4))

    ax.plot([8.0e+2, 8.0e+4], [1.0, 1.0], '--', color='gray', lw=2.0, alpha=0.6, zorder=1)
    ax.plot(wave[np.argsort(wave)], zpts['corr'][np.argsort(wave)],
            '-', color='slateblue', lw=2.0, alpha=0.8, zorder=2)
    ax.text(0.05, 0.95, run_names[i], fontsize=15.0, fontweight='bold',
            ha='left', va='top', transform=ax.transAxes)

    ax.set_xlim([1.5e+3, 7.0e+4])
    ax.set_xticks([5.0e+3, 1.0e+4, 5.0e+4])
    ax.set_xticklabels([5.0e+3, 1.0e+4, 5.0e+4])
    ax.set_xscale('log')
    ax.set_ylim([0.5, 1.5])
    ax.set_xlabel(r"Wavelength $[{\rm \AA}]$", fontsize=12.0, fontweight='bold')
    ax.set_ylabel("Zeropoint correction", fontsize=12.0, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12.0)
    ax.tick_params(width=2.0, length=8.0)
    ax.tick_params(which='minor', width=2.0, length=4.0)

    plt.tight_layout()
    # plt.show(block=False)
    plt.savefig("ZP_"+run_names[i]+".png", dpi=300)


