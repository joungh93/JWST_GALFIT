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

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings


try:
    print('EAZYCODE = '+os.getenv('EAZYCODE'))
except:
    os.environ['EAZYCODE'] = "/data01/jhlee/Downloads/eazy-photoz/"


# Write .translate file
id_flt_hst  = [233, 236, 238, 239, 240, 202, 203, 205]    # from 'FILTER.RES.latest.info' file
id_flt_jwst = [364, 365, 366, 375, 376, 377]    # from 'FILTER.RES.latest.info' file
bands_hst  = ['f435w', 'f606w', 'f775w', 'f814w', 'f850lp',
              'f105w', 'f125w', 'f160w']
bands_jwst = ['f115w', 'f150w', 'f200w',
              'f277w', 'f356w', 'f444w']
              
def create_translate(band_name, band_id, out):
    with open(out, "w") as fw:
        for i in range(len(band_name)):
            fw.write(band_name[i]+f"_tot_1   F{band_id[i]:d}\n")
            fw.write(band_name[i]+f"_etot_1  E{band_id[i]:d}\n")

create_translate(bands_hst+bands_jwst, id_flt_hst+id_flt_jwst, "total.translate")


# Input parameter set up 
def eazypy_input(catalog, dir_output, output_file,
                 filters="FILTER.RES.latest", temp_file="carnall_sfhz_13.param", #temp_combo="a",
                 wave_file="lambda.def", temperr_file="template_error_cosmos2020.txt",
                 # laf_file=dir_temp+"LAFcoeff.txt", dla_file=dir_temp+"DLAcoeff.txt",
                 apply_prior=False, prior_file="prior_F160W_TAO.dat", prior_filter=365,
                 fix_zspec=False, zmin=0.01, zmax=12.0, zstep=0.01):#, H0=68.4, Omega_M=0.3, Omega_L=0.7):

    params = {}
    params['FILTERS_RES']       = filters
    params['TEMPLATES_FILE']    = temp_file  #'carnall_sfhz_13.param'
    params['WAVELENGTH_FILE ']  = wave_file
    params['TEMP_ERR_FILE']     = temperr_file

    params['TEMP_ERR_A2']       = 1.0
    params['SYS_ERR']           = 0.05

    params['MW_EBV']            = 0.012
    params['CAT_HAS_EXTCORR']   = True

    params['CATALOG_FILE']      = catalog  #"EAZY_INPUT/flux_EAZY_total_run1.cat"

    # dir_out = "EAZY_OUTPUT"
    if not os.path.exists(dir_output):
        os.system("mkdir "+dir_output)
    params['OUTPUT_DIRECTORY']  = dir_output
    params['MAIN_OUTPUT_FILE']  = output_file  #'output.eazypy'

    params['APPLY_PRIOR']       = apply_prior  #True
    params['PRIOR_FILE']        = prior_file  #"prior_F160W_TAO.dat"
    params['PRIOR_FILTER']      = prior_filter  #365
    params['PRIOR_ABZP']        = 23.9
    params['PRIOR_FLOOR']       = 0.01

    params['FIX_ZSPEC']         = fix_zspec
    params['Z_MIN']             = zmin  #0.05
    params['Z_MAX']             = zmax  #12.0
    params['Z_STEP']            = zstep  #0.005
    params['Z_STEP_TYPE']       = 1

    return params


## Load the data 
# translate_file = os.path.join(os.getenv('EAZYCODE'), 'inputs/zphot.translate')


def start_eazypy(param_file, params, NITER=10, translate_file="zphot.translate",
                 output_prefix="output.eazypy", dir_output="EAZY_OUTPUT",
                 zeropoint_file=None, z_def='z_ml', prior=True, bprior=True):

    pred = eazy.photoz.PhotoZ(param_file=param_file,
                              translate_file=translate_file,
                              zeropoint_file=zeropoint_file, 
                              params=params,
                              load_prior=True, load_products=False)
 
    if (zeropoint_file == None):
        NBIN = np.minimum(pred.NOBJ//100, 180)
        pred.param.params['VERBOSITY'] = 1.
        for i in range(NITER):
            print('Iteration: ', i)
            sn = pred.fnu/pred.efnu
            clip = (sn > 1).sum(axis=1) > 4 # Generally make this higher to ensure reasonable fits
            pred.iterate_zp_templates(idx=pred.idx[clip], update_templates=False, 
                                      update_zeropoints=True, iter=i, n_proc=8, 
                                      save_templates=False, error_residuals=False, 
                                      NBIN=NBIN, get_spatial_offset=False)
    else:
        pass

    # Turn off error corrections derived above
    pred.set_sys_err(positive=True)

    # Full catalog
    sample = np.isfinite(pred.cat['z_spec'])

    # fit_parallel renamed to fit_catalog 14 May 2021
    pred.fit_catalog(pred.idx[sample], n_proc=16, verbose=True,
                     get_best_fit=True, prior=prior, beta_prior=bprior)

    # # Show zspec-zphot comparison
    # fig = pred.zphot_zspec()

    # Derived parameters (z params, RF colors, masses, SFR, unobserved magnitude, etc.)
    warnings.simplefilter('ignore', category=RuntimeWarning)
    if (z_def == 'z_ml'):
        zbest = pred.zml
    if (z_def == 'z_chi2'):
        zbest = pred.zchi2
    if (z_def == 'z_ml_fix'):
        zbest = np.where(pred.ZSPEC.data > 0., pred.ZSPEC.data, pred.zml)
    if (z_def == 'z_chi2_fix'):
        zbest = np.where(pred.ZSPEC.data > 0., pred.ZSPEC.data, pred.zchi2)
    zout, hdu = pred.standard_output(zbest=zbest, rf_pad_width=0.5, rf_max_err=2, 
                                     prior=prior, beta_prior=bprior)
    # 'zout' also saved to [MAIN_OUTPUT_FILE].zout.fits
    if (dir_output[-1] != "/"):
        dir_output = dir_output + "/"
    os.system("mv -v "+output_prefix+"* "+dir_output)

    plt.close('all')

    return pred


# ----- Running eazy-py ----- #
param_file = os.path.join(os.getenv('EAZYCODE'), 'src/zphot.param.default')
os.system("cp -rpv "+os.path.join(os.getenv('EAZYCODE'),
          'filters/FILTER.RES.latest')+" .")
os.system("cp -rpv "+os.path.join(os.getenv('EAZYCODE'),
          'templates/uvista_nmf/lambda.def')+" .")
os.system("cp -rpv "+os.path.join(os.getenv('EAZYCODE'),
          'templates/template_error_cosmos2020.txt')+" .")
os.system("cp -rpv "+os.path.join(os.getenv('EAZYCODE'),
          'templates/prior_F160W_TAO.dat')+" .")

dir_input, dir_output = "EAZY_INPUT/", "EAZY_OUTPUT/"
run_name = []
temp_files = "carnall_sfhz_13.param"
translate_files = "total.translate"

for i in range(len(run_name)):
    catalog = dir_input+"flux_EAZY_"+run_name[i]+".cat"
    ezParam = {'filters':"FILTER.RES.latest", 'wave_file':"lambda.def",
               'temperr_file':"template_error_cosmos2020.txt",
               'apply_prior':True, 'prior_file':"prior_F160W_TAO.dat", 'prior_filter':375, 
               'fix_zspec':False, 'zmin':0.05, 'zmax':12.0, 'zstep':0.005}
    params = eazypy_input(catalog, dir_output, output_file=run_name[i]+".eazypy",
                          temp_file=temp_files, **ezParam)
    pred = start_eazypy(param_file, params, NITER=10, translate_file=translate_files,
                        output_prefix=run_name[i]+".eazypy", dir_output=dir_output,
                        zeropoint_file=None, z_def='z_ml', prior=False, bprior=False)
    with open(run_name[i]+".eazypy.zphot.pickle","wb") as fw:
        pickle.dump(pred, fw)
    os.system("mv -v "+run_name[i]+".eazypy"+"* "+dir_output)

# def eazypy_input(catalog, dir_output, output_file,
#                  filters="FILTER.RES.latest", temp_file=temp_file,
#                  wave_file="lambda.def", temperr_file="template_error_cosmos2020.txt",
#                  apply_prior=False, prior_file="prior_F160W_TAO.dat", prior_filter=365,
#                  zmin=0.01, zmax=12.0, zstep=0.01)

# def start_eazypy(param_file, params, NITER=10, translate_file="zphot.translate",
#                  output_prefix="output.eazypy", dir_output="EAZY_OUTPUT",
#                  zeropoint_file=None):
