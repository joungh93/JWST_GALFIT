#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:58:18 2020

@author: jlee
"""


import time
start_time = time.time()

import numpy as np
import glob, os, copy
import pandas as pd


# Copying the template file to the current diretory
dir_eazy = "/data01/jhlee/Downloads/eazy-photoz/"
dirs_temp = [dir_eazy+"templates/fsps_full/",
             # dir_eazy+"templates/PEGASE2.0/",
             dir_eazy+"templates/sfhz/"]
temp_files = ["tweak_fsps_QSF_12_v3.param",
              # "pegase13.spectra.param",
              "carnall_sfhz_13.param"]

for i in range(len(temp_files)):
    os.system("cp -rpv "+dirs_temp[i]+temp_files[i]+" .")
    os.system("cp -rpv "+dirs_temp[i]+temp_files[i]+".fits .")


# Reading & rewriting the template file with right path
for i in range(len(temp_files)):
    with open(temp_files[i], "r") as f:
        ll = f.readlines()

    nll = []
    for j in range(len(ll)):
        ls = ll[j].split()
        datfile = ls[1]
        ls[1] = dirs_temp[i]+datfile.split("/")[-1]
        nll.append("  ".join(ls)+"\n")

    with open(temp_files[i], "w") as f:
        f.writelines(nll)


# Creating the running file of EAZY-PY
def replace_line(read_file, out_file, line_num, text):
    with open(read_file, 'r') as f:
        lines = f.readlines()
    lines[line_num-1] = text

    with open(out_file, 'w') as g:
        g.writelines(lines)


with open("run_eazy.py", "r") as f:
    ll = f.readlines()
pll = pd.Series(ll)
lni = 1 + np.arange(len(ll))

find_line = ["run_name = ", 
             "               'fix_zspec':",
             "                        zeropoint_file="]
ln1 = lni[pll.str.startswith(find_line[0]).values][0]
ln2 = lni[pll.str.startswith(find_line[1]).values][0]
ln3 = lni[pll.str.startswith(find_line[2]).values][0]


os.system("rm -rfv run_eazy_*.sh")
dir_output = "EAZY_OUTPUT/"
run_names = ["run1i", "run2i", "run3i", "run4i",
             "run1z", "run2z", "run3z", "run4z", "run5z", "run6z"]


### Initial run
code0 = "run_eazy.py"
for i in range(len(run_names)):
    code1 = "run_eazy_"+run_names[i][3:]+".py"
    if (run_names[i][-1] == 'i'):
        replace_line(code0, code1, ln1,
                     find_line[0]+'["'+run_names[i]+'"]\n')
        replace_line(code1, code1, ln2,
                     find_line[1]+ \
                     "False, 'zmin':0.05, 'zmax':12.0, 'zstep':0.005}\n")
        replace_line(code1, code1, ln3,
                     find_line[2] + \
                     'None, z_def="z_ml", prior=False, bprior=False)\n')
                     # '"reference.zeropoint", z_def="z_chi2", prior=False, bprior=False)\n')
    else:
        pass

with open("run_eazy_init1.sh", "w") as f:
    # f.write("START=$(date "+%s")\n")
    f.write("nohup time python run_eazy_1i.py > eazy_run1i.log &\n")
    f.write("nohup time python run_eazy_2i.py > eazy_run2i.log &\n")
    # f.write("END=$(date "+%s")\n")
    # f.write('echo "Running time: "$((END-START))" sec"')

with open("run_eazy_init2.sh", "w") as f:
    f.write("nohup time python run_eazy_3i.py > eazy_run3i.log &\n")
    f.write("nohup time python run_eazy_4i.py > eazy_run4i.log &\n")


###############################################################
########## Comment this part before the initial run! ##########
###############################################################

### Fixed zeropoint run
zrun_names = ["run1p", "run1z", "run5z"]    # Need to be revised!
zp_file = dir_output+"run1i.eazypy.zphot.zeropoint"    # Need to be revised!
best_ztype = "z_chi2"    # Need to be revised!

os.system("cp -rpv "+zp_file+" reference.zeropoint")
for irun in zrun_names[:-1]:
    code1 = "run_eazy_"+irun[3:]+".py"
    replace_line(code0, code1, ln1,
                 find_line[0]+'["'+irun+'"]\n')

    if (irun[-1] == "p"):
        replace_line(code1, code1, ln2,
                     find_line[1]+ \
                     "False, 'zmin':0.05, 'zmax':12.0, 'zstep':0.005}\n")

        replace_line(code1, code1, ln3,
                     find_line[2] + \
                     '"reference.zeropoint", z_def="'+best_ztype+'", prior=False, bprior=False)\n')

    if (irun[-1] == "z"):
        replace_line(code1, code1, ln2,
                     find_line[1]+ \
                     "True, 'zmin':0.05, 'zmax':12.0, 'zstep':0.005}\n")

        replace_line(code1, code1, ln3,
                     find_line[2] + \
                     '"reference.zeropoint", z_def="'+best_ztype+'_fix", prior=False, bprior=False)\n')

with open("run_eazy_zero1.sh", "w") as f:
    for irun in zrun_names[:-1]:
        f.write("nohup time python run_eazy_"+irun[3:]+".py > ")
        f.write("eazy_"+irun+".log &\n")


### Final run (fixed zeropoint + fixed spec-z value)
code1 = "run_eazy_"+zrun_names[-1][3:]+".py"

replace_line(code0, code1, ln1,
             find_line[0]+'["'+zrun_names[-1]+'"]\n')

replace_line(code1, code1, ln2,
             find_line[1]+ \
             "True, 'zmin':0.05, 'zmax':12.0, 'zstep':0.005}\n")

replace_line(code1, code1, ln3,
             find_line[2] + \
             '"reference.zeropoint", z_def="'+best_ztype+'_fix", prior=False, bprior=False)\n')

with open("run_eazy_zero2.sh", "w") as f:
    f.write("nohup time python run_eazy_"+zrun_names[-1][3:]+".py > ")
    f.write("eazy_"+zrun_names[-1]+".log &\n")

###############################################################
###############################################################
###############################################################


# Printing the running time
print(f"--- {time.time()-start_time:.4f} sec ---")

