import glob
import os
import re
from enum import Enum
from pathlib import Path

import Calculation
import matplotlib.pyplot as plt
import pandas as pd

import Dicts

folder = '../../Experimente/'
# possible operations: LSF, ESF, CT

class Names(Enum):
    FDG_11 = 'FDG_11'
    FDG_12 = 'FDG_12'
    FDG_61 = 'FDG_61'
    FDG_62 = 'FDG_62'
    FDG_151 = 'FDG_151'
    FDG_152 = 'FDG_152'
    FDG_201 = 'FDG_201'
    FDG_202 = 'FDG_202'
    FDG_301 = 'FDG_301'
    FDG_302 = 'FDG_302'
    FDG_PET_6 = 'FDG_PET_6'
    FDG_PET_15 = 'FDG_PET_15'
    FDG_PET_20 = 'FDG_PET_20'
    FDG_PET_30 = 'FDG_PET_30'

    GA_11 = 'GA_11'
    GA_12 = 'GA_12'
    GA_61 = 'GA_61'
    GA_62 = 'GA_62'
    GA_151 = 'GA_151'
    GA_152 = 'GA_152'
    GA_201 = 'GA_201'
    GA_202 = 'GA_202'
    GA_301 = 'GA_301'
    GA_302 = 'GA_302'
    GA_PET_6 = 'GA_PET_6'
    GA_PET_15 = 'GA_PET_15'
    GA_PET_20 = 'GA_PET_20'
    GA_PET_30 = 'GA_PET_30'

    ub_039 = 'ub_039'
    b_039 = 'b_039'
    ub_069 = 'ub_068'
    ub_077 = 'ub_077'
    b_097 = 'b_097'
    b_117 = 'b_117'

do_ct = 1
root = '../../Experimente'
#save = '../../Experimente/images'
paths = [x[0] for x in os.walk(folder)][1:]
print(paths)
patient_names = [Path(p).name for p in paths]
print(patient_names)


names = [name for name in patient_names if 'FDG' not in name and 'GA' not in name]

#names = [e.value for e in Names if 'FDG' not in e.value and 'GA' not in e.value]
Calculation.pre_init()
operation = 'CT'

#if not os.path.exists(save):
    # Create a new directory because it does not exist
   # os.makedirs(save)
GA_FWHMs_fit = []
GA_FWHMs_org = []
FDG_FWHMs_fit = []
FDG_FWHMs_org = []
CT_FWHMs_fit = []
CT_FWHMs_org = []

noise_GA_mean = []
noise_GA_std = []
noise_CT_mean = []
noise_CT_std = []
noise_CT_mean_center = []
noise_CT_std_center = []
noise_CT_mean_not_center = []
noise_CT_std_not_center = []
noise_FDG_mean = []
noise_FDG_std =[]
MTF50s_org = []
MTF50s_fit = []
MTF10s_org = []
MTF10s_fit = []
if do_ct:
    for name in names:

        x_pred, y_pred, rebin_x, rebin_y, spacing = Calculation.do_lsf_radial(root, name, operation)
        mean, std = Calculation.noise_all_image(root,name)
        mean_center,std_center = Calculation.noise_center(root,name)
        mean_not_center,std_not_center = Calculation.noise_not_center(root,name)
    #    save_path = save + '/' + name
        FWHM_fit = Calculation.fw_hm_fit(x_pred, y_pred)

        FWHM_org = Calculation.fw_hm_org(rebin_x, rebin_y)
        noise_CT_mean.append(mean)
        noise_CT_std.append(std)
        noise_CT_mean_center.append(mean_center)
        noise_CT_std_center.append(std_center)
        noise_CT_mean_not_center.append(mean_not_center)
        noise_CT_std_not_center.append(std_not_center)
        print(FWHM_org)
        print(FWHM_fit)

        CT_FWHMs_fit.append(FWHM_fit)
        CT_FWHMs_org.append(FWHM_org)
        xf1_fit, yf_fit = Calculation.do_mtf_fit(x_pred, y_pred, spacing)
        x50, x10 = Calculation.mtf_val_fit(xf1_fit, yf_fit)
        MTF50s_fit.append(x50)
        MTF10s_fit.append(x10)
        print(str(x50) + ' ,' + str(x10))
        xf1_org, yf_org = Calculation.do_mtf_org(rebin_x, rebin_y, spacing)
        x50, x10 = Calculation.mtf_val_org(xf1_org, yf_org)
        MTF50s_org.append(x50)
        MTF10s_org.append(x10)
        print(str(x50) + ' ,' + str(x10))
    d = {
        "MTF50_org": pd.Series(MTF50s_org, index=names),
        'MTF50_fit': pd.Series(MTF50s_fit, index=names),
        "MTF10_org": pd.Series(MTF10s_org, index=names),
        "MTF10_fit": pd.Series(MTF10s_fit, index=names),
    }


    df = pd.DataFrame(d)
    file_name = root + '/MTF_val_CT.csv'
    df.to_csv(file_name)
    index_CT = ['ub_039','b_039','ub_068','ub_077','b_098','b_117']
    d = {
        "FWHM_org": pd.Series(CT_FWHMs_org, index=names),
        'FWHM_fit': pd.Series(CT_FWHMs_fit, index=names),
        "noise_mean_all_image": pd.Series(noise_CT_mean, index=names),
        "noise_std_all_image": pd.Series(noise_CT_std, index=names),
        "noise_mean_center": pd.Series(noise_CT_mean_center, index=names),
        "noise_std_center": pd.Series(noise_CT_std_center, index=names),
        "noise_mean_not_center": pd.Series(noise_CT_mean_not_center, index=names),
        "noise_std_not_center": pd.Series(noise_CT_std_not_center, index=names)
    }


    df = pd.DataFrame(d)
    file_name = root + '/FWHM_CT.csv'
    df.to_csv(file_name)
def sor(x):
    return x[4:-2]


namesFDG = [name for name in patient_names if 'FDG' in name and 'PET' not in name]
namesFDG = sorted(namesFDG,key=lambda x: int(x[4:]))
names_FDG_PET = [name for name in patient_names if 'FDG_PET' in name]
names_FDG_PET = sorted(names_FDG_PET,key=lambda x: int(x[8:]))
names_GA = [name for name in patient_names if 'GA' in name and 'PET' not in name]
names_GA = sorted(names_GA,key=lambda x: int(x[3:]))
names_GA_PET = [name for name in patient_names if 'GA_PET' in name]
names_GA_PET = sorted(names_GA_PET,key=lambda x: int(x[7:]))
names2 = namesFDG + names_GA
#names2 = namesFDG + names_FDG_PET + names_GA + names_GA_PET
#names2 = [e.value for e in Names if 'FDG' in e.value or 'GA' in e.value]
Calculation.pre_init()
operation = 'ESF'
root = '../../Experimente'
#save = '../../Experimente/images'
#if not os.path.exists(save):
    # Create a new directory because it does not exist
    #os.makedirs(save)
for name in names2:
    x, LSF, x_pred, y_pred = Calculation.esf(root, name,operation)
    #name = save + '/' + name
    FWHM_fit = Calculation.fw_hm_fit(x_pred, y_pred)
    mean,std = Calculation.noise_all_image(root,name)
    #plt.plot(x_pred, y_pred)
   # plt.plot(x, LSF, linestyle='--', marker='o', label=name)
   # plt.grid()
    #plt.legend()
    #plt.show()

    FWHM_org = Calculation.fw_hm_org(x, LSF)
    print(name)
    print(FWHM_fit)
    print(FWHM_org)
    if 'FDG' in name:
        FDG_FWHMs_fit.append(FWHM_fit)
        FDG_FWHMs_org.append(FWHM_org)
        noise_FDG_mean.append(mean)
        noise_FDG_std.append(std)
    elif 'GA' in name:
        GA_FWHMs_fit.append(FWHM_fit)
        GA_FWHMs_org.append(FWHM_org)
        noise_GA_mean.append(mean)
        noise_GA_std.append(std)
    print(FWHM_org)
names = names + names2


d = {
    "FDG_FWHM_fit": pd.Series(FDG_FWHMs_fit, index=namesFDG),
    "FDG_FWHM_org": pd.Series(FDG_FWHMs_org, index=namesFDG),
    "noise mean FDG": pd.Series(noise_FDG_mean,index=namesFDG),
    "noise std FDG": pd.Series(noise_FDG_std,index=namesFDG),
    "GA_FWHM_fit": pd.Series(GA_FWHMs_fit, index=namesFDG),
    "GA_FWHM_org": pd.Series(GA_FWHMs_org, index=namesFDG),
    "noise mean GA": pd.Series(noise_GA_mean,index=namesFDG),
    "noise std GA": pd.Series(noise_GA_std,index=namesFDG),
}

df = pd.DataFrame(d)
df.index.name = 'Iterations'
file_name = root + '/FWHM_all.csv'
df.to_csv(file_name)
