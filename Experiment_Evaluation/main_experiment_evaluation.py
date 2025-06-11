"""
Medical Image Quality Analysis Main Application
==============================================
Main application for medical image quality assessment.
"""

import os
from enum import Enum
from pathlib import Path

import pandas as pd

import calculation_utils
import constants
import image_processing
import noise_analysis

folder = constants.BASE_FOLDER


class Names(Enum):
    """Enumeration of available dataset names."""
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


def main() -> None:
    """Main execution function for medical image analysis."""

    # Configuration
    do_ct = 1
    root = constants.ROOT_DIRECTORY

    # Discover available datasets
    paths = [x[0] for x in os.walk(folder)][1:]
    print(paths)
    patient_names = [Path(p).name for p in paths]
    print(patient_names)

    ct_dataset_names = [name for name in patient_names if 'FDG' not in name and 'GA' not in name]

    calculation_utils.pre_init()
    operation = 'CT'

    # Initialize result storage
    ga_fwhms_fit = []
    ga_fwhms_org = []
    fdg_fwhms_fit = []
    fdg_fwhms_org = []
    ct_fwhms_fit = []
    ct_fwhms_org = []

    noise_ga_mean = []
    noise_ga_std = []
    noise_ct_mean = []
    noise_ct_std = []
    noise_ct_mean_center = []
    noise_ct_std_center = []
    noise_ct_mean_not_center = []
    noise_ct_std_not_center = []
    noise_fdg_mean = []
    noise_fdg_std = []
    mtf50s_org = []
    mtf50s_fit = []
    mtf10s_org = []
    mtf10s_fit = []

    Path(constants.IMSAVE_PATH).mkdir(parents=True, exist_ok=True)
    # CT Analysis
    if do_ct:
        print("Starting CT analysis...")
        for name in ct_dataset_names:
            try:
                x_pred, y_pred, rebin_x, rebin_y, spacing = image_processing.do_lsf_radial(root, name, operation)
                mean, std = noise_analysis.noise_all_image(root, name)
                mean_center, std_center = noise_analysis.noise_center(root, name)
                mean_not_center, std_not_center = noise_analysis.noise_not_center(root, name)

                fwhm_fit = image_processing.fw_hm_fit(x_pred, y_pred)
                fwhm_org = image_processing.fw_hm_org(rebin_x, rebin_y)

                noise_ct_mean.append(mean)
                noise_ct_std.append(std)
                noise_ct_mean_center.append(mean_center)
                noise_ct_std_center.append(std_center)
                noise_ct_mean_not_center.append(mean_not_center)
                noise_ct_std_not_center.append(std_not_center)

                print(f"{name}: FWHM_org={fwhm_org}, FWHM_fit={fwhm_fit}")

                ct_fwhms_fit.append(fwhm_fit)
                ct_fwhms_org.append(fwhm_org)

                xf1_fit, yf_fit = image_processing.do_mtf_fit(x_pred, y_pred, spacing)
                x50, x10 = image_processing.mtf_val_fit(xf1_fit, yf_fit)
                mtf50s_fit.append(x50)
                mtf10s_fit.append(x10)
                print(f"MTF fit: {x50}, {x10}")

                xf1_org, yf_org = image_processing.do_mtf_org(rebin_x, rebin_y, spacing)
                x50, x10 = image_processing.mtf_val_org(xf1_org, yf_org)
                mtf50s_org.append(x50)
                mtf10s_org.append(x10)
                print(f"MTF org: {x50}, {x10}")

            except Exception as e:
                print(f"Error processing {name}: {e}")

        # Save CT MTF results
        d = {
            "MTF50_org": pd.Series(mtf50s_org, index=ct_dataset_names),
            'MTF50_fit': pd.Series(mtf50s_fit, index=ct_dataset_names),
            "MTF10_org": pd.Series(mtf10s_org, index=ct_dataset_names),
            "MTF10_fit": pd.Series(mtf10s_fit, index=ct_dataset_names),
        }

        df = pd.DataFrame(d)
        file_name = root + '/MTF_val_CT.csv'
        df.to_csv(file_name)

        # Save CT FWHM and noise results
        d = {
            "FWHM_org": pd.Series(ct_fwhms_org, index=ct_dataset_names),
            'FWHM_fit': pd.Series(ct_fwhms_fit, index=ct_dataset_names),
            "noise_mean_all_image": pd.Series(noise_ct_mean, index=ct_dataset_names),
            "noise_std_all_image": pd.Series(noise_ct_std, index=ct_dataset_names),
            "noise_mean_center": pd.Series(noise_ct_mean_center, index=ct_dataset_names),
            "noise_std_center": pd.Series(noise_ct_std_center, index=ct_dataset_names),
            "noise_mean_not_center": pd.Series(noise_ct_mean_not_center, index=ct_dataset_names),
            "noise_std_not_center": pd.Series(noise_ct_std_not_center, index=ct_dataset_names)
        }

        df = pd.DataFrame(d)
        file_name = root + '/FWHM_CT.csv'
        df.to_csv(file_name)

    # ESF Analysis for nuclear medicine datasets
    fdg_dataset_names = [name for name in patient_names if 'FDG' in name and 'PET' not in name]
    fdg_dataset_names = sorted(fdg_dataset_names, key=lambda x: int(x[4:]))
    fdg_pet_dataset_names = [name for name in patient_names if 'FDG_PET' in name]
    fdg_pet_dataset_names = sorted(fdg_pet_dataset_names, key=lambda x: int(x[8:]))
    ga_dataset_names = [name for name in patient_names if 'GA' in name and 'PET' not in name]
    ga_dataset_names = sorted(ga_dataset_names, key=lambda x: int(x[3:]))
    ga_pet_dataset_names = [name for name in patient_names if 'GA_PET' in name]
    ga_pet_dataset_names = sorted(ga_pet_dataset_names, key=lambda x: int(x[7:]))
    esf_dataset_names = fdg_dataset_names + ga_dataset_names

    calculation_utils.pre_init()
    operation = 'ESF'

    print("Starting ESF analysis...")
    for name in esf_dataset_names:
        try:
            x, lsf, x_pred, y_pred = image_processing.esf(root, name, operation)
            fwhm_fit = image_processing.fw_hm_fit(x_pred, y_pred)
            mean, std = noise_analysis.noise_all_image(root, name)

            fwhm_org = image_processing.fw_hm_org(x, lsf)
            print(f"{name}: FWHM_fit={fwhm_fit}, FWHM_org={fwhm_org}")

            if 'FDG' in name:
                fdg_fwhms_fit.append(fwhm_fit)
                fdg_fwhms_org.append(fwhm_org)
                noise_fdg_mean.append(mean)
                noise_fdg_std.append(std)
            elif 'GA' in name:
                ga_fwhms_fit.append(fwhm_fit)
                ga_fwhms_org.append(fwhm_org)
                noise_ga_mean.append(mean)
                noise_ga_std.append(std)

        except Exception as e:
            print(f"Error processing {name}: {e}")

    all_dataset_names = ct_dataset_names + esf_dataset_names

    # Save ESF results
    d = {
        "FDG_FWHM_fit": pd.Series(fdg_fwhms_fit, index=fdg_dataset_names),
        "FDG_FWHM_org": pd.Series(fdg_fwhms_org, index=fdg_dataset_names),
        "noise mean FDG": pd.Series(noise_fdg_mean, index=fdg_dataset_names),
        "noise std FDG": pd.Series(noise_fdg_std, index=fdg_dataset_names),
        "GA_FWHM_fit": pd.Series(ga_fwhms_fit, index=fdg_dataset_names),
        "GA_FWHM_org": pd.Series(ga_fwhms_org, index=fdg_dataset_names),
        "noise mean GA": pd.Series(noise_ga_mean, index=fdg_dataset_names),
        "noise std GA": pd.Series(noise_ga_std, index=fdg_dataset_names),
    }

    df = pd.DataFrame(d)
    df.index.name = 'Iterations'
    file_name = root + '/FWHM_all.csv'
    df.to_csv(file_name)

    print("Analysis completed successfully!")


if __name__ == '__main__':
    main()
