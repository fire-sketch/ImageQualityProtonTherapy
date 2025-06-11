"""
Patient Evaluation System
=========================
"""

import os
import re
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd

import Dicts
import evaluation_utils as utils


def make_overlay_gamma(patient: str, external: np.ndarray, cts: np.ndarray, roi: np.ndarray,
                       gammas: np.ndarray, mods: str, name: str) -> None:
    """
    Create gamma overlay visualizations for all orientations.

    Args:
        patient (str): Patient identifier
        external (np.ndarray): External contour
        cts (np.ndarray): CT images
        roi (np.ndarray): Region of interest
        gammas (np.ndarray): Gamma values
        mods (str): Modification type
        name (str): Analysis name
    """
    data_path_out = utils.OUTPUT_PATHS['gamma_overlay'] + '/' + mods + '/' + patient + '/' + name
    Path(data_path_out).mkdir(parents=True, exist_ok=True)

    roi = roi / 255
    roi = roi != 0.0000
    roi = roi.astype(np.uint8)

    ct_corner = np.asarray(Dicts.ct_corners[patient])
    grid_corner = np.asarray(Dicts.grid_corners[patient])
    corner_indexes = np.round(np.abs(ct_corner - grid_corner) * 10).astype(np.int)

    gammas[external == 0.0] = 0
    gammas = np.transpose(gammas, axes=[2, 1, 0])
    roi = np.transpose(roi, axes=[2, 1, 0])
    gammas = gammas.astype(cts.dtype)

    cts = cts[
          corner_indexes[1]:corner_indexes[1] + gammas.shape[1],
          corner_indexes[0]:corner_indexes[0] + gammas.shape[0],
          corner_indexes[2]:corner_indexes[2] + gammas.shape[2]
          ]

    gammas = np.transpose(gammas, axes=[1, 0, 2])
    roi = np.transpose(roi, axes=[1, 0, 2])
    out_img = []

    # Transversal slices
    for i in range(cts.shape[2]):
        out_path = data_path_out + '/transversal'
        Path(out_path).mkdir(parents=True, exist_ok=True)
        heatmap_img = cv2.applyColorMap(gammas[:, :, i], cv2.COLORMAP_JET)
        ct = cv2.merge((cts[:, :, i], cts[:, :, i], cts[:, :, i]))
        added_image = cv2.addWeighted(ct, utils.CT_WEIGHT, heatmap_img, utils.HEATMAP_WEIGHT, 0)
        roi_img = cv2.bitwise_and(added_image, added_image, mask=gammas[:, :, i])
        roi_img[gammas[:, :, i] == 0, ...] = ct[gammas[:, :, i] == 0, ...]
        contours, hierarchy = cv2.findContours(roi[:, :, i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        utils.draw(roi_img, contours, out_path, out_img, i)

    # Sagittal slices
    for i in range(cts.shape[1]):
        out_path = data_path_out + '/sagittal'
        Path(out_path).mkdir(parents=True, exist_ok=True)
        heatmap_img = cv2.applyColorMap(gammas[:, i, :], cv2.COLORMAP_JET)
        ct = cv2.merge((cts[:, i, :], cts[:, i, :], cts[:, i, :]))
        added_image = cv2.addWeighted(ct, utils.CT_WEIGHT, heatmap_img, utils.HEATMAP_WEIGHT, 0)
        roi_img = cv2.bitwise_and(added_image, added_image, mask=gammas[:, i, :])
        roi_img[gammas[:, i, :] == 0, ...] = ct[gammas[:, i, :] == 0, ...]
        contours, hierarchy = cv2.findContours(roi[:, i, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        utils.draw(roi_img, contours, out_path, out_img, i)

    # Coronal slices
    for i in range(cts.shape[0]):
        out_path = data_path_out + '/coronal'
        Path(out_path).mkdir(parents=True, exist_ok=True)
        heatmap_img = cv2.applyColorMap(gammas[i, :, :], cv2.COLORMAP_JET)
        ct = cv2.merge((cts[i, :, :], cts[i, :, :], cts[i, :, :]))
        added_image = cv2.addWeighted(ct, utils.CT_WEIGHT, heatmap_img, utils.HEATMAP_WEIGHT, 0)
        roi_img = cv2.bitwise_and(added_image, added_image, mask=gammas[i, :, :])
        roi_img[gammas[i, :, :] == 0, ...] = ct[gammas[i, :, :] == 0, ...]
        contours, hierarchy = cv2.findContours(roi[i, :, :], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        utils.draw(roi_img, contours, out_path, out_img, i)


def progress(do_overlay: int, folder_selected: str, pat: str, mods: str,
             overlay_widths: List[int], overlay_ground: str = 'mod') -> None:
    """
    Main processing function for patient evaluation.

    Args:
        do_overlay (int): Whether to create overlay visualizations
        folder_selected (str): Selected folder path
        pat (str): Patient identifier
        mods (str): Modification type
        overlay_widths (List[int]): Width parameters for overlay
        overlay_ground (str): Ground truth type for overlay
    """
    utils.clear_dic(Dicts.paths)
    utils.clear_dic(Dicts.dic_gamma)

    output_path = folder_selected + '/analysis_doses'
    os.makedirs(output_path, exist_ok=True)
    print('Selected folder ' + str(folder_selected))
    utils.path_extraction(Dicts.paths, folder_selected)

    dose_original = np.load(Dicts.paths['path_original']) / utils.DOSE_NORMALIZATION_FACTOR
    doses_modified = [np.load(evaluation) / utils.DOSE_NORMALIZATION_FACTOR for evaluation in
                      Dicts.paths['path_evaluations']]
    gammas = [np.load(g) for g in Dicts.paths['path_gamma']]
    rois = [np.load(r) for r in Dicts.paths['path_roi']]

    max_dose = np.max(dose_original)
    dose_cutoff = utils.DOSE_CUTOFF_FRACTION * max_dose
    relevant_slices = (np.max(dose_original, axis=(1, 2)) >= dose_cutoff)
    print('relevant ' + str(relevant_slices.sum()))
    print(f'sum {np.sum(dose_original)}')

    external = rois[0]
    rois_2 = []
    names = []

    for roi, name in zip(rois, Dicts.paths['name_rois']):
        if name == utils.EXTERNAL_ROI_NAME:
            external = roi
            print('External')
            print(external.shape)
        else:
            print(name)
            print(roi.shape)
            rois_2.append(roi)
            names.append(name)

    for i, (gamma, dose_modified) in enumerate(zip(gammas, doses_modified)):
        print('gamma shape' + str(gamma.shape))
        regex = utils.REGEX_PATTERNS['width_pattern']
        name = re.search(regex, Dicts.paths['path_gamma'][i]).group()[:-4]
        w = re.search(r'\d\d|\d', name).group()
        print(f'Evaluating {name} gamma')

        if do_overlay and int(w) in overlay_widths:
            cts = utils.get_cts(pat, overlay_ground, w, mods)
            show_gam = gamma.copy()
            show_gam[show_gam <= utils.GAMMA_PASS_THRESHOLD] = 0
            show_gam[show_gam > utils.GAMMA_PASS_THRESHOLD] = 255
            make_overlay_gamma(pat, external, cts, rois_2[0], show_gam, mods, name)

        maxi = Dicts.pres_dose[pat]
        mask = dose_modified > utils.DOSE_CUTOFF_FRACTION * maxi
        utils.evaluate_gamma(gamma, external, mask, Dicts.dic_gamma, name)

    df = pd.DataFrame.from_dict(Dicts.dic_gamma)
    df = df.sort_values(by=['name', 'noise'])
    patient = re.search(utils.REGEX_PATTERNS['patient_id'], folder_selected).group()
    df.to_csv(folder_selected + '/' + patient + '_analysis.csv')

    for i, roi in enumerate(rois_2):
        utils.clear_dic(Dicts.dic_roi)
        for j, (gamma, dose_modified) in enumerate(zip(gammas, doses_modified)):
            regex = utils.REGEX_PATTERNS['width_pattern']
            name = re.search(regex, Dicts.paths['path_gamma'][j]).group()[:-4]
            print(f'Evaluating {name} gamma roi')
            maxi = Dicts.pres_dose[pat]
            mask = dose_modified > utils.DOSE_CUTOFF_FRACTION * maxi
            utils.evaluate_gamma(gamma, roi, mask, Dicts.dic_roi, name)

        df = pd.DataFrame.from_dict(Dicts.dic_roi)
        df = df.sort_values(by=['name', 'noise'])
        patient = re.search(utils.REGEX_PATTERNS['patient_id'], folder_selected).group()
        df.to_csv(folder_selected + '/' + patient + '_roi' + names[i] + '_analysis.csv')
