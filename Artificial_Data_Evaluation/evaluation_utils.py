"""
Evaluation Utility Functions
============================
Utility functions for patient evaluation with constants and helper methods.
"""

import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import natsort
import numpy as np
import pandas as pd
import PIL.Image

# Constants
DATA_PATHS = {
    'rect': Path("../data/rect"),
    'gaussian': Path("../data/gauss"),
    'noise': Path("../data/noise"),
    'gauss_noise': Path("../data/gauss_noise")
}

OUTPUT_PATHS = {
    'png_original': r"../data/output_data/png_original",
    'png_modified': r"../data/output_data/png",
    'gamma_overlay': "../data/output_data/gamma_overlay"
}

# Processing constants
IMAGE_SCALE_PERCENT = 500
DOSE_NORMALIZATION_FACTOR = 100
ROUNDING_PRECISION = 2
GAMMA_PASS_THRESHOLD = 1.0
GAMMA_T2_THRESHOLD = 1.5
DOSE_CUTOFF_FRACTION = 0.1
EXTERNAL_ROI_NAME = 'External_2'
CT_IDENTIFIER = 'CT 1'

# Color and drawing constants
CONTOUR_COLOR = (0, 255, 0)
CONTOUR_THICKNESS = 1
CT_WEIGHT = 0.1
HEATMAP_WEIGHT = 0.9

# Regex patterns
REGEX_PATTERNS = {
    'patient_number': r'\d{2}',
    'width_pattern': r'w(\d|\d\d).*.npy',
    'number_extract': r'\d{1,2}\.?\d?',
    'noise_extract': r'\d{1,3}\.\d{1,2}',
    'patient_id': r'zzzCFPatient\d\d'
}


def get_cts(patient: str, use: str = 'mod', w: str = '1', mods: str = 'gaussian') -> np.ndarray:
    """
    Load CT images for a patient.

    Args:
        patient (str): Patient identifier
        use (str): Type of images to load ('original' or 'mod')
        w (str): Width parameter for modified images
        mods (str): Modification type

    Returns:
        np.ndarray: Stacked CT images as 3D array
    """
    if use == 'original':
        path = OUTPUT_PATHS['png_original'] + "/" + patient
    elif use == 'mod':
        path = OUTPUT_PATHS['png_modified'] + '/' + mods + '/' + patient + 'w' + w
    else:
        raise ValueError(f"Unknown use type: {use}")

    ct_paths = glob.glob(path + '/*.png')
    ct_paths = natsort.natsorted(ct_paths, reverse=True)

    cts = []
    for ct_path in ct_paths:
        cts.append(np.asarray(PIL.Image.open(ct_path)))

    return np.dstack(cts)


def draw(roi_img: np.ndarray, contours: List, out_path: str, out_img: List, i: int) -> None:
    """
    Draw contours on image and save to file.

    Args:
        roi_img (np.ndarray): Image with ROI overlay
        contours (List): Contours to draw
        out_path (str): Output directory path
        out_img (List): List to store output images
        i (int): Index of the current slice
    """
    cv2.drawContours(roi_img, contours, -1, CONTOUR_COLOR, CONTOUR_THICKNESS)
    out_path = out_path + '/' + str(i) + '.png'

    width = int(roi_img.shape[1] * IMAGE_SCALE_PERCENT / 100)
    height = int(roi_img.shape[0] * IMAGE_SCALE_PERCENT / 100)
    size = (width, height)

    output = cv2.resize(roi_img, size)
    output = cv2.rotate(output, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(out_path, output)
    out_img.append(output)


def path_extraction(paths: Dict, folder_selected: str) -> None:
    """
    Extract and organize file paths for analysis.

    Args:
        paths (Dict): Dictionary to store extracted paths
        folder_selected (str): Selected folder path
    """
    path_evaluations = folder_selected + '/doses/*.npy'
    path_evaluations = glob.glob(path_evaluations)
    paths['path_evaluation'] = path_evaluations

    path_original = list(filter(lambda x: CT_IDENTIFIER in x, path_evaluations))[0]
    paths['path_original'] = path_original
    print('original file is ' + str(path_original))
    path_evaluations.remove(path_original)

    path_gamma = folder_selected + '/gamma/*.npy'
    path_gamma = glob.glob(path_gamma)
    paths['path_gamma'] = path_gamma
    print('gamma files are ' + str([os.path.splitext(os.path.basename(g))[0] for g in path_gamma]))

    paths['path_evaluations'] = path_evaluations
    print('evaluation files are ' + str([os.path.splitext(os.path.basename(p))[0] for p in path_evaluations]))

    path_roi = folder_selected + '/roi/*.npy'
    path_roi = glob.glob(path_roi)
    paths['path_roi'] = path_roi
    name_rois = [Path(filepath).stem for filepath in paths['path_roi']]
    paths['name_rois'] = name_rois


def evaluate_gamma(gamma: np.ndarray, roi: np.ndarray, mask: np.ndarray,
                   params: Dict[str, List], name: str) -> None:
    """
    Evaluate gamma values for a given ROI and mask.

    Args:
        gamma (np.ndarray): Gamma values
        roi (np.ndarray): Region of interest
        mask (np.ndarray): Dose mask
        params (Dict[str, List]): Dictionary to store results
        name (str): Name of the evaluation
    """
    n = re.search(REGEX_PATTERNS['number_extract'], name).group()
    n = float(n)

    if 'gamma' in name:
        n = n * 2

    if 'gamma_noise' in name:
        noi = re.search(REGEX_PATTERNS['noise_extract'], name).group()
        noi = float(noi)
        params['noise'].append(noi)
    else:
        params['noise'].append(0)

    params['name'].append(n)
    roi_bin = roi != 0
    mask_2 = mask & roi_bin

    gam_trash = gamma[mask_2]
    valid_gamma = gam_trash[~np.isnan(gam_trash)]

    pass_rate = np.round(np.sum(valid_gamma <= GAMMA_PASS_THRESHOLD) / len(valid_gamma) * 100, ROUNDING_PRECISION)
    mean_gamma = np.round(np.mean(valid_gamma), ROUNDING_PRECISION)
    t2_gamma = np.sum(valid_gamma > GAMMA_T2_THRESHOLD) / len(valid_gamma)

    params['valid_gammas'].append(len(valid_gamma))
    params['pass_rate'].append(pass_rate)
    params['mean_gamma'].append(mean_gamma)
    params['T2_gamma'].append(np.round(t2_gamma, ROUNDING_PRECISION))


def clear_dic(dic: Dict) -> None:
    """
    Clear all lists in a dictionary.

    Args:
        dic (Dict): Dictionary with list values to clear
    """
    for key in dic:
        dic[key] = []