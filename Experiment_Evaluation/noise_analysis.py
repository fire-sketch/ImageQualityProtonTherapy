"""
Noise Analysis Functions

"""

import glob
import numpy as np
import pydicom
from typing import Tuple, List

import Lookup_Data
import constants


def noise_all_image(root: str, name: str) -> Tuple[float, float]:
    """
    Calculate noise statistics from entire image region.

    Args:
        root (str): Root directory path
        name (str): Dataset name

    Returns:
        Tuple[float, float]: Mean and standard deviation of noise
    """
    params = Lookup_Data.get_noise_data(root, name)
    cor = params.cor
    roi = constants.NOISE_ROI_SIZE_DEFAULT

    if name in constants.SPECIAL_DATASETS_UB_B:
        roi = constants.NOISE_ROI_SIZE_SPECIAL

    if 'FDG' in name or 'GA' in name:
        cor = constants.NOISE_FDG_GA_COORDINATES
        roi = constants.NOISE_ROI_SIZE_FDG_GA

    mean, std = noi(roi, params, cor)
    return mean, std


def noi(roi: int, params, cor: List[int]) -> Tuple[float, float]:
    """
    Calculate noise statistics from specified region of interest.

    Args:
        roi (int): Region of interest size (half-width)
        params: Parameter object containing path and slice information
        cor (List[int]): Center coordinates [x, y]

    Returns:
        Tuple[float, float]: Rounded mean and standard deviation
    """
    slices = params.slices
    files = sorted(glob.glob(params.path + '/*.dcm'), key=len)
    data = []

    for file in files:
        data.append(pydicom.read_file(file))

    slices = data[slices[0]:slices[1]]

    s = np.array([s.pixel_array for s in slices])
    sl = s[:, cor[1] - roi:cor[1] + roi, cor[0] - roi:cor[0] + roi]

    mean = np.round(np.mean(sl), constants.NOISE_PRECISION)
    std = np.round(np.std(sl), constants.NOISE_PRECISION)

    return mean, std


def noise_center(root: str, name: str) -> Tuple[float, float]:
    """
    Calculate noise statistics from center region of image.

    Args:
        root (str): Root directory path
        name (str): Dataset name

    Returns:
        Tuple[float, float]: Mean and standard deviation from center region
    """
    params = Lookup_Data.get_noise_data(root, name)
    cor = params.cor
    roi = constants.NOISE_ROI_SIZE_SMALL
    return noi(roi, params, cor)


def noise_not_center(root: str, name: str) -> Tuple[float, float]:
    """
    Calculate noise statistics from off-center region of image.

    Args:
        root (str): Root directory path
        name (str): Dataset name

    Returns:
        Tuple[float, float]: Mean and standard deviation from off-center region
    """
    params = Lookup_Data.get_noise_data(root, name)
    cor = constants.NOISE_NOT_CENTER_DEFAULT
    roi = constants.NOISE_ROI_SIZE_SMALL

    if name in constants.SPECIAL_DATASETS_UB_B:
        cor = constants.NOISE_NOT_CENTER_SPECIAL

    return noi(roi, params, cor)