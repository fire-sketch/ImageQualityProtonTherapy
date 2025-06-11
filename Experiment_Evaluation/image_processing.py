"""
Image Processing and Analysis Functions

"""

from typing import Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq
from scipy import ndimage
from scipy.optimize import curve_fit

import calculation_utils
import constants
from Experiment_Evaluation.constants import IMSAVE_PATH


def esf(root: str, name: str, operation: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Edge Spread Function (ESF) analysis.

    Calculates the Edge Spread Function and derives the Line Spread Function (LSF)
    from medical imaging data. Includes preprocessing, curve fitting, and normalization.

    Args:
        root (str): Root directory path
        name (str): Dataset name
        operation (str): Operation type ('ESF')

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            x coordinates, LSF data, fitted x coordinates, fitted y values
    """
    s, cor_y, cor_x, data, factor = calculation_utils.init(root, name, operation)
    p0 = constants.GAUSS_P0_DEFAULT
    if 'FDG' in name or 'GA' in name:
        p0 = constants.GAUSS_P0_FDG_GA
    z_summed = np.zeros((cor_x[1] - cor_x[0], cor_y[1] - cor_y[0]))
    for i, sli in enumerate(s):
        sl = sli[cor_x[0]:cor_x[1], cor_y[0]:cor_y[1]]
        z_summed = z_summed - sl
    debug_datasets = constants.DEBUG_DATASETS
    if name in debug_datasets:
        plt.imshow(z_summed, cmap='gray')
        plt.title("summed over z")
        plt.gca().set_axis_off()
        plt.savefig(f'{IMSAVE_PATH}/LSF_CT_filter_{name}.png')
        plt.close()
    # plt.show()
    z_summed = z_summed / s.shape[0]
    esf_ = -np.sum(z_summed, axis=1) / z_summed.shape[1]
    if name in debug_datasets:
        plt.plot(esf_, marker='o')
        plt.title("ESF")
        plt.xlabel('distance in mm')
        plt.grid()
        plt.ylabel('counts')

        plt.savefig(IMSAVE_PATH + '/ESF.png')
        plt.close()
        # plt.show()
    lsf = np.diff(esf_)
    if name in debug_datasets:
        plt.plot(lsf, marker='o')
        plt.title("LSF")
        plt.xlabel('distance in mm')
        plt.grid()
        plt.ylabel('counts')

        plt.savefig(IMSAVE_PATH + '/LSF_step1.png')
        plt.close()
        # plt.show()
    noise_indices = constants.NOISE_INDICES_ESF
    x = np.arange(len(lsf), dtype='float')
    x = x * data[0].PixelSpacing[0]
    x = x - x[np.argmax(lsf)]
    lsf = calculation_utils.linear_detrend(lsf, noise_indices)

    lsf = lsf - np.mean(lsf[noise_indices])
    if name in debug_datasets:
        plt.plot(x, lsf, marker='o')
        plt.title("LSF detrended and denoised")
        plt.xlabel('distance in mm')
        plt.grid()
        plt.ylabel('counts')

        plt.legend()
        plt.savefig(IMSAVE_PATH + '/LSFstep2.png')
        plt.close()
        # plt.show()
    function = calculation_utils.gauss
    x_pred = np.arange(x[0], x[-1], constants.PIXEL_SPACING_PRECISION * data[0].PixelSpacing[0])
    p_opt, p_cov = curve_fit(function, x, lsf, p0=p0)
    y_pred = function(x_pred, *p_opt)
    y_max = np.max(y_pred)
    y_pred = y_pred / y_max
    lsf = lsf / y_max

    x_max = x_pred[np.argmax(y_pred)]
    x = x - x_max
    x_pred = x_pred - x_max
    if name in debug_datasets:
        plt.plot(x, lsf, marker='o', label='normalize')
        plt.plot(x_pred, y_pred)
        plt.title("fitted and normalized LSF")
        plt.xlabel('distance in mm')
        plt.grid()
        plt.ylabel('counts (normalized)')
        plt.savefig(IMSAVE_PATH + '/LSFstep3.png')
        plt.close()
        # plt.show()
    return x, lsf, x_pred, y_pred


def do_lsf_radial(root: str, name: str, operation: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Perform radial Line Spread Function analysis for CT imaging.

    Analyzes point spread function by rotating image and averaging LSF measurements
    across multiple angles to obtain rotationally averaged MTF characteristics.

    Args:
        root (str): Root directory path
        name (str): Dataset name
        operation (str): Operation type ('CT')

    Returns:
        Tuple: Fitted x coordinates, fitted y values, rebinned x, rebinned y, pixel spacing
    """
    p0 = constants.GAUSS_P0_DEFAULT
    if 'FDG' in name or 'GA' in name:
        p0 = constants.GAUSS_P0_FDG_GA
    s, cor, data, factor = calculation_utils.init(root, name, operation)
    roi = constants.CT_ROI_SIZE
    s = s - constants.CT_OFFSET_VALUE
    z_summed = np.zeros((2 * roi, 2 * roi))

    print(name)
    for i, sli in enumerate(s):
        sl = sli[cor[1] - roi:cor[1] + roi, cor[0] - roi:cor[0] + roi]
        z_summed = z_summed - sl
    z_summed = z_summed / s.shape[0]

    angles = np.arange(constants.CT_ANGLE_START, constants.CT_ANGLE_END, constants.CT_ANGLE_STEP)
    bins = constants.CT_BINS
    rebin_x = np.linspace(constants.CT_REBIN_RANGE_MIN, constants.CT_REBIN_RANGE_MAX, bins)
    re_x = np.zeros(bins)
    rebin_y = np.zeros(bins)
    bin_counter = np.zeros(bins)

    for angle in angles:
        img_rot = ndimage.rotate(z_summed, angle, reshape=False, mode='reflect')
        img_rot = -factor * img_rot

        noise_ind = [0, 1, 2, 3, -4, -3, -2, -1]
        line_int = np.sum(img_rot, axis=0) / img_rot.shape[0]
        x = np.arange(len(line_int), dtype='float')

        line_int = calculation_utils.linear_detrend(line_int, noise_ind)
        line_int = line_int - np.mean(line_int[noise_ind])
        line_int = line_int / np.max(line_int)

        mid = calculation_utils.find_center(line_int)

        x = x - mid
        argmax = np.argmax(line_int)
        x[argmax] = 0
        x = x * data[0].PixelSpacing[0]

        plt.title(f'LSF calculation for CT Filter kernel: {name}')
        plt.xlabel('distance in mm')
        plt.grid()
        plt.ylabel('counts (normalized)')
        plt.plot(x, line_int, label=f'{angle}\N{DEGREE SIGN}')
        plt.legend()

        bin_places = np.digitize(x, rebin_x)

        for i, bin_place in enumerate(bin_places):
            if bin_place != bins:
                re_x[bin_place] += x[i]
                rebin_y[bin_place] += line_int[i]
                bin_counter[bin_place] += 1

    for i, count in enumerate(bin_counter):
        if count != 0:
            rebin_y[i] /= count
            re_x[i] /= count
    rebin_x = re_x
    plt.savefig(IMSAVE_PATH + f'/LSF_CT_filter_{name}.png')
    plt.close()
    # plt.show()
    to_delete = []
    for i in range(len(rebin_y)):
        if np.abs(rebin_y[i]) < constants.REBIN_THRESHOLD:
            to_delete.append(i)

    rebin_y = np.delete(rebin_y, to_delete)
    rebin_x = np.delete(rebin_x, to_delete)

    noise_indices = constants.NOISE_INDICES_LSF
    rebin_y = rebin_y - np.mean(rebin_y[noise_indices])
    rebin_y = rebin_y / np.max(rebin_y)

    rebin_x = rebin_x - rebin_x[np.argmax(rebin_y)]

    function = calculation_utils.gauss
    pos_max = np.argmax(rebin_y)
    x_pred = np.arange(-pos_max, len(rebin_y) - pos_max, constants.PIXEL_SPACING_PRECISION) * data[0].PixelSpacing[0]
    p_opt, p_cov = curve_fit(function, rebin_x, rebin_y, p0=p0)
    y_pred = function(x_pred, *p_opt)
    y_pred = y_pred / np.max(y_pred)

    spacing = data[0].PixelSpacing[0]
    x_max = x_pred[np.argmax(y_pred)]
    x_pred = x_pred - x_max

    return x_pred, y_pred, rebin_x, rebin_y, spacing


def do_mtf_org(x: np.ndarray, y: np.ndarray, spacing: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Modulation Transfer Function from original data.

    Args:
        x (np.ndarray): X coordinates
        y (np.ndarray): Y values (LSF)
        spacing (float): Pixel spacing

    Returns:
        Tuple[np.ndarray, np.ndarray]: Frequency array and MTF values
    """
    if len(x) % 2 != 0:
        y = y[:-1]
        x = x[:-1]
    line = np.zeros(constants.MTF_ZERO_PAD_SIZE)
    line[constants.MTF_ZERO_PAD_CENTER - int(len(x) / 2):constants.MTF_ZERO_PAD_CENTER + int(len(x) / 2)] = y

    xf1, y1 = do_mtf(line, spacing)

    return xf1 * constants.MTF_MULTIPLICATION_FACTOR, y1


def do_mtf(line: np.ndarray, spacing: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate MTF using Fourier Transform.

    Args:
        line (np.ndarray): Line spread function
        spacing (float): Pixel spacing

    Returns:
        Tuple[np.ndarray, np.ndarray]: Frequency array and normalized MTF
    """
    yf = fft(line)
    x_f = np.arange(len(line)) * spacing
    n = len(x_f)
    xf1 = fftfreq(n, spacing)[:n // 2]
    y = np.abs(yf[0:n // 2])
    y1 = y / y[0]
    return xf1, y1


def do_mtf_fit(x: np.ndarray, y: np.ndarray, spacing: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate MTF from fitted data with higher resolution.

    Args:
        x (np.ndarray): X coordinates
        y (np.ndarray): Fitted Y values
        spacing (float): Pixel spacing

    Returns:
        Tuple[np.ndarray, np.ndarray]: Frequency array and MTF values (limited range)
    """
    spacing = spacing * constants.PIXEL_SPACING_PRECISION
    line = np.zeros(constants.MTF_FIT_ZERO_PAD_SIZE)
    line[constants.MTF_FIT_ZERO_PAD_CENTER - int(len(x) / 2):constants.MTF_FIT_ZERO_PAD_CENTER + int(len(x) / 2)] = y
    xf1, y1 = do_mtf(line, spacing)

    return xf1[0:constants.MTF_FIT_FREQUENCY_LIMIT] * constants.MTF_MULTIPLICATION_FACTOR, y1[
                                                                                           0:constants.MTF_FIT_FREQUENCY_LIMIT]


def fw_hm_org(x: np.ndarray, y: np.ndarray, argmax: Optional[int] = None) -> float:
    """
    Calculate Full Width at Half Maximum from original data.

    Args:
        x (np.ndarray): X coordinates
        y (np.ndarray): Y values
        argmax (Optional[int]): Index of maximum value

    Returns:
        float: FWHM value rounded to 3 decimal places
    """
    yminus = y - 0.5
    first_index, second_index = calculation_utils.find_intersection_with_zero(yminus, argmax)
    x1 = x[first_index]
    x2 = x[first_index + 1]
    x3 = x[second_index - 1]
    x4 = x[second_index]
    x_left = calculation_utils.zero_intersection(x1, yminus[first_index], x2, yminus[first_index + 1])
    x_right = calculation_utils.zero_intersection(x3, yminus[second_index - 1], x4, yminus[second_index])
    fw_hm = np.round(np.abs(x_left) + x_right, constants.FWHM_PRECISION)
    return fw_hm


def fw_hm_fit(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate Full Width at Half Maximum from fitted data.

    Args:
        x (np.ndarray): X coordinates
        y (np.ndarray): Fitted Y values

    Returns:
        float: FWHM value rounded to 3 decimal places
    """
    y1 = np.argmin(np.abs(y[:int(len(y) / 2)] - 0.5))
    y2 = np.argmin(np.abs(y[int(len(y) / 2):-1] - 0.5))
    fw_hm = np.round(np.abs(x[y1]) + x[y2 + int(len(y) / 2)], constants.FWHM_PRECISION)
    return fw_hm


def mtf_val_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Calculate MTF values at 50% and 10% from fitted data.

    Args:
        x (np.ndarray): Frequency coordinates
        y (np.ndarray): MTF values

    Returns:
        Tuple[float, float]: MTF50 and MTF10 values
    """
    x_50 = np.round(x[np.argmin(np.abs(y - 0.5))], constants.MTF_PRECISION)
    x_10 = np.round(x[np.argmin(np.abs(y - 0.1))], constants.MTF_PRECISION)
    return x_50, x_10


def mtf_val_org(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Calculate MTF values at 50% and 10% from original data using interpolation.

    Args:
        x (np.ndarray): Frequency coordinates
        y (np.ndarray): MTF values

    Returns:
        Tuple[float, float]: MTF50 and MTF10 values
    """
    index_50 = 0
    index_10 = 0
    y_50 = y - 0.5
    y_10 = y - 0.1
    for i in range(len(y)):
        if y_50[i] < 0:
            index_50 = i
            break
    for i in range(len(y)):
        if y_10[i] < 0:
            index_10 = i
            break
    x1 = x[index_50]
    x2 = x[index_50 + 1]
    x_50 = np.round(calculation_utils.zero_intersection(x1, y_50[index_50], x2, y_50[index_50 + 1]),
                    constants.MTF_PRECISION)
    x1 = x[index_10]
    x2 = x[index_10 + 1]
    x_10 = np.round(calculation_utils.zero_intersection(x1, y_10[index_10], x2, y_10[index_10 + 1]),
                    constants.MTF_PRECISION)
    return x_50, x_10
