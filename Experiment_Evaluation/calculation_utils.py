"""
Calculation Utility Functions
"""

import glob
import math
from typing import Tuple, List, Optional, Union

import numpy as np
import pydicom
from sklearn.linear_model import LinearRegression

import Lookup_Data
import constants


def gauss(x: np.ndarray, *p: Tuple[float, ...]) -> np.ndarray:
    """
    Gaussian function for curve fitting.

    Args:
        x (np.ndarray): Input x values
        *p (Tuple[float, ...]): Parameters (a1, a2, a3, a4, a5)

    Returns:
        np.ndarray: Gaussian function values
    """
    a1, a2, a3, a4, a5 = p
    return a2 ** 2 * np.exp(-0.5 * ((x - a1) / a3) ** 2)


def pre_init() -> None:
    """Initialize experiment data lookup tables."""
    Lookup_Data.init_experiment_data()


def init(root: str, name: str, operation: str) -> Union[
    Tuple[np.ndarray, List[int], List[int], List, float],
    Tuple[np.ndarray, List[int], List, float]
]:
    """
    Initialize data loading for different analysis operations.

    Args:
        root (str): Root directory path
        name (str): Dataset/patient name
        operation (str): Type of operation ('ESF', 'LSF', 'CT')

    Returns:
        Union[Tuple[np.ndarray, List[int], List[int], List, float],
              Tuple[np.ndarray, List[int], List, float]]: Loaded data and parameters based on operation type
    """
    params = None
    cor_x = None
    cor_y = None
    cor = None
    if operation == 'ESF':
        params = Lookup_Data.get_experiment_data(root, name, operation)
        cor_x = params.cor_x
        cor_y = params.cor_y
    elif operation == 'LSF' or operation == 'CT':
        params = Lookup_Data.get_experiment_data_ct(root, name)
        cor = params.cor

    slices = params.slices
    path = params.path
    factor = params.factor

    files = sorted(glob.glob(path + '/*.dcm'), key=len)
    data = []

    for file in files:
        data.append(pydicom.read_file(file))

    slices = data[slices[0]:slices[1]]
    s = np.array([s.pixel_array for s in slices])

    if operation == 'ESF':
        return s, cor_x, cor_y, data, factor
    else:
        return s, cor, data, factor


def zero_intersection(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate zero intersection point using linear interpolation.

    Args:
        x1, y1 (float): First point coordinates
        x2, y2 (float): Second point coordinates

    Returns:
        float: X-coordinate of zero intersection
    """
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1  # y!=0  y=mx+c
    x = -c / m
    return x


def calc_area_left(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate area under curve on the left side using trapezoidal integration.

    Args:
        x (np.ndarray): X coordinates
        y (np.ndarray): Y coordinates

    Returns:
        float: Calculated area
    """
    area = 0
    for i in range(1, len(y)):
        area += 0.5 * (y[i] - y[i - 1]) * (x[i] - x[i - 1])
        if i > 1:
            area += (x[i] - x[i - 1]) * y[i - 1]
    return area


def calc_area_right(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate area under curve on the right side using trapezoidal integration.

    Args:
        x (np.ndarray): X coordinates
        y (np.ndarray): Y coordinates

    Returns:
        float: Calculated area
    """
    area = 0
    for i in range(len(y) - 2, -1, -1):
        area += 0.5 * (y[i] - y[i + 1]) * (x[i + 1] - x[i])
        if i < len(y) - 2:
            area += (x[i + 1] - x[i]) * y[i + 1]
    return area


def linear_detrend(y: np.ndarray, noise_ind: List[int]) -> np.ndarray:
    """
    Remove linear trend from data using specified noise indices.

    Args:
        y (np.ndarray): Input data
        noise_ind (List[int]): Indices representing noise regions

    Returns:
        np.ndarray: Detrended data
    """
    x = np.arange(0, len(y)).reshape((-1, 1))
    trend_y = y[noise_ind]
    trend_x = x[noise_ind]
    model = LinearRegression()
    model.fit(trend_x, trend_y)
    for i, val in enumerate(y):
        y[i] = y[i] - (model.coef_ * x[i] + model.intercept_)
    return y


def calc_missing_area(left_x: np.ndarray, left_y: np.ndarray, right_x: np.ndarray, right_y: np.ndarray,
                      left: float, right: float) -> float:
    """
    Calculate missing area between left and right curves for center finding.

    Args:
        left_x, left_y (np.ndarray): Left curve coordinates
        right_x, right_y (np.ndarray): Right curve coordinates
        left, right (float): Search boundaries

    Returns:
        float: X-coordinate that minimizes area difference
    """
    search_points = np.linspace(left, right, constants.AREA_SEARCH_POINTS)
    search_points = search_points[1:-1]
    min_area = constants.MIN_AREA_THRESHOLD
    x_mid = 0
    for p in search_points:
        a_left_calc = calc_area_left(np.append(left_x, p), np.append(left_y, 1))
        a_right_calc = calc_area_right(np.append(p, right_x), np.append(1, right_y))
        if np.abs(a_left_calc - a_right_calc) < min_area:
            min_area = np.abs(a_left_calc - a_right_calc)
            x_mid = p
    return x_mid


def find_intersection_with_zero(y: np.ndarray, argmax: Optional[int] = None) -> Tuple[int, int]:
    """
    Find indices where function intersects with zero level.

    Args:
        y (np.ndarray): Function values
        argmax (Optional[int]): Index of maximum value

    Returns:
        Tuple[int, int]: First and second intersection indices
    """
    size = len(y) // 2
    if argmax:
        size = argmax
    y_left = y[0:size]
    y_right = y[size:-1]
    first_index = 0
    second_index = 0
    for i in range(0, len(y_left) - 1):
        if y_left[i] < 0:
            first_index = i

    for j in range(len(y_right) - 1, 0, -1):
        if y_right[j] < 0:
            second_index = j + size
    if first_index >= second_index:
        first_index = 0
    if second_index <= first_index:
        second_index = size - 1

    return first_index, second_index


def find_center(y: np.ndarray) -> float:
    """
    Find center position of a peak using area balance method.

    Args:
        y (np.ndarray): Function values

    Returns:
        float: Center position
    """
    highest = np.argmax(y)
    left = highest - 1
    right = highest + 1
    first_index, second_index = find_intersection_with_zero(y)

    zero_left = zero_intersection(first_index, y[first_index], first_index + 1, y[first_index + 1])
    zero_right = zero_intersection(second_index - 1, y[second_index - 1], second_index, y[second_index])

    left_x = np.append(np.array([zero_left]), np.arange(first_index + 1, highest))
    left_y = np.append(np.array(0), y[first_index + 1:highest])

    right_x = np.append(np.arange(highest + 1, second_index), np.array([zero_right]))
    right_y = np.append(y[highest + 1:second_index], np.array(0))

    return calc_missing_area(left_x, left_y, right_x, right_y, left, right)


def normpdf(x: float, mean: float, sd: float) -> float:
    """
    Calculate normal probability density function value.

    Args:
        x (float): Input value
        mean (float): Distribution mean
        sd (float): Standard deviation

    Returns:
        float: PDF value
    """
    var = float(sd) ** 2
    denom = (2 * math.pi * var) ** .5
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom


def gauss(x: np.ndarray, *p: Tuple[float, ...]) -> np.ndarray:
    """
    Gaussian function for curve fitting.

    Args:
        x (np.ndarray): Input x values
        *p (Tuple[float, ...]): Parameters (a1, a2, a3, a4, a5)

    Returns:
        np.ndarray: Gaussian function values
    """
    a1, a2, a3, a4, a5 = p
    return a2 ** 2 * np.exp(-0.5 * ((x - a1) / a3) ** 2)


def pre_init() -> None:
    """Initialize experiment data lookup tables."""
    Lookup_Data.init_experiment_data()


def init(root: str, name: str, operation: str) -> Tuple:
    """
    Initialize data loading for different analysis operations.

    Args:
        root (str): Root directory path
        name (str): Dataset/patient name
        operation (str): Type of operation ('ESF', 'LSF', 'CT')

    Returns:
        Tuple: Loaded data and parameters based on operation type
    """
    params = None
    cor_x = None
    cor_y = None
    cor = None
    if operation == 'ESF':
        params = Lookup_Data.get_experiment_data(root, name, operation)
        cor_x = params.cor_x
        cor_y = params.cor_y
    elif operation == 'LSF' or operation == 'CT':
        params = Lookup_Data.get_experiment_data_ct(root, name)
        cor = params.cor

    slices = params.slices
    path = params.path
    factor = params.factor

    files = sorted(glob.glob(path + '/*.dcm'), key=len)
    data = []

    for file in files:
        data.append(pydicom.read_file(file))

    slices = data[slices[0]:slices[1]]
    s = np.array([s.pixel_array for s in slices])

    if operation == 'ESF':
        return s, cor_x, cor_y, data, factor
    else:
        return s, cor, data, factor


def zero_intersection(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate zero intersection point using linear interpolation.

    Args:
        x1, y1 (float): First point coordinates
        x2, y2 (float): Second point coordinates

    Returns:
        float: X-coordinate of zero intersection
    """
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1  # y!=0  y=mx+c
    x = -c / m
    return x


def calc_area_left(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate area under curve on the left side using trapezoidal integration.

    Args:
        x (np.ndarray): X coordinates
        y (np.ndarray): Y coordinates

    Returns:
        float: Calculated area
    """
    area = 0
    for i in range(1, len(y)):
        area += 0.5 * (y[i] - y[i - 1]) * (x[i] - x[i - 1])
        if i > 1:
            area += (x[i] - x[i - 1]) * y[i - 1]
    return area


def calc_area_right(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate area under curve on the right side using trapezoidal integration.

    Args:
        x (np.ndarray): X coordinates
        y (np.ndarray): Y coordinates

    Returns:
        float: Calculated area
    """
    area = 0
    for i in range(len(y) - 2, -1, -1):
        area += 0.5 * (y[i] - y[i + 1]) * (x[i + 1] - x[i])
        if i < len(y) - 2:
            area += (x[i + 1] - x[i]) * y[i + 1]
    return area


def linear_detrend(y: np.ndarray, noise_ind: List[int]) -> np.ndarray:
    """
    Remove linear trend from data using specified noise indices.

    Args:
        y (np.ndarray): Input data
        noise_ind (List[int]): Indices representing noise regions

    Returns:
        np.ndarray: Detrended data
    """
    x = np.arange(0, len(y)).reshape((-1, 1))
    trend_y = y[noise_ind]
    trend_x = x[noise_ind]
    model = LinearRegression()
    model.fit(trend_x, trend_y)
    for i, val in enumerate(y):
        y[i] = y[i] - (model.coef_ * x[i] + model.intercept_)
    return y


def calc_missing_area(left_x: np.ndarray, left_y: np.ndarray, right_x: np.ndarray, right_y: np.ndarray, left: float,
                      right: float) -> float:
    """
    Calculate missing area between left and right curves for center finding.

    Args:
        left_x, left_y (np.ndarray): Left curve coordinates
        right_x, right_y (np.ndarray): Right curve coordinates
        left, right (float): Search boundaries

    Returns:
        float: X-coordinate that minimizes area difference
    """
    search_points = np.linspace(left, right, 10)
    search_points = search_points[1:-1]
    min_area = 100
    x_mid = 0
    for p in search_points:
        a_left_calc = calc_area_left(np.append(left_x, p), np.append(left_y, 1))
        a_right_calc = calc_area_right(np.append(p, right_x), np.append(1, right_y))
        if np.abs(a_left_calc - a_right_calc) < min_area:
            min_area = np.abs(a_left_calc - a_right_calc)
            x_mid = p
    return x_mid


def find_intersection_with_zero(y: np.ndarray, argmax: Optional[int] = None) -> Tuple[int, int]:
    """
    Find indices where function intersects with zero level.

    Args:
        y (np.ndarray): Function values
        argmax (Optional[int]): Index of maximum value

    Returns:
        Tuple[int, int]: First and second intersection indices
    """
    size = len(y) // 2
    if argmax:
        size = argmax
    y_left = y[0:size]
    y_right = y[size:-1]
    first_index = 0
    second_index = 0
    for i in range(0, len(y_left) - 1):
        if y_left[i] < 0:
            first_index = i

    for j in range(len(y_right) - 1, 0, -1):
        if y_right[j] < 0:
            second_index = j + size
    if first_index >= second_index:
        first_index = 0
    if second_index <= first_index:
        second_index = size - 1

    return first_index, second_index


def find_center(y: np.ndarray) -> float:
    """
    Find center position of a peak using area balance method.

    Args:
        y (np.ndarray): Function values

    Returns:
        float: Center position
    """
    highest = np.argmax(y)
    left = highest - 1
    right = highest + 1
    first_index, second_index = find_intersection_with_zero(y)

    zero_left = zero_intersection(first_index, y[first_index], first_index + 1, y[first_index + 1])
    zero_right = zero_intersection(second_index - 1, y[second_index - 1], second_index, y[second_index])

    left_x = np.append(np.array([zero_left]), np.arange(first_index + 1, highest))
    left_y = np.append(np.array(0), y[first_index + 1:highest])

    right_x = np.append(np.arange(highest + 1, second_index), np.array([zero_right]))
    right_y = np.append(y[highest + 1:second_index], np.array(0))

    return calc_missing_area(left_x, left_y, right_x, right_y, left, right)


def normpdf(x: float, mean: float, sd: float) -> float:
    """
    Calculate normal probability density function value.

    Args:
        x (float): Input value
        mean (float): Distribution mean
        sd (float): Standard deviation

    Returns:
        float: PDF value
    """
    var = float(sd) ** 2
    denom = (2 * math.pi * var) ** .5
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom
