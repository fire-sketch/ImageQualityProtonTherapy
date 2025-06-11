"""
Experiment Parameter Management System
=====================================
Manages experimental parameters and configurations for medical imaging analysis.
"""

from typing import List


class ExperimentParameter:
    """Base class for experiment parameters."""

    def __init__(self, name: str, path: str, factor: float):
        self.name = name
        self.path = path
        self.factor = factor


class Wire4mm(ExperimentParameter):
    """Parameters for 4mm wire experiments."""
    cor = []
    slices = []

    def __init__(self, name: str, path: str, factor: float):
        super().__init__(name, path, factor)


class ESF(ExperimentParameter):
    """Base class for ESF experiments."""

    def __init__(self, name: str, path: str, factor: float):
        super().__init__(name, path, factor)


class ESFGA(ESF):
    """ESF parameters for GA experiments."""
    cor_x = []
    cor_y = []
    slices = []

    def __init__(self, name: str, path: str, factor: float):
        super().__init__(name, path, factor)


class ESFFDG(ESF):
    """ESF parameters for FDG experiments."""
    cor_x = []
    cor_y = []
    slices = []

    def __init__(self, name: str, path: str, factor: float):
        super().__init__(name, path, factor)


class ESFFDG_PET(ESF):
    """ESF parameters for FDG PET experiments."""
    cor_x = []
    cor_y = []
    slices = []

    def __init__(self, name: str, path: str, factor: float):
        super().__init__(name, path, factor)


class CTExperimentData(ExperimentParameter):
    """Parameters for CT experiment data."""

    def __init__(self, name: str, path: str, factor: float, slices: List[int], cor: List[int]):
        super().__init__(name, path, factor)
        self.slices = slices
        self.cor = cor


class Noise:
    """Parameters for noise analysis."""
    slices = []
    cor = []

    def __init__(self, path: str, slices: List[int]):
        self.path = path
        self.slices = slices


def init_experiment_data() -> None:
    """Initialize experiment data coordinates and parameters."""
    Wire4mm.cor = [234, 210]
    ESFFDG.cor_x = [209, 225]
    ESFFDG.cor_y = [168, 186]
    ESFFDG_PET.cor_x = [209, 225]
    ESFFDG_PET.cor_y = [168, 186]
    Wire4mm.slices = [49, 123]
    ESFFDG.slices = [49, 128]
    ESFFDG_PET.slices = [49, 128]
    ESFGA.cor_x = [208, 223]
    ESFGA.cor_y = [164, 182]
    ESFGA.slices = [45, 123]
    Noise.cor = [250, 250]


def get_experiment_data(root: str, name: str, operation: str):
    """
    Get experiment data for ESF or LSF operations.

    Args:
        root (str): Root directory path
        name (str): Dataset name
        operation (str): Operation type ('ESF' or 'LSF')

    Returns:
        ExperimentParameter: Configured experiment parameters
    """
    factors = {
        'FDG_1': -1, 'FDG_2': 1, 'FDG_3': 1, 'FDG_4': 1, 'FDG_5': 1,
        'FDG_6': 1, 'FDG_7': 1, 'FDG_8': 1, 'FDG_9': 1, 'FDG_10': 1,
        'FDG_11': 1, 'FDG_12': 1, 'FDG_13': 1, 'FDG_14': 1, 'FDG_15': 1,
        'FDG_16': 1, 'FDG_17': 1, 'FDG_18': 1, 'FDG_19': 1, 'FDG_20': 1,
        'FDG_21': 1, 'FDG_22': 1, 'FDG_23': 1, 'FDG_24': 1, 'FDG_25': 1,
        'FDG_26': 1, 'FDG_27': 1, 'FDG_28': 1, 'FDG_29': 1, 'FDG_30': 1,
        'FDG_PET_1': -1, 'FDG_PET_2': -1, 'FDG_PET_3': -1, 'FDG_PET_4': -1,
        'FDG_PET_5': -1, 'FDG_PET_6': -1, 'FDG_PET_7': -1, 'FDG_PET_8': -1,
        'FDG_PET_9': -1, 'FDG_PET_10': -1, 'FDG_PET_11': -1, 'FDG_PET_12': -1,
        'FDG_PET_13': -1, 'FDG_PET_14': -1, 'FDG_PET_15': -1, 'FDG_PET_16': -1,
        'FDG_PET_17': -1, 'FDG_PET_18': -1, 'FDG_PET_19': -1, 'FDG_PET_20': -1,
        'FDG_PET_21': -1, 'FDG_PET_22': -1, 'FDG_PET_23': -1, 'FDG_PET_24': -1,
        'FDG_PET_25': -1, 'FDG_PET_26': -1, 'FDG_PET_27': -1, 'FDG_PET_28': -1,
        'FDG_PET_29': -1, 'FDG_PET_30': -1,
        'GA_1': -1, 'GA_2': 1, 'GA_3': 1, 'GA_4': 1, 'GA_5': 1,
        'GA_6': 1, 'GA_7': 1, 'GA_8': 1, 'GA_9': 1, 'GA_10': 1,
        'GA_11': 1, 'GA_12': 1, 'GA_13': 1, 'GA_14': 1, 'GA_15': 1,
        'GA_16': 1, 'GA_17': 1, 'GA_18': 1, 'GA_19': 1, 'GA_20': 1,
        'GA_21': 1, 'GA_22': 1, 'GA_23': 1, 'GA_24': 1, 'GA_25': 1,
        'GA_26': 1, 'GA_27': 1, 'GA_28': 1, 'GA_29': 1, 'GA_30': 1,
        'GA_PET_1': -1, 'GA_PET_2': -1, 'GA_PET_3': -1, 'GA_PET_4': -1,
        'GA_PET_5': -1, 'GA_PET_6': -1, 'GA_PET_7': -1, 'GA_PET_8': -1,
        'GA_PET_9': -1, 'GA_PET_10': -1, 'GA_PET_11': -1, 'GA_PET_12': -1,
        'GA_PET_13': -1, 'GA_PET_14': -1, 'GA_PET_15': -1, 'GA_PET_16': -1,
        'GA_PET_17': -1, 'GA_PET_18': -1, 'GA_PET_19': -1, 'GA_PET_20': -1,
        'GA_PET_21': -1, 'GA_PET_22': -1, 'GA_PET_23': -1, 'GA_PET_24': -1,
        'GA_PET_25': -1, 'GA_PET_26': -1, 'GA_PET_27': -1, 'GA_PET_28': -1,
        'GA_PET_29': -1, 'GA_PET_30': -1,
    }

    path = root + '/' + name
    print(name)
    if operation == 'LSF':
        return Wire4mm(name, path, factors[name[:]])
    elif operation == 'ESF':
        if 'GA_PET' in name:
            return ESFGA(name, path, factors[name])
        if 'GA' in name:
            return ESFGA(name, path, factors[name[:]])
        elif 'FDG_PET' in name:
            return ESFFDG_PET(name, path, factors[name])
        elif 'FDG' in name:
            return ESFFDG(name, path, factors[name[:]])


def get_experiment_data_ct(root: str, name: str) -> CTExperimentData:
    """
    Get experiment data for CT operations.

    Args:
        root (str): Root directory path
        name (str): Dataset name

    Returns:
        CTExperimentData: CT experiment parameters
    """
    names = ['ub_039', 'b_039', 'ub_068', 'ub_098', 'b_097', 'b_117']
    slices = [[101, 131], [20, 50], [141, 170], [91, 119], [21, 48], [24, 52]]
    cor = [[140, 320], [138, 320], [190, 287], [210, 277], [198, 283], [217, 274]]
    index = names.index(name)
    path = root + '/' + name
    return CTExperimentData(name, path, 1, slices[index], cor[index])


def get_noise_data(root: str, name: str) -> Noise:
    """
    Get noise analysis parameters.

    Args:
        root (str): Root directory path
        name (str): Dataset name

    Returns:
        Noise: Noise analysis parameters
    """
    names = ['ub_039', 'b_039', 'ub_068', 'ub_098', 'b_097', 'b_117']
    slices = [[69, 97], [56, 81], [108, 134], [58, 82], [57, 80], [60, 84]]
    path = root + '/' + name
    if name in names:
        index = names.index(name)
        s = slices[index]
    elif 'FDG' in name:
        s = ESFFDG.slices
    else:
        s = ESFGA.slices
    return Noise(path, s)