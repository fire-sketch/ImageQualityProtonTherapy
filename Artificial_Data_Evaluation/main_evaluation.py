"""
Main Evaluation Script
======================
"""

import re
from pathlib import Path
from typing import List, Tuple

import patient_evaluation as PatientEvaluation
import evaluation_utils as utils


def get_patient_selection() -> Tuple[List[Path], List[str]]:
    """
    Get patient selection from user input and filter paths.

    Returns:
        Tuple[List[Path], List[str]]: Selected patient paths and names
    """
    # Configuration
    modification_type = 'gaussian'  # gaussian, rect, noise, both, gauss_noise

    folder_selected = utils.DATA_PATHS[modification_type]

    # Get all patient directories
    paths = [f for f in folder_selected.iterdir() if f.is_dir()]
    patient_names = [p.name for p in paths]
    print(patient_names)

    # Get user selection
    values = input("Enter indices of patient to evaluate comma separated:\n")
    values = values.split(',')
    values = [x.zfill(2) for x in values]

    # Filter based on selection
    selected_paths = []
    selected_patients = []

    for path, patient in zip(paths, patient_names):
        patient_match = re.search(utils.REGEX_PATTERNS['patient_number'], patient)
        if patient_match and patient_match[0] in values:
            selected_paths.append(path)
            selected_patients.append(patient)

    return selected_paths, selected_patients


def main() -> None:
    """Main execution function for patient evaluation."""
    # Configuration parameters
    do_overlay = 1
    overlay_widths = [4, 7, 16]
    modification_type = 'gaussian'
    overlay_ground_truth = 'original'

    # Get patient selection
    selected_paths, selected_patients = get_patient_selection()

    # Process each selected patient
    for path, patient in zip(selected_paths, selected_patients):
        PatientEvaluation.progress(
            do_overlay,
            str(path),
            patient,
            modification_type,
            overlay_widths=overlay_widths,
            overlay_ground=overlay_ground_truth
        )


if __name__ == '__main__':
    main()