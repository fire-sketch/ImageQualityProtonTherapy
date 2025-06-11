import os
from glob import glob
from pathlib import Path
from typing import Any, Generator

import pydicom

from Artificial_Data_Generation.CTData import CT3D, CTLayer


class Patient:
    """
    Represents a single patient with associated CT scan data
    Attributes:
        patient_id (str): Unique patient identifier
        data_path (Path): Path to patients's DICOM files
        ct_3d (CT3D): 3D CT volume data
        width (float): Currently applied filter width
        noise_sigma (float) Currently applied noise level
    """

    def __init__(self, patient_id: str, path: str) -> None:
        """
        Initialize patient with CT scan data.

        :param patient_id: unique identifier for the patient
        :param path: base directory containing the image data
        """
        self.id = patient_id
        self.path = os.path.join(path, patient_id)
        self.ct_3d = self.__set_ct3d()
        self.width = None
        self.noise_sigma = None

    def __set_ct3d(self) -> CT3D:
        """
        Load CT DICOM files and create CT3D.

        :return: 3DCT volume containing all slices

        :raises: FileNotFoundError
        """
        ct_paths = glob(self.path + '/CT*')
        if not ct_paths:
            raise FileNotFoundError(f'No CT files found in {self.path}')
        names = [os.path.splitext(Path(os.path.basename(ct_path)))[0] for ct_path in ct_paths]
        cts = [CTLayer(pydicom.read_file(ct_path), name) for ct_path, name in zip(ct_paths, names)]
        print(cts[0].dicom_header)
        return CT3D(cts)

    def __eq__(self, other) -> bool:
        """
        Enable comparision with string patient_id
        :param other: the other patient to compare with
        :return: if patient ids are the same or not
        """
        if isinstance(other, str):
            return self.id == other
        return False

    def write_modified_as_png(self, data_path="../data/output_data/png", mods='gaussian',
                              numbered=True, safe_original=True, data_path_original="../data/output_data/png_original",
                              center=None, width=None) -> None:
        """
        Save CT slices as PNG images with windowing options
        :param data_path: base output directory for processed images
        :param mods: processing mode identifier [gaussian, rectangle, noise, noise_gauss]
        :param numbered: add slice numbers to filenames
        :param safe_original: also save original unprocessed images
        :param data_path_original: path for original images
        :param center: window center (HU)
        :param width: window width (HU)
        """
        data_path = f'{data_path}/{mods}/{self.id}w{self.width}'
        data_path_original = f'{data_path_original}/{self.id}'
        self.ct_3d.write_modified_as_png(data_path, numbered, safe_original, data_path_original, center, width)
        print(f'finished saving files for Patient {self.id}')

    def write_modified_as_dicom(self, mods='w', data_path="../data/output_data/dicom", number=0) -> None:
        """
        Save processed 3D CT as DICOM files
        :param mods: Processing mode identifer [gaussian, rectangle, noise, noise_gauss]
        :param data_path: base output directory
        :param number: file number for batch processing
        :return:
        """
        os.makedirs(f'{data_path}/{self.id}', exist_ok=True)
        if mods == 'gaussian' or mods == 'rectangle':
            data_path = f'{data_path}/{self.id}/{self.id}w{self.width}'
        if mods == 'noise':
            data_path = f'{data_path}/{self.id}/{self.id}n{self.noise_sigma}_{number}'
        if mods == 'noise_gauss':
            data_path = f'{data_path}/{self.id}/{self.id}w{self.width}n{self.noise_sigma}'
        self.ct_3d.write_modified_as_dicom(data_path)
        print(f'finished saving files for Patient {self.id}')

    # sigma in mm in real world
    def convolve_with_filter(self, width=1, filter_type='gaussian', mode="reflect") -> None:
        """
        Apply spatial filtering to CT.

        :param width: filter width in mm
        :param filter_type: filter type ('gaussian', 'rectangle', 'triangle')
        :param mode: boundary condition handling mode
        :return:
        """
        print('start convolution with width ' + str(width) + ' type ' + filter_type)
        self.width = width
        self.ct_3d.convolve_3dct_with_filter(width, filter_type, mode)
        print('end convolution')

    def add_noise(self, sigma, mean=0) -> None:
        """
        Add gaussian white noise to CT

        :param sigma: standard deviation of noise
        :param mean: mean value of noise (default: 0)
        :return:
        """
        self.noise_sigma = sigma
        self.ct_3d.add_gaussian_white_noise(sigma, mean)


class PatientDataBase:
    """
    Manages a database of patients with CT data.

    Provides efficient access to patient data and batch processing capabilities.
    """

    def __init__(self, path) -> None:
        """
        Initialize patient database.
        :param path: Path to directory containing patient folders
        """

        self.__data_path = path
        self.patient_ids = [f for f in os.listdir(path) if
                            os.path.isdir(os.path.join(path, f))]
        self.patients = []
        self.number_of_patients = len(self.patient_ids)

    def __initialize_patient(self, patient_id) -> Patient:
        """
        Initialize or retrieve a patient from cache

        :param patient_id: patient identifier
        :return: patient object
        """
        if patient_id not in self.patients:

            patient = Patient(patient_id, self.__data_path)
            self.patients.append(patient)
        else:
            patient = self.patients[self.patients.index(patient_id)]
        return patient

    def patient_generator(self, *args) -> Generator[Patient, Any, None]:
        """
        Generate patient objects for batch processing.

        :param args: Optional argument for number of patients to process
        :yields: Patient objects
        """
        n = self.number_of_patients
        if args:
            n = args[0]
        for patient_id in self.patient_ids[:n]:
            yield self.__initialize_patient(patient_id)

    def get_patient_from_id(self, patient_id) -> Patient:
        """
        Get a specific patient by ID

        :param patient_id: patient identifier
        :return:  patient object
        """
        return self.__initialize_patient(patient_id)
