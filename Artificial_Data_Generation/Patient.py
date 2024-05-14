from glob import glob
import os
from pathlib import Path
import pydicom
from Artificial_Data_Generation.CTData import CT3D, CTLayer


class Patient:
    def __init__(self, patient_id, path):
        self.id = patient_id
        self.path = os.path.join(path, patient_id)
        self.ct_3d = self.__set_ct3d()
        self.width = None
        self.noise_sigma = None

    def __set_ct3d(self):
        ct_paths = glob(self.path + '/CT*')
        names = [os.path.splitext(Path(os.path.basename(ct_path)))[0] for ct_path in ct_paths]
        cts = [CTLayer(pydicom.read_file(ct_path), name) for ct_path, name in zip(ct_paths, names)]
        print(cts[0].dicom_header)
        return CT3D(cts)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.id == other
        return False

    def write_modified_as_png(self, data_path="../../data/output_data/png", mods='gaussian', numbered=True,
                              safe_original=True, data_path_original="../../data/output_data/png_original",
                              center=None, width=None):
        data_path = f'{data_path}/{mods}/{self.id}w{self.width}'
        data_path_original = f'{data_path_original}/{self.id}'
        self.ct_3d.write_modified_as_png(data_path, numbered, safe_original, data_path_original, center, width)
        print(f'finished saving files for Patient {self.id}')

    def write_modified_as_dicom(self, mods='w', data_path="../data/output_data/dicom", number=0):
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
    def convolve_with_filter(self, width=1, filter_type='gaussian', mode="reflect"):
        print('start convolution with width ' + str(width) + ' type ' + filter_type)
        self.width = width
        self.ct_3d.convolve_3dct_with_filter(width, filter_type, mode)
        print('end convolution')

    def add_noise(self, sigma, mean=0):
        self.noise_sigma = sigma
        self.ct_3d.add_gaussian_white_noise(sigma, mean)


class PatientDataBase:
    def __init__(self, path):
        self.__data_path = path
        self.patient_ids = [f for f in os.listdir(path) if
                            os.path.isdir(os.path.join(path, f))]
        self.patients = []
        self.number_of_patients = len(self.patient_ids)

    def __initialize_patient(self, patient_id):
        if patient_id not in self.patients:

            patient = Patient(patient_id, self.__data_path)
            self.patients.append(patient)
        else:
            patient = self.patients[self.patients.index(patient_id)]
        return patient

    def patient_generator(self, *args):  # args[0] number of how many patients you want
        n = self.number_of_patients
        if args:
            n = args[0]
        for patient_id in self.patient_ids[:n]:
            yield self.__initialize_patient(patient_id)

    def get_patient_from_id(self, patient_id):
        return self.__initialize_patient(patient_id)
