"""
Core classes for handling 3D CT data and individual CT slices.
Provides functionality for DICOM processing, spatial filtering and data export.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
import pydicom
from matplotlib import pyplot as plt
from numpy.random import default_rng
from pydicom.pixel_data_handlers.util import apply_modality_lut
from pydicom.uid import generate_uid
from scipy.ndimage import zoom

import Filter


class CT3D:
    """
    Represents a 3D CT volume composed of multiple CT layers.
    Provides functionality for 3D-filtering, noise addition and export operations.
    """

    def __init__(self, ct_layers):
        """
        Initialize CT3D with a list of CT layers.

        :param ct_layers: List of CTLayer objects representing individual slices
        """
        self.ct_layers = ct_layers

    def write_modified_as_png(self, data_path, numbered, save_original, data_path_original, center, width) -> None:
        """
        Export modified CT data as png images with proper windowing
        :param data_path: output directory for modified imagaes
        :param numbered: whether to use numbered filenames
        :param save_original: whether to save original images alongside modified
        :param data_path_original: directory for original images
        :param center: window center (HU)
        :param width: window width (HU)
        """
        print('Start saving pngs')
        shutil.rmtree(data_path, ignore_errors=True)
        Path(data_path).mkdir(parents=True, exist_ok=True)
        pix_org = self.get_hu_3d_pixel_array()
        pix_mod = self.get_hu_3d_pixel_array_modified()
        zoom_x = self.ct_layers[0].dicom_header.PixelSpacing[0]
        zoom_y = self.ct_layers[0].dicom_header.PixelSpacing[1]
        zoom_z = self.ct_layers[0].dicom_header.SliceThickness
        out_org = zoom(pix_org, (zoom_x, zoom_y, zoom_z))
        out_mod = zoom(pix_mod, (zoom_x, zoom_y, zoom_z))
        if save_original:
            Path(data_path_original).mkdir(parents=True, exist_ok=True)
        if numbered:
            for i in range(out_org.shape[2]):
                filename = f'{data_path}/{i}.png'
                filename_original = f'{data_path_original}/{i}.png'
                self.ct_layers[0].write_modified_as_png(out_org[:, :, i], out_mod[:, :, i], filename, numbered,
                                                        save_original,
                                                        filename_original, center, width)
            print(f'Saved modified png files to {data_path}, saving original to {data_path_original}')
        else:
            for ct in self.ct_layers:
                ct.write_modified_as_png(out_org, data_path, numbered, save_original, data_path_original)
            print(f'Saved modified png files to {data_path}')

    def write_modified_as_dicom(self, data_path) -> None:
        """
        Export modified CT data as dicom

        :param data_path: output directory
        """
        print(f'Saving Dicom files to {data_path}')
        shutil.rmtree(data_path, ignore_errors=True)
        Path(data_path).mkdir(parents=True, exist_ok=True)
        media_uid = pydicom.uid.generate_uid(prefix="1.2.752.243.1.1.")
        series_uid = pydicom.uid.generate_uid(prefix="1.2.752.243.1.1.")
        frame_uid = pydicom.uid.generate_uid(prefix="1.2.752.243.1.1.")
        instance_uid = [generate_uid(prefix="1.2.752.243.1.1.") for _ in self.ct_layers]
        instance_uid.sort()
        for ct, ins_uid in zip(self.ct_layers, instance_uid):
            ct.write_modified_as_dicom(data_path)

    def convolve_3dct_with_filter(self, sigma, filter_type, mode) -> None:
        """
        Apply 3D spatial filtering to degrade image resolution.

        :param sigma: filter width in mm
        :param filter_type: type of filter ('gaussian', 'triangle', 'rectangle')
        :param mode: boundary condition for filtering
        """
        spacings = self.ct_layers[0].get_std_in_pixel()  # assume same values in all ct layers
        pix3, maxi = self.get_3d_pixel_array()
        if filter_type == 'gaussian':
            mod_pix = Filter.gaussian_filter_3d(pix3, sigma, spacings, mode)
        elif filter_type == 'triangle':
            mod_pix = Filter.triangle_filter_3d(pix3, sigma, spacings, mode)
        elif filter_type == 'rectangle':
            mod_pix = Filter.rectangle_filter_3d(pix3, sigma, spacings, mode)
        else:
            raise ValueError('Mode not supported')
        for i, ct_layer in enumerate(self.ct_layers):
            ct_layer.set_modified_pixel_array(mod_pix[:, :, i])

    def get_3d_pixel_array(self) -> (np.ndarray, float):
        """
        Construct 3D pixel array from individual CT layers.

        :return: Tuple of (3D pixel array, maximum value of array)
        """
        shape_2d = self.ct_layers[0].pixel_array.shape
        pixel_3d = np.zeros((shape_2d[0], shape_2d[1], len(self.ct_layers)), dtype=self.ct_layers[0].pixel_array.dtype)
        for i, ct in enumerate(self.ct_layers):
            pixel_3d[:, :, i] = ct.pixel_array
        maxi = np.max(pixel_3d)

        pixel_3d = pixel_3d.astype(np.float64)

        return pixel_3d, maxi

    def get_hu_3d_pixel_array(self) -> np.ndarray:
        """
        Get  3D Pixel array in HU values.

        :return: 3D Pixel Array (HU)
        """
        shape_2d = self.ct_layers[0].pixel_array.shape
        hu_3d = np.zeros((shape_2d[0], shape_2d[1], len(self.ct_layers)), dtype=np.float64)
        for i, ct in enumerate(self.ct_layers):
            hu_3d[:, :, i] = ct.get_hu_2d()
        return hu_3d

    def get_hu_3d_pixel_array_modified(self) -> np.ndarray:
        """
        Get the modified (filtered and or added noise) pixel array in HU values.

        :return: Modified pixel array (HU)
        """
        shape_2d = self.ct_layers[0].pixel_array.shape
        hu_3d = np.zeros((shape_2d[0], shape_2d[1], len(self.ct_layers)), dtype=np.float64)
        for i, ct in enumerate(self.ct_layers):
            hu_3d[:, :, i] = ct.get_hu_2d_modified()
        return hu_3d

    def add_gaussian_white_noise(self, sigma, mean=0) -> None:
        """
        Add gaussian white noise.

        :param sigma: standard deviation of noise in HU
        :param mean: mean of noise distribution (default: 0)
        """
        rng = default_rng()
        for ct in self.ct_layers:
            ct.add_gaussian_white_noise(sigma, mean, rng)


class CTLayer:
    """
    Represents a single CT slice with DICOM metadata and pixel data.

    Handles conversions between raw pixel values and HU values and noise addition.
    """

    def __init__(self, dicom, filename) -> None:
        """
        Initialize CT Layer from DICOM dataset

        :param dicom: pydicom Dataset containing CT slice
        :param filename:Associated filename
        """
        self.pixel_array = dicom.pixel_array  # uint16
        self.dicom_header = dicom
        self.filename = filename
        self.__modified_pixel_array = None  # float64

    def set_modified_pixel_array(self, modified_pixel_array) -> None:
        """
        Set the modified pixel array after filtering operations.

        :param modified_pixel_array: the modified pixel array
        """
        self.__modified_pixel_array = modified_pixel_array

    def get_modified_pixel_array(self) -> np.ndarray:
        """
        Get the modifed pixel array.

        :return: The modified pixel array
        """
        if self.__modified_pixel_array is None:
            raise ValueError('No modified pixel data calculated')
        return self.__modified_pixel_array

    def get_hu_2d(self) -> np.ndarray:
        """
        Convert raw pixel values array to HU.

        :return: 2D array in HU values
        """
        img = apply_modality_lut(self.pixel_array, self.dicom_header)
        return img

    def get_hu_2d_modified(self) -> np.ndarray:
        """
        Convert modified pixel array values to HU.

        :return: 2D array of modified data in HU
        """
        mod_pixel = self.__modified_pixel_array
        mod_pixel[mod_pixel < 0] = 0
        mod_pixel = np.round(mod_pixel).astype(np.uint16)
        img = apply_modality_lut(mod_pixel, self.dicom_header)
        return img

    def get_window_ct(self, ct, center, width) -> np.ndarray:
        """
        Apply windowing to CT data for displaying.

        :param ct: input CT data in HU
        :param center: window center (HU)
        :param width: window width (HU)
        :return: windowed image data scaled to 0-255 (uint8)
        """
        img_hu = ct
        img_min = center - width // 2
        img_max = center + width // 2
        # plt.hist(img_hu.flatten(),bins=50)
        # plt.show()
        img_hu[img_hu < img_min] = img_min
        img_hu[img_hu > img_max] = img_max
        img_hu = (img_hu - img_min) / (img_max - img_min) * 255.0
        return img_hu.astype(np.uint8)

    def write_modified_as_png(self, out_org, out_mod, filename, numbered, save_original, data_path_original,
                              center, width) -> None:
        """
        Export CT slices as PNG with proper windowing.

        :param out_org: original image data
        :param out_mod: modified image data
        :param filename: output filename
        :param numbered: whether using numbered filename convention
        :param save_original: whether to save original alongside modified
        :param data_path_original: path for original image
        :param center: window center (HU)
        :param width: window width (HU)
        """
        if center is None:
            center = self.dicom_header.WindowCenter[0]
        if width is None:
            width = self.dicom_header.WindowWidth[0]
        if numbered:
            if save_original:
                cv2.imwrite(data_path_original, self.get_window_ct(out_org, center, width))
            cv2.imwrite(filename, self.get_window_ct(out_mod, center, width))

        else:
            plt.imsave(filename + '.png', self.get_window_ct(self.__modified_pixel_array, center, width),
                       cmap='gray')
            if save_original:
                plt.imsave(data_path_original + '.png', self.get_window_ct(self.pixel_array, center, width),
                           cmap='gray')

    def write_modified_as_dicom(self, data_path) -> None:
        """
        Export modified CT data as DICOM file with proper metadata.

        :param data_path: output directory
        :return:
        """
        mod_dicom = self.dicom_header.copy()
        mod_pixel = self.__modified_pixel_array
        mod_pixel[mod_pixel < 0] = 0
        mod_pixel = np.round(mod_pixel).astype(np.uint16)
        mod_dicom.PixelData = mod_pixel.tobytes()
        filename = f'{data_path}/CT{mod_dicom.SOPInstanceUID}.dcm'
        mod_dicom.save_as(filename)

    def get_std_in_pixel(self) -> np.ndarray:
        """
        Get pixel spacing information from DICOM header

        :return: Array of [dx, dy, dz] spacing values in mm
        """
        dx, dy = self.dicom_header.PixelSpacing
        dz = self.dicom_header.SliceThickness
        return np.array([dx, dy, dz])

    def add_gaussian_white_noise(self, sigma, mean, rng) -> None:
        """
        Add gaussian white noise to pixel data.
        :param sigma: standard deviation of noise in HU
        :param mean: mean of noise distribution
        :param rng: random number generator
        """
        n = rng.normal(mean, sigma, self.pixel_array.shape)
        pix_hu = self.dicom_header.RescaleSlope * self.pixel_array + self.dicom_header.RescaleIntercept
        if not self.__modified_pixel_array is None:
            n = rng.normal(mean, sigma, self.__modified_pixel_array.shape)
            pix_hu = self.dicom_header.RescaleSlope * self.__modified_pixel_array + self.dicom_header.RescaleIntercept
        pix_hu = pix_hu + n
        self.__modified_pixel_array = (pix_hu - self.dicom_header.RescaleIntercept) / self.dicom_header.RescaleSlope
