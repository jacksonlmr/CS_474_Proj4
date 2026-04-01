import numpy as np
import cv2
import os
from exp_1_helpers import remove_pad

FILE_SAVE_PATH = 'exp_2'

def filter_freq(input_img: np.ndarray, spatial_filter: np.ndarray):
    N, M = input_img.shape
    fN, fM = spatial_filter.shape
    pad_N, pad_M = (N+fN-1, M+fM-1)
    padded_shape = (pad_N, pad_M)

    # Pad image and filter, fft on filter
    freq_filter = convert_filter(spatial_filter, padded_shape)
    padded_image = pad(input_img, padded_shape)

    # visualize spectrum of filter
    freq_spectrum = np.abs(freq_filter)
    freq_spectrum = np.fft.fftshift(freq_spectrum)
    freq_spectrum = mapValues(freq_spectrum)
    cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'spectrum_freq_sobel_filter.png'), freq_spectrum.astype(np.uint8))
    
    # fft on image
    pad_img_fft = np.fft.fft2(padded_image)

    # compute filtered frequency domain, ifft
    result_fft = freq_filter * pad_img_fft
    result = np.fft.ifft2(result_fft).real

    result = remove_pad(result, N, M)
    result = mapValues(result)

    return result


def convert_filter(input_filter: np.ndarray, output_size: tuple[int, int]):
    # pad filter
    filter_padded = pad(input_filter, output_size)
    # fft filter
    filter_fft = np.fft.fft2(filter_padded)

    return filter_fft


def pad(input_arr: np.ndarray, padded_shape: tuple[int, int]):
    N, M = input_arr.shape
    padded_array = np.zeros(padded_shape)
    padded_array[:N, :M] = input_arr

    return padded_array


def mapValues(input_img_array: np.ndarray):
    """
    Maps values from detected range to [0, 255]

    **Parameters**
    ---------------
    >**input_img_array**:
    >np.ndarray representing image

    **Returns**
    -----------
    >**output_img_array**: np.ndarray representing an image with the values mapped to [0, 255]
    """
    input_row, input_col = input_img_array.shape
    output_img_array = np.zeros((input_row, input_col), dtype=np.uint8)

    max_value = np.max(input_img_array)

    for current_row in range(input_row):
        for current_col in range(input_col):
            current_value = input_img_array[current_row, current_col]
            mapped_value = int(max(0, min(255, 255*(current_value/max_value))))
            output_img_array[current_row, current_col] = mapped_value

    return output_img_array