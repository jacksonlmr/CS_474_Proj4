import numpy as np
import cv2
import os
from exp_1_helpers import pq_pad, remove_pad, mapValues, compute_spectrum

FILE_SAVE_PATH = 'exp_3'

def construct_high_pass(shape: tuple[int, int], yh: float, yl: float, c: float, d0: float):
    N, M = shape
    output_filter = np.empty(shape)

    exp_denominator = d0**2
    for u in range(N):
        for v in range(M):      
            exp_numerator = (u**2)+(v**2)
            exp_power = (-c)*(exp_numerator/exp_denominator)
            output_filter[u, v] = (yh-yl)*(1 - (np.exp(exp_power))) + yl

    spectrum = compute_spectrum(output_filter)
    spectrum = 4*spectrum
    spectrum = mapValues(spectrum)
    cv2.imwrite(os.path.join(FILE_SAVE_PATH, f'filter_yh-{yh}_yl-{yl}_c-{c}d0-{d0}.png'), spectrum)
    return output_filter

def homo_filter(input_img: np.ndarray, yh: float, yl: float, c: float, d0: float):
    N, M = input_img.shape
    logged_img = np.log(input_img)
    padded_input = pq_pad(logged_img)

    hp_filter = construct_high_pass(padded_input.shape, yh, yl, c, d0)

    logged_fft = np.fft.fft2(padded_input)
    filtered_img_fft = logged_fft * hp_filter

    filtered_img = np.fft.ifft2(filtered_img_fft).real
    filtered_img = remove_pad(filtered_img, N, M)
    filtered_img = mapValues(filtered_img)

    return filtered_img