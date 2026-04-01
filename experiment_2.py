import numpy as np
import cv2
import os
from scipy.signal import convolve2d
from exp_2_helpers import filter_freq, mapValues
from exp_1_helpers import compute_spectrum

outfile_save_path = 'exp_2'
# spatial domain sobel
sobel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

#lenna sobel spatial domain
lenna = cv2.imread(os.path.join(outfile_save_path, 'lenna.gif'), flags=0)
lenna_sobel_spatial = convolve2d(lenna, sobel, mode='same')
lenna_sobel_spatial = mapValues(lenna_sobel_spatial)
cv2.imwrite(os.path.join(outfile_save_path, 'lenna_sobel_spatial.png'), lenna_sobel_spatial)

# lenna sobel freq domain
lenna_sobel_freq = filter_freq(lenna, sobel)
cv2.imwrite(os.path.join(outfile_save_path, 'lenna_sobel_freq.png'), lenna_sobel_freq)

# get spectrums
lenna_spectrum = compute_spectrum(lenna)
cv2.imwrite(os.path.join(outfile_save_path, 'spectrum_lenna.png'), lenna_spectrum)

lenna_sobel_spatial_spectrum = compute_spectrum(lenna_sobel_spatial)
cv2.imwrite(os.path.join(outfile_save_path, 'spectrum_lenna_sobel_spatial.png'), lenna_sobel_spatial_spectrum)

lenna_sobel_freq_spectrum = compute_spectrum(lenna_sobel_freq)
cv2.imwrite(os.path.join(outfile_save_path, 'spectrum_lenna_sobel_freq.png'), lenna_sobel_freq_spectrum)

# in report mention that images are not exactly the same to to floating point rounding errors in mapvalues