import cv2
import os
import numpy as np
from exp_1_helpers import band_reject, notch_reject, compute_spectrum, gaussian, extract_noise

FILE_SAVE_PATH = 'exp_1'

gaussian7 = np.array([
    [1, 1, 2, 2, 2, 1, 1],
    [1, 2, 2, 4, 2, 2, 1],
    [2, 2, 4, 8, 4, 2, 2],
    [2, 4, 8, 16, 8, 4, 2],
    [2, 2, 4, 8, 4, 2, 2],
    [1, 2, 2, 4, 2, 2, 1],
    [1, 1, 2, 2, 2, 1, 1]
])

gaussian15 = np.array([
    [2,2,3,4,5,5,6,6,6,5,5,4,3,2,2],
    [2,3,4,5,7,7,8,8,8,7,7,5,4,3,2],
    [3,4,6,7,9,10,10,11,10,10,9,7,6,4,3],
    [4,5,7,9,10,12,13,13,13,12,10,9,7,5,4],
    [5,7,9,11,13,14,15,16,15,14,13,11,9,7,5],
    [5,7,10,12,14,16,17,18,17,16,14,12,10,7,5],
    [6,8,10,13,15,17,19,19,19,17,15,13,10,8,6],
    [6,8,11,13,16,18,19,20,19,18,16,13,11,8,6],
    [6,8,10,13,15,17,19,19,19,17,15,13,10,8,6],
    [5,7,10,12,14,16,17,18,17,16,14,12,10,7,5],
    [5,7,9,11,13,14,15,16,15,14,13,11,9,7,5],
    [4,5,7,9,10,12,13,13,13,12,10,9,7,5,4],
    [3,4,6,7,9,10,10,11,10,10,9,7,6,4,3],
    [2,3,4,5,7,7,8,8,8,7,7,5,4,3,2],
    [2,2,3,4,5,5,6,6,6,5,5,4,3,2,2]
])

boy_noisy_array = cv2.imread(os.path.join(FILE_SAVE_PATH, 'boy_noisy.gif'), flags=0)

# band_filtered_boy_array = band_reject(boy_noisy_array, 70, 40)
# cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'boy_band_filtered.png'), band_filtered_boy_array)

# notch_filtered_boy_array = notch_reject(boy_noisy_array, [(550, 448), (544, 576)], 22)
# cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'boy_notch_filtered.png'), notch_filtered_boy_array)

# boy_noisy_spectrum = compute_spectrum(boy_noisy_array)
# cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'boy_noisy_spectrum.png'), boy_noisy_spectrum)

# boy_band_spectrum = compute_spectrum(band_filtered_boy_array)
# cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'boy_band_spectrum.png'), boy_band_spectrum)

# boy_notch_spectrum = compute_spectrum(notch_filtered_boy_array)
# cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'boy_notch_spectrum.png'), boy_notch_spectrum)

# boy_gaussian_7 = gaussian(boy_noisy_array, gaussian7)
# cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'boy_gaussian_7.png'), boy_gaussian_7)

# boy_gaussian_15 = gaussian(boy_noisy_array, gaussian15)
# cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'boy_gaussian_15.png'), boy_gaussian_15)

# PART B
boy_extracted_noise = extract_noise(boy_noisy_array, [(550, 448), (544, 576)], 22)
cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'boy_extracted_noise.png'), boy_extracted_noise)

spectrum_extracted_noise = compute_spectrum(boy_extracted_noise)
cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'spectrum_extracted_noise.png'), spectrum_extracted_noise)
