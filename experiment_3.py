from exp_1_helpers import compute_spectrum
from exp_3_helpers import homo_filter
import numpy as np
import os
import cv2

FILE_SAVE_PATH = 'exp_3'

girl = cv2.imread(os.path.join(FILE_SAVE_PATH, 'girl.gif'), flags=0)
spectrum_girl = compute_spectrum(girl)
cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'spectrum_girl.png'), spectrum_girl)

# # Starting point
# homo_filter_girl = homo_filter(girl, 1.5, .5, 1, 1.8)
# cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'homo_filtered_girl_start.png'), homo_filter_girl)

# spectrum_homo_filter_girl = compute_spectrum(homo_filter_girl)
# cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'spectrum_homo_filtered_girl_start.png'), spectrum_homo_filter_girl)


# # Increased yl
# yl_up_homo_filter_girl = homo_filter(girl, 1.5, 4, 1, 1.8)
# cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'yl_up_homo_filtered_girl.png'), yl_up_homo_filter_girl)

# spectrum_yl_up_homo_filter_girl = compute_spectrum(yl_up_homo_filter_girl)
# cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'spectrum_yl_up_homo_filtered_girl.png'), spectrum_yl_up_homo_filter_girl)


# # Decreased yl
# yl_down_homo_filter_girl = homo_filter(girl, 1.5, 0.01, 1, 1.8)
# cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'yl_down_homo_filtered_girl.png'), yl_down_homo_filter_girl)

# spectrum_yl_down_homo_filter_girl = compute_spectrum(yl_down_homo_filter_girl)
# cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'spectrum_yl_down_homo_filtered_girl.png'), spectrum_yl_down_homo_filter_girl)


# # Increased yh
# yh_up_homo_filter_girl = homo_filter(girl, 5, .5, 1, 1.8)
# cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'yh_up_homo_filtered_girl.png'), yh_up_homo_filter_girl)

# spectrum_yh_up_homo_filter_girl = compute_spectrum(yh_up_homo_filter_girl)
# cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'spectrum_yh_up_homo_filtered_girl.png'), spectrum_yh_up_homo_filter_girl)


# # Decreased yh
# yh_down_homo_filter_girl = homo_filter(girl, .1, .5, 1, 1.8)
# cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'yh_down_homo_filtered_girl.png'), yh_down_homo_filter_girl)

# spectrum_yh_down_homo_filter_girl = compute_spectrum(yh_down_homo_filter_girl)
# cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'spectrum_yh_down_homo_filtered_girl.png'), spectrum_yh_down_homo_filter_girl)


# Ideal
ideal_homo_filter_girl = homo_filter(girl, 5, .01, 1, 1.8)
cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'ideal_homo_filtered_girl.png'), ideal_homo_filter_girl)

spectrum_ideal_homo_filter_girl = compute_spectrum(ideal_homo_filter_girl)
cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'spectrum_ideal_homo_filtered_girl.png'), spectrum_ideal_homo_filter_girl)