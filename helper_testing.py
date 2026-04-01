from exp_1_helpers import pq_pad, band_reject, remove_pad, create_notch_filter
from exp_2_helpers import pad, convert_filter
from exp_3_helpers import construct_high_pass
import numpy as np
import cv2

# Test pq_pad
# test_array = np.full((4, 4), 255)
# test_array_padded = pq_pad(test_array)
# print(test_array_padded)

# padding_removed = remove_pad(test_array_padded, test_array.shape[0], test_array.shape[1])
# print(padding_removed)

# Test create filter 
# test_filter = create_filter((11, 11), 2, .5, True)
# print(test_filter)

# Test band reject
# band_test_array = np.array([[1, 1, 1, 1, 1],
#                             [1, 2, 2, 2, 1],
#                             [1, 2, 3, 2, 1],
#                             [1, 2, 2, 2, 1],
#                             [1, 1, 1, 1, 1]])

# print(band_reject(band_test_array, 1, 1))

# Test create notch filter
# notch_filter = create_notch_filter((100, 100), [(14, 14)], 3)
# print(notch_filter)

# Test pad
# pad_test_array = np.array([[1, 1, 1, 1, 1],
#                             [1, 2, 2, 2, 1],
#                             [1, 2, 3, 2, 1],
#                             [1, 2, 2, 2, 1],
#                             [1, 1, 1, 1, 1]])

# padded_test_array = pad(pad_test_array, (7, 7))
# print(padded_test_array)

# Test convert filter
# convert_test_array = np.array([[-1, 0, 1],
#                                [-2, 0, 2],
#                                [-1, 0, 1]])

# converted_test_array = convert_filter(convert_test_array, (7, 7))
# print(np.round(converted_test_array, 2))

# Test construct_high_pass
high_pass = construct_high_pass((10, 10), 1, 0, 1, 3)
print(np.round(high_pass, 2))