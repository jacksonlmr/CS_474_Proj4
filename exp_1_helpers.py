import numpy as np
from math import sqrt
import cv2
import os
import random
from typing import Callable
from multipledispatch import dispatch

FILE_SAVE_PATH = 'exp_1'

def pq_pad(input_img: np.ndarray):
    
    in_rows = input_img.shape[0]
    in_cols = input_img.shape[1]

    padded_rows = 2*in_rows
    padd_cols = 2*in_cols

    output_img = np.zeros((padded_rows, padd_cols))

    output_img[:in_rows, :in_cols] = input_img

    return output_img

def remove_pad(input_img: np.ndarray, og_rows: int, og_cols: int):
    output_img = np.empty((og_rows, og_cols))

    output_img = input_img[:og_rows, :og_cols]

    return output_img


def extract_noise(input_img: np.ndarray, notches: list[tuple[int, int]], sigma: int):
    M, N = input_img.shape
    # do padding, fft, and shift
    input_fft = shifted_fft(input_img)
    # M, N = input_fft.shape
    output_fft = np.empty_like(input_fft)

    # create notch filter
    notch_filter = create_notch_filter(input_fft.shape, notches, sigma)
    notch_filter = 1 - notch_filter

    centered_mag = np.abs(notch_filter)
    centered_mag = np.log(1 + centered_mag)
    centered_mag = cv2.normalize(centered_mag, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'spectrum_filter_extract_noise.png'), centered_mag)
    # multiply notch filter and fft
    output_fft = input_fft * notch_filter
    # unshift result
    output_fft = np.fft.ifftshift(output_fft)
    # inverse fft
    output_img = np.fft.ifft2(output_fft)
    # remove padding
    output_img = remove_pad(output_img, M, N)
    output_img = np.clip(output_img, 0, 255)

    return output_img.astype(np.uint8)

def notch_reject(input_img: np.ndarray, notches: list[tuple[int, int]], sigma: int):
    
    M, N = input_img.shape
    # do padding, fft, and shift
    input_fft = shifted_fft(input_img)
    # M, N = input_fft.shape
    output_fft = np.empty_like(input_fft)

    # create notch filter
    notch_filter = create_notch_filter(input_fft.shape, notches, sigma)
    # multiply notch filter and fft
    output_fft = input_fft * notch_filter
    # unshift result
    output_fft = np.fft.ifftshift(output_fft)
    # inverse fft
    output_img = np.fft.ifft2(output_fft)
    # remove padding
    output_img = remove_pad(output_img, M, N)
    output_img = np.clip(output_img, 0, 255)

    return output_img.astype(np.uint8)


def create_notch_filter(shape: tuple[int, int], notches: list[tuple[int, int]], sigma: int):
    M, N = shape
    notch_functions = []

    all_notches = []
    for notch in notches:
        all_notches.append(notch)
        u, v = notch
        u_sym = M - u
        v_sym = N - v
        all_notches.append((u_sym, v_sym))

    for i in range(len(all_notches)):
        func = np.empty(shape)
        for row in range(M):
            for col in range(N):
                func[row, col] = gaussian_hp((row, col), all_notches[i], sigma)

        notch_functions.append(func)

    final_filter = np.ones_like(notch_functions[0])
    for func in notch_functions:
        final_filter = final_filter * func
    
    filter_normalized = cv2.normalize(final_filter, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'boy_notch_filter.png'), filter_normalized.astype(np.uint8))
    return final_filter


def gaussian_hp(loc: tuple[int, int], notch_center: tuple[int, int], sigma: int):
    dist = sqrt((loc[0]-notch_center[0])**2 + (loc[1]-notch_center[1])**2) #computes distance between notch location and corrent point. Move to function later
    numerator = -(dist**2)
    denominator = 2*(sigma**2)
    return 1 - np.exp(numerator/denominator)


def band_reject(input_img: np.ndarray, cutoff: float, width: float):
    """
    Calculates the value of gaussian_band band reject filtered image for a certain input image.

    :param input_img: Image to be filtered
    :type input_image: np.ndarray
    :param cutoff: Cutoff frequency of filter.
    :type cutoff: float
    :param width: Width of filter.
    :type width: float
    """
    # pad image, compute fft, shift fft, create empty output image
    input_fft = shifted_fft(input_img)
    output_img_fft = np.empty_like(input_fft)

    # compute gaussian_band filter mask H(u, v)
    centered_filter = create_band_filter(input_fft.shape, cutoff, width)

    # element wise multiplication between img fft and filter, transform back to spatial, remove padding
    output_img_fft = input_fft * centered_filter
    output_img_fft = np.fft.ifftshift(output_img_fft)
    output_img = np.fft.ifft2(output_img_fft)
    output_img = remove_pad(output_img, input_img.shape[0], input_img.shape[1])

    return output_img.real.astype(np.uint8)

def shifted_fft(input_img: np.ndarray):
    input_padded = pq_pad(input_img)
    input_fft = np.fft.fft2(input_padded)
    input_fft = np.fft.fftshift(input_fft)

    return input_fft

def compute_spectrum(input_img: np.ndarray):
    input_fft = np.fft.fft2(input_img)
    centered_fft = np.fft.fftshift(input_fft)
    centered_mag = np.abs(centered_fft)
    centered_mag = np.log(1 + centered_mag)

    centered_mag = cv2.normalize(centered_mag, None, 0, 255, cv2.NORM_MINMAX)
    return centered_mag.astype(np.uint8)


def create_band_filter(shape: tuple[int, int], cutoff: float, width: float):
    """
    Creates a gaussian_band band-reject filter mask 
    
    :param shape: Tuple containing desired filter shape
    :type shape: tuple[int, int]
    :param cutoff: Cutoff frequency to use for filter
    :type cutoff: float
    :param width: Width of band 
    :type width: float
    """
    filter = np.empty(shape)
    M, N = shape

    for row in range(shape[0]):
        for col in range(shape[1]):
            dist = distance((row, col), (M, N))
            filter[row, col] = gaussian_band(dist, cutoff, width)
    
    filter_normalized = cv2.normalize(filter, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(FILE_SAVE_PATH, 'boy_band_filter.png'), filter_normalized.astype(np.uint8))

    return filter


def gaussian_band(distance: float, cutoff: float, width: float) -> float:
    """
    Calculates the value of gaussian_band filter for a certain (u, v). 
    Note input is not (u, v), but distance from the center of the spectrum of (u, v).
    
    :param distance: D(u, v), distance of point (u, v) from center of spectrum.
    :type distance: float
    :param cutoff: Cutoff frequency of filter.
    :type cutoff: float
    :param width: Width of filter.
    :type width: float
    """
    exp_numerator = (distance**2)-(cutoff**2)
    exp_denominator = distance*width

    if exp_denominator == 0:
        return 1
    
    result = 1 - np.exp(-((exp_numerator/exp_denominator)**2))
    return result # need to remove rounding in final product
    

def distance(coord: tuple[int, int], shape: tuple[int, int]):
    u, v = coord
    return sqrt(((u-(shape[0]/2))**2)+((v-(shape[1]/2))**2))

def gaussian(input_img_array: np.ndarray, weights: np.ndarray):
    """
    Computes the gaussian of an image with a given mask.

    **Parameters**
    ---------------
    >**input_img_array**:
    >np.ndarray representing image. Should have dtype=np.uint8.

    >**weights**:
    >np.ndarray representing the mask to be used. Should have dtype=np.uint8.

    **Returns**
    -----------
    >**output_array**: 2D array representing the guassian blurred image.
    """
    #normalize weights
    weights = weights*(1/np.sum(weights))
    return traverseImage(input_img_array, weights, weightSumMatrix)

@dispatch(np.ndarray, np.ndarray, object)
def traverseImage(input_img_array: np.ndarray, weights: np.ndarray, operation: Callable[[np.ndarray, np.ndarray], np.ndarray]):
    """
    Computes the correlation of an image with a given mask.

    **Parameters**
    ---------------
    >**input_img_array**:
    >np.ndarray representing image. Should have dtype=np.uint8.

    >**weights**:
    >np.ndarray representing the mask to be used. Should have dtype=np.uint8.

    >**operation**:
    >Function that take 2 np.ndarray's as arguments and returns an integer value for pixel value

    **Returns**
    -----------
    >**output_array**: 2D array representing the output of performing the passed functionality at each pixel
    """
    input_row, input_col = input_img_array.shape

    #determine padding size and pad image
    weights = np.array(weights)
    mask_size = weights.shape[1] #since mask should always be square
    pad_size = mask_size//2
    padded_img_array = np.pad(array=input_img_array, pad_width=pad_size)

    # height for rows, width for cols
    output_array = np.zeros((input_row, input_col), dtype=np.int64)
    for current_row in range(input_row):
        for current_col in range(input_col):
            padded_row = current_row+pad_size
            padded_col = current_col+pad_size

            neighborhood = getNeighborhood(padded_img_array, (padded_row, padded_col), mask_size)
            pixel_value = operation(neighborhood, weights)

            output_array[current_row, current_col] = pixel_value

    output_array = mapValues(output_array)
    return output_array

@dispatch(np.ndarray, int, object)
def traverseImage(input_img_array: np.ndarray, size: int, operation: Callable[[np.ndarray, np.ndarray], np.ndarray]):
    """
    Computes the correlation of an image with a given mask.

    **Parameters**
    ---------------
    >**input_img_array**:
    >np.ndarray representing image. Should have dtype=np.uint8.

    >**size**:
    >size of the neighborhood to compute at each pixel

    >**operation**:
    >Function that takes 1 np.ndarray and size as arguments and returns an integer value for pixel value

    **Returns**
    -----------
    >**output_array**: 2D array representing the output of performing the passed functionality at each pixel
    """
    input_row, input_col = input_img_array.shape

    #determine padding size and pad image
    pad_size = size//2
    padded_img_array = np.pad(array=input_img_array, pad_width=pad_size)

    # height for rows, width for cols
    output_array = np.zeros((input_row, input_col), dtype=np.uint64)
    for current_row in range(input_row):
        for current_col in range(input_col):
            padded_row = current_row+pad_size
            padded_col = current_col+pad_size

            neighborhood = getNeighborhood(padded_img_array, (padded_row, padded_col), size)
            pixel_value = operation(neighborhood)

            output_array[current_row, current_col] = pixel_value

    output_array = mapValues(output_array)
    return output_array

def getNeighborhood(input_img_array: np.ndarray, pixel: tuple, size: int):
    """
    Gets the neighborhood surrounding pixel

    **Parameters**
    ---------------
    >**input_img_array**:
    >np.ndarray representing image

    >**pixel**:
    >Tuple containing (row, column) coordinates of the pixel to use for computing neighborhood

    >**size**:
    >equal to the width and height (neighborhood is always square) of the desired neighborhood shape

    **Returns**
    -----------
    >**neighborhood**: 2D numpy array of shape (size, size)
    """
    #straight line distance from center pixel to edge of neighborhood
    neighbor_distance = size//2
    
    #row and column position for top left corner of neighborhood in input_img
    top_left_row = pixel[0] - neighbor_distance
    top_left_col = pixel[1] - neighbor_distance
     
    #size of input image array
    input_row, input_col = input_img_array.shape
    
    neighborhood = np.zeros((size, size), dtype=np.uint8)

    for current_row, n_current_row in zip(range(top_left_row, top_left_row+size), range(size)):
        for current_col, n_current_col in zip(range(top_left_col, top_left_col+size), range(size)):
            
            #check to make sure coordinate is in bounds
            if 0 <= current_row < input_row and 0 <= current_col < input_col:
                #getpixel takes (width, height) -> (col, row)
                neighborhood[n_current_row, n_current_col] = input_img_array[current_row, current_col]
            else:
                neighborhood[n_current_row, n_current_col] = 0
    
    return neighborhood

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

def weightSumMatrix(matrix: np.ndarray, weight: np.ndarray):
    """
    Sums all values of **matrix**, each weighted with the corresponding value in **weight**

    **Parameters**
    ---------------
    >**matrix**:
    >np.ndarray representing the matrix to be summed

    >**weight**:
    >np.ndarray representing the weights

    **Returns**
    -----------
    >**sum**: integer result of the operation
    """
    sum = 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            sum += matrix[row, col]*weight[row, col]

    return int(sum)

def salt_pepper_noise(input_img_array: np.ndarray, noise_percent: float):
    #calculate step to go through array based on percent
    input_rows = input_img_array.shape[0]
    input_cols = input_img_array.shape[1]
    output_img_array = input_img_array.copy()

    #at each, randomly make the pixel black or white (50% chance of each)
    for row in range(input_rows):
        for col in range(input_cols):
            if (random.random() < noise_percent):
                if random.randint(0, 1) == 0:
                    output_img_array[row, col] = 0
                else:
                    output_img_array[row, col] = 255

    return output_img_array

def get_median(input_array: np.ndarray):
    return int(np.median(input_array))

def add_images(input_img_1: np.ndarray, input_image_2: np.ndarray):
    """
    Computes the addition of 2 images. Input images should be the same shape. 

    **Parameters**
    ---------------
    >**x_values**:
    >np.ndarray. Should have dtype=np.uint8.

    >**y_values**:
    >np.ndarray. Should have dtype=np.uint8.

    **Returns**
    -----------
    >**output_array**: 2D array representing the addition of the 2 images.
    """
    #both images should be same shape
    rows, cols = input_img_1.shape

    #calculate the magnitude of the gradient at every pixel in the image
    summed_array = np.zeros(shape=(rows, cols), dtype=np.int64)
    for row in range(rows):
        for col in range(cols):
            img_1_value = input_img_1[row, col]
            img_2_value = input_image_2[row, col]

            values_sum = img_1_value+img_2_value
            summed_array[row, col] = values_sum

    summed_array = mapValues(summed_array)
    return summed_array