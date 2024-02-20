import cv2
import numpy as np


def load_img(file_path) -> cv2.Mat:
    """
    Load an OpenCV image object from file
    :param file_path: path to image file
    :return: image object
    """
    return cv2.imread(file_path)


def display_img(image):
    """
    Display an image
    :param image: image object
    :return: None
    """
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_gaussian(sigma, filter_w, filter_h) -> np.ndarray:
    """
    Generate a 1D or 2D Gaussian filter. If either filter_w or filter_h are 1, it is a 1D filter
    :param sigma: standard deviation
    :param filter_w: width of filter
    :param filter_h: height of filter
    :return: 1D or 2D Gaussian filter
    """
    # 2D Gaussian
    if filter_w != 1 and filter_h != 1:
        if filter_w != filter_h:
            raise ValueError("Filter width and height must be equal")

        if filter_w % 2 == 0:
            filter_w += 1
        return _get_2d_kernel(sigma, filter_w)

    # 1D Gaussian
    if filter_w == 1 and filter_h == 1:
        raise ValueError("Filter width and height cannot both be 1")

    filter_length = max(filter_w, filter_h)
    if filter_length % 2 == 0:
        filter_length += 1
    return _get_1d_kernel(sigma, filter_length)


def apply_filter(image, mask, pad_pixels, pad_value):
    """
    Apply a filter to an image and return the result
    :param image: image object
    :param mask: 1D or 2D filter
    :param pad_pixels: number of pixels to pad on each side of the image
    :param pad_value: value to use for padding, if 0 uses black, else pad with edge values
    :return: filtered image
    """
    array = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
    if pad_value != 0:
        padded_image = np.pad(array, pad_pixels, mode='edge')
    else:
        padded_image = np.pad(array, pad_pixels, mode='constant', constant_values=pad_value)

    is_1d = len(mask.shape) == 1
    if is_1d:
        return _apply_1d_convolution(padded_image, mask, pad_pixels)
    else:
        return _apply_2d_convolution(padded_image, mask, pad_pixels)


def _apply_1d_convolution(image, mask, pad_pixels):
    """
    Apply a 1D filter to an image and return the result
    :param image: image object
    :param mask: 1D filter
    :return: filtered image
    """


def _apply_2d_convolution(image, mask, pad_pixels):
    """
    Apply a 2D filter to an image and return the result
    :param image: image object
    :param mask: 2D mask
    :return: filtered image
    """


def _get_1d_kernel(sigma, filter_length) -> np.ndarray:
    """
    Generate a 1D Gaussian mask
    :param sigma: standard deviation
    :param filter_length: width of mask
    :return: 1D Gaussian mask
    """
    mask = np.zeros(filter_length)
    center = filter_length // 2
    for i in range(filter_length):
        mask[i] = np.exp(-((i - center) ** 2) / (2 * (sigma ** 2)))
    return mask / np.sum(mask)


def _get_2d_kernel(sigma, filter_length) -> np.ndarray:
    """
    Generate a 2D Gaussian filter
    :param sigma: standard deviation
    :param filter_length: length of filter
    :return: 2D Gaussian filter
    """
    mask = np.zeros((filter_length, filter_length))
    center = filter_length // 2
    for i in range(filter_length):
        for j in range(filter_length):
            mask[i, j] = np.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * (sigma ** 2)))
    return mask / np.sum(mask)
