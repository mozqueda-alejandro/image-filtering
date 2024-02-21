import cv2
import numpy as np
import math


def load_img(file_path) -> cv2.Mat:
    """
    Load an OpenCV image object from file
    :param file_path: path to image file
    :return: image object
    """
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)


def display_img(image) -> None:
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
    # Ensure filter length is odd
    def make_odd(num):
        if num % 2 == 0:
            return num + 1
        return num

    # 2D Gaussian
    if filter_w != 1 and filter_h != 1:
        return _get_2d_kernel(sigma, make_odd(filter_w), make_odd(filter_h))

    # 1D Gaussian
    if filter_w == 1 and filter_h == 1:
        raise ValueError("Filter width and height cannot both be 1")

    filter_length = max(filter_w, filter_h)
    return _get_1d_kernel(sigma, make_odd(filter_length))


def apply_filter(image, mask, pad_pixels, pad_value):
    """
    Apply a filter to an image and return the result
    :param image: image object
    :param mask: 1D or 2D filter
    :param pad_pixels: number of pixels to pad on each side of the image
    :param pad_value: value to use for padding, if 0 uses black, else pad with edge values
    :return: filtered image
    """

    padding_flag = ((1, 1), (1, 1)) if len(image.shape) == 2 else ((1, 1), (1, 1), (0, 0))
    if pad_value == 0:
        padded_image = np.pad(image, padding_flag, mode='constant', constant_values=0)
    else:
        padded_image = np.pad(image, padding_flag, mode='edge')
    print(padded_image)
    # Check if mask is bigger than image and padding
    if (mask.shape[0] > padded_image.shape[0] or
            len(mask.shape) > 1 and (mask.shape[1] > padded_image.shape[1])):
        raise ValueError(f'''Mask is larger than image with padding
        Mask shape:           {mask.shape}
        Image (padded) shape: {padded_image.shape}''')

    # return padded_image

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
    print("1D Convolution")
    output_shape = (image.shape[0] - 2 * pad_pixels, image.shape[1] - 2 * pad_pixels)
    output = np.zeros(output_shape, dtype=int)
    for i in range(pad_pixels, image.shape[0] - pad_pixels):
        for j in range(pad_pixels, image.shape[1] - pad_pixels):
            # try:
            output[i - pad_pixels, j - pad_pixels] = np.sum(image[i, j - pad_pixels:j + mask.shape[0] - pad_pixels] * mask)
            # except Exception:
            #     print("Errrorrrr->", image[i, j - pad_pixels:j + mask.shape[0] - pad_pixels])
            #     return

    return output


def _apply_2d_convolution(image, mask, pad_pixels):
    """
    Apply a 2D filter to an image and return the result
    :param image: image object
    :param mask: 2D mask
    :return: filtered image
    """
    print("2D Convolution")


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


def _get_2d_kernel(sigma, filter_w, filter_h) -> np.ndarray:
    """
    Generate a 2D Gaussian filter
    :param sigma: standard deviation
    :param filter_w: width of filter
    :param filter_h: height of filter
    :return: 2D Gaussian filter
    """
    mask = np.zeros((filter_w, filter_h))
    center_w = filter_w // 2
    center_h = filter_h // 2
    for i in range(filter_w):
        for j in range(filter_h):
            mask[i, j] = np.exp(-((i - center_w) ** 2 + (j - center_h) ** 2) / (2 * (sigma ** 2)))
    return mask / np.sum(mask)
