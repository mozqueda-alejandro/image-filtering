from image_utils import image_utils as iu
import numpy as np
import cv2

if __name__ == '__main__':
    # image = iu.load_img("images/color_lenna.png")
    mask = iu.generate_gaussian(1, 1, 5)
    print(mask)
    print(cv2.getGaussianKernel(5, 1))
    # print(iu.apply_filter(image, mask, 0, 3))
    # width, height = 10, 1
    # color_image = np.random.randint(0, 256, (width, height, 3), dtype=np.uint8)
    # print(color_image)
    #
    # # Save the array as an image using OpenCV
    # cv2.imwrite('color_image.jpg', color_image)
    #
    # # Optionally, display the image using OpenCV
    # cv2.imshow('Color Image', color_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
