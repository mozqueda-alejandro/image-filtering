from Project1 import project1 as p1
import numpy as np
import cv2


def gaussian_filter(height, width, sigma=1.0):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - height//2)**2 + (y - width//2)**2) / (2*sigma**2)), (height, width))
    kernel /= np.sum(kernel)
    return kernel


if __name__ == '__main__':
    # image = p1.load_img("images/color_lenna.png")
    image = p1.load_img("images/bw_stones.jpg")
    # print(image)
    mask = p1.generate_gaussian(1, 9, 1)
    print(mask)
    out = p1.apply_filter(image, mask, 4, 0)
    cv2.imwrite("1d.jpg", out)

    # im = p1.load_img("images/bw_stones.jpg")
    # cv2.imshow("Image", im)

    # width, height = 1, 20
    # color_image = np.random.randint(0, 256, (width, height), dtype=np.uint8)
    # print(color_image)

    # color_image = np.array([[0,0,0,0,0,0,0,0],
    #                         [0,0,0,0,0,0,0,0],
    #                         [0,0,0,0,0,0,0,0],
    #                         [0,0,0,0,0,0,0,0],
    #                         [0,0,0,0,0,0,0,0],
    #                         [0,0,0,0,0,0,0,0],
    #                         [0,0,0,0,0,0,0,0],
    #                         [200,200,200,200,200,200,200,200]])
    # # Save the array as an image using OpenCV
    # cv2.imwrite('bw_image2.jpg', color_image)
