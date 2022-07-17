import cv2 as cv
import numpy as np

"""
This file contains different function for preprocessing a single image.
Used during creations of new variations of datasets.
"""


def r_resized(img):
    return cv.resize(img, (227, 227))


def r_canny(img):
    return cv.Canny(img, 100, 200, apertureSize=3)


def r_laplacian(img):
    return cv.Laplacian(img, -1)


def r_roberts(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img = img.astype("float32")

    out = np.zeros(img.shape, dtype=np.float32)

    # Convolves the image with the roberts kernels
    for i in range(0, img.shape[0] - 1):
        for j in range(0, img.shape[1] - 1):
            dx = img[i + 1, j] - img[i, j + 1]

            dy = img[i + 1, j + 1] - img[i, j]
            out[i, j] = abs(dx) + abs(dy)

    out = out.astype("uint8")

    return out


def r_sobel(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sobelx = cv.Sobel(img, -1, 1, 0, ksize=1)
    sobely = cv.Sobel(img, -1, 0, 1, ksize=1)

    sobel_com = sobelx + sobely

    return sobel_com


def r_morphgrad(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # creates 5x5 rectangel structuring element
    SE = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

    # the morhpologial operations
    dilated = cv.dilate(img, SE, iterations=1)
    eroded = cv.erode(img, SE, iterations=1)

    gradient = dilated - eroded

    return gradient


# returns the function based on string name
def r_preprocess(f: str):
    if f == 'resize':
        return r_resized
    elif f == 'canny':
        return r_canny
    elif f == 'laplacian':
        return r_laplacian
    elif f == 'roberts':
        return r_roberts
    elif f == 'sobel':
        return r_sobel
    elif f == 'morphgrad':
        return r_morphgrad

    else:
        raise Exception("This preprocess method doesn't exist")


# testing preprocess function system works
if __name__ == '__main__':
    img = cv.imread(f'image_data/NI/train/dog/dog_0003.jpg')

    img = r_preprocess('morphgrad')(img)

    cv.imshow('resized', img)
    cv.waitKey(0)
