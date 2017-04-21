from skimage.filters import threshold_otsu
from scipy.ndimage.filters import median_filter
import numpy as np


def trim(img, dx=44):
    post = img[dx:img.shape[0]-dx, dx:img.shape[1]-dx, dx:img.shape[2]-dx]
    return post


def normalize(img):
    return img/255.


def despeckle(img):
    return median_filter(img, size=(3, 3, 3))


def threshold(img):
    threshold_global_otsu = threshold_otsu(img)
    segmented = (img >= threshold_global_otsu).astype(np.int32)
    return segmented