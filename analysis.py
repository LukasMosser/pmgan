import numpy as np
from numba import jit
from dask import delayed


#@delayed
def porosity(img):
    counts = np.unique(img, return_counts=True)
    return counts[1][0]/float(counts[1][1])


#@delayed
def two_point_probability(img, pore_phase=0):
    two_point_covariance_pore_phase = {}
    for i, direc in enumerate(["x", "y", "z"]):
        two_point_direc = two_point_correlation(img, i, var=pore_phase)
        two_point_covariance_pore_phase[direc] = two_point_direc

    direc_covariances_pore_phase = {}
    for direc in ["x", "y", "z"]:
        direc_covariances_pore_phase[direc] = np.mean(np.mean(two_point_covariance_pore_phase[direc], axis=0), axis=0)

    return direc_covariances_pore_phase


@jit
def two_point_correlation(im, dim, var=0):
    """
    This method computes the two point correlation,
    also known as second order moment,
    for a segmented binary image in the three principal directions.

    dim = 0: x-direction
    dim = 1: y-direction
    dim = 2: z-direction

    var should be set to the pixel value of the pore-space. (Default 0)

    The input image im is expected to be three-dimensional.
    """
    if dim == 0:  # x_direction
        dim_1 = im.shape[2]  # y-axis
        dim_2 = im.shape[1]  # z-axis
        dim_3 = im.shape[0]  # x-axis
    elif dim == 1:  # y-direction
        dim_1 = im.shape[0]  # x-axis
        dim_2 = im.shape[1]  # z-axis
        dim_3 = im.shape[2]  # y-axis
    elif dim == 2:  # z-direction
        dim_1 = im.shape[0]  # x-axis
        dim_2 = im.shape[2]  # y-axis
        dim_3 = im.shape[1]  # z-axis

    two_point = np.zeros((dim_1, dim_2, dim_3))
    for n1 in range(dim_1):
        for n2 in range(dim_2):
            for r in range(dim_3):
                lmax = dim_3 - r
                for a in range(lmax):
                    if dim == 0:
                        pixel1 = im[a, n2, n1]
                        pixel2 = im[a + r, n2, n1]
                    elif dim == 1:
                        pixel1 = im[n1, n2, a]
                        pixel2 = im[n1, n2, a + r]
                    elif dim == 2:
                        pixel1 = im[n1, a, n2]
                        pixel2 = im[n1, a + r, n2]

                    if pixel1 == var and pixel2 == var:
                        two_point[n1, n2, r] += 1
                two_point[n1, n2, r] = two_point[n1, n2, r] / (float(lmax))
    return two_point