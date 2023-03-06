import numpy
import scipy.signal

import eagle.gaussian.filter

# differential invariant vector
# def differential_invariant(img: numpy.ndarray, sigma: float = numpy.sqrt(2) / 2) -> numpy.ndarray:

#     gradient = eagle.gaussian.filter.gradient(sigma)
#     img_dx = scipy.signal.convolve2d(img, gradient[0], mode='same')
#     img_dy = scipy.signal.convolve2d(img, gradient[0], mode='same')

#     hessian = eagle.gaussian.filter.hessian(sigma)
#     img_dxdx = scipy.signal.convolve2d(img, hessian[0, 0], mode='same')
#     img_dxdy = scipy.signal.convolve2d(img, hessian[0, 1], mode='same')
#     # Not use because symetric
#     # img_dydx = scipy.signal.convolve2d(img, hessian[1, 0], mode='same')
#     img_dydy = scipy.signal.convolve2d(img, hessian[1, 1], mode='same')

#     img_dx2, img_dy2 = img_dx**2, img_dy**2
#     img_dxy = img_dx * img_dy

#     v0 = img
#     v1 = img_dx2 + img_dy2
#     v2 = img_dxdx*img_dx2 + 2*img_dxdy*img_dxy + img_dydy*img_dy2
#     v3 = img_dxdx + img_dydy
#     v4 = img_dxdx*img_dxdx + 2*img_dxdy*img_dxdy + img_dydy*img_dydy

#     # v[:, i, j]
#     v = numpy.array([v0, v1, v2, v3, v4])
    
#     return v

# def get_vector(diff_inv: numpy.ndarray)


class DifferentialInvariant:

    def __init__(self, img: numpy.ndarray, sigma: float = numpy.sqrt(2) / 2):

        gradient = eagle.gaussian.filter.gradient(sigma)
        img_dx = scipy.signal.convolve2d(img, gradient[0], mode='same')
        img_dy = scipy.signal.convolve2d(img, gradient[0], mode='same')

        hessian = eagle.gaussian.filter.hessian(sigma)
        img_dxdx = scipy.signal.convolve2d(img, hessian[0, 0], mode='same')
        img_dxdy = scipy.signal.convolve2d(img, hessian[0, 1], mode='same')
        # Not use because symetric
        # img_dydx = scipy.signal.convolve2d(img, hessian[1, 0], mode='same')
        img_dydy = scipy.signal.convolve2d(img, hessian[1, 1], mode='same')

        img_dx2, img_dy2 = img_dx**2, img_dy**2
        img_dxy = img_dx * img_dy

        v0 = img
        v1 = img_dx2 + img_dy2
        v2 = img_dxdx*img_dx2 + 2*img_dxdy*img_dxy + img_dydy*img_dy2
        v3 = img_dxdx + img_dydy
        v4 = img_dxdx*img_dxdx + 2*img_dxdy*img_dxdy + img_dydy*img_dydy

        # v[:, i, j]
        self.__vectors = numpy.array([v0, v1, v2, v3, v4])

    def __getitem__(self, indices) -> numpy.ndarray:
        i, j = indices
        return self.__vectors[:, i, j]






