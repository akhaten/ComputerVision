import numpy
import scipy.signal

import eagle.gaussian.filter
import eagle.points.detector


def moravec(img: numpy.ndarray, size_search: int) -> numpy.ndarray:
    
    if size_search%2 != 1:
        raise AssertionError('search_size must be odd')

    # limit = size_search // 2

    N, M = img.shape
    response = numpy.zeros(img.shape)

    for i in range(size_search, N-size_search):
        for j in range(size_search, M-size_search):
            response[i, j] = eagle.points.detector.moravec(img, i, j, size_search)

    # Remove negative value
    response[response < 0.] = 0.

    return response


def kitchen_rosenfeld(img: numpy.ndarray, sigma: float = numpy.sqrt(2)/2) -> numpy.ndarray:

    gradient = eagle.gaussian.filter.gradient(sigma)
    img_dx = scipy.signal.convolve2d(img, gradient[0], mode='same')
    img_dy = scipy.signal.convolve2d(img, gradient[1], mode='same')

    hessian = eagle.gaussian.filter.hessian(sigma)
    img_dxdx = scipy.signal.convolve2d(img, hessian[0, 0], mode='same')
    img_dxdy = scipy.signal.convolve2d(img, hessian[0, 1], mode='same')
    img_dydx = scipy.signal.convolve2d(img, hessian[1, 0], mode='same')
    img_dydy = scipy.signal.convolve2d(img, hessian[1, 1], mode='same')

    response = eagle.points.detector.kitchen_rosenfeld(img_dx, img_dy, img_dxdx, img_dxdy, img_dydx, img_dydy)
    # Remove negative value
    response[response < 0.] = 0.

    return response 


def beaudet(img: numpy.ndarray, sigma: float = numpy.sqrt(2)/2) -> numpy.ndarray:
    
    hessian = eagle.gaussian.filter.hessian(sigma)
    img_dxdx = scipy.signal.convolve2d(img, hessian[0, 0], mode='same')
    img_dxdy = scipy.signal.convolve2d(img, hessian[0, 1], mode='same')
    img_dydx = scipy.signal.convolve2d(img, hessian[1, 0], mode='same')
    img_dydy = scipy.signal.convolve2d(img, hessian[1, 1], mode='same')
    
    response = eagle.points.detector.beaudet(img_dxdx, img_dxdy, img_dydx, img_dydy)
    # Remove negative value
    response[response < 0.] = 0.

    return response 


def harris(
    img: numpy.ndarray, 
    size_search: int,
    gaussian_weight_params: tuple[int, int],
    method: str,
    sigma: float = numpy.sqrt(2)/2
) -> numpy.ndarray:

    if method == 'harris_plessey':
        fmethod = eagle.points.detector.harris_plessey
    elif method == 'noble':
        fmethod = eagle.points.detector.noble
    elif method == 'shi_tomasi':
        fmethod = eagle.points.detector.shi_tomasi
    else:
        raise AssertionError('method must be odd')

    if size_search%2 != 1:
        raise AssertionError('search_size must be odd')

    limit = size_search // 2

    N, M = img.shape
    response = numpy.zeros(img.shape)
    
    gradient = eagle.gaussian.filter.gradient(sigma)
    img_dx = scipy.signal.convolve2d(img, gradient[0], mode='same')
    img_dy = scipy.signal.convolve2d(img, gradient[1], mode='same')

    img_dx2 = img_dx**2
    img_dy2 = img_dy**2

    w = numpy.random.normal(
        gaussian_weight_params[0],
        gaussian_weight_params[1],
        size=(size_search, size_search)
    )

    Mij = numpy.zeros(shape=(2, 2))

    for i in range(limit, N-limit):
        for j in range(limit, M-limit):
            Mij[0, 0] =  numpy.sum(w *(img_dx2[i-limit:i+limit+1, j-limit:j+limit+1]))
            Mij[1, 1] =  numpy.sum(w *(img_dy2[i-limit:i+limit+1, j-limit:j+limit+1]))
            Mij[0, 1] =  numpy.sum(w * img_dx[i-limit:i+limit+1, j-limit:j+limit+1] * img_dy[i-limit:i+limit+1, j-limit:j+limit+1])
            Mij[1, 0] = Mij[0, 1]
            response[i, j] = fmethod(Mij)
            # response[i, j] = eagle.points.detector.harris_plessey(Mij, k=0.04)
            # response[i, j] = eagle.points.detector.noble(Mij, epsilon=1e-15)
            # response[i, j] = eagle.points.detector.shi_tomasi(Mij)

    # Remove negative value
    response[response < 0.] = 0.

    return response


def remove_non_maxima(response: numpy.ndarray, size_neigh: int) -> numpy.ndarray:
    
    limit = size_neigh // 2
    response_p = response.copy()
    N, M = response_p.shape

    for i in range(limit, N-limit):
        for j in range(limit, M-limit):
            if response[i, j] != numpy.max(response[i-limit:i+limit+1, j-limit:j+limit+1]):
                response_p[i, j] = 0

    return response_p

# TODO:
# def select_best(response: numpy.ndarray, k: int) -> tuple[numpy.ndarray, numpy.ndarray]:
#     pass

# def select_with_threshold(response: numpy.ndarray, threshold: int) -> tuple[numpy.ndarray, numpy.ndarray]:
#     return numpy.where(threshold < response)

def select_with_threshold(response: numpy.ndarray, epsilon: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    value_max = numpy.max(response)
    threshold = epsilon*value_max
    return numpy.where(threshold < response)


    