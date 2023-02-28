import numpy
from typing import Callable

import eagle.gaussian.function

def make_filter(sigma: float, fgaussian: Callable[[float, float, float], float], dim: int = None) -> numpy.ndarray:

    d = int(2*numpy.ceil(3*sigma)+1) if dim is None else dim

    filter = numpy.zeros(shape=(d, d), dtype=numpy.float64)
    limit_i, limit_j = filter.shape[0]//2 + 1, filter.shape[1]//2 + 1

    for i in range(-limit_i, limit_i):
        for j in range(-limit_j, limit_j):
            x, y = i+limit_i-1, j+limit_j-1
            filter[x, y] = fgaussian(sigma, i, j)

    return filter


def gradient(sigma: float, dim_filter: int = None) -> numpy.ndarray:

    filter_dx = make_filter(
        sigma = sigma,
        fgaussian = eagle.gaussian.function.dx,
        dim = dim_filter
    )

    filter_dy = make_filter(
        sigma = sigma,
        fgaussian = eagle.gaussian.function.dy,
        dim = dim_filter
    )
   
    return numpy.array([filter_dx, filter_dy])


def hessian(sigma: float, dim_filter: int = None) -> numpy.ndarray:

    filter_dxdx = make_filter(
        sigma = sigma,
        fgaussian = eagle.gaussian.function.dxdx,
        dim = dim_filter
    )

    filter_dxdy = make_filter(
        sigma = sigma,
        fgaussian = eagle.gaussian.function.dxdy,
        dim = dim_filter
    )

    # TODO : optmization -> dxdy == dydx because symetric
    filter_dydx = make_filter(
        sigma = sigma,
        fgaussian = eagle.gaussian.function.dydx,
        dim = dim_filter
    )

    filter_dydy = make_filter(
        sigma = sigma,
        fgaussian = eagle.gaussian.function.dydy,
        dim = dim_filter
    )
   
    return numpy.array(
        [
            [filter_dxdx, filter_dxdy],
            [filter_dydx, filter_dydy]
        ]
    )
