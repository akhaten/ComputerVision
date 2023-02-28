import numpy

def point_spread_function(sigma: float, x: float, y: float) -> float:
    return 1 / (2*numpy.pi*sigma**2) * numpy.exp(- (x**2 + y**2) / (2*sigma**2))

def dx(sigma: float, x: float, y: float) -> float:
    return (-x / (sigma**2)) * point_spread_function(sigma, x, y)
    # return (-x / (2*numpy.pi*sigma**4)) * numpy.exp(- (x**2 + y**2) / (2*sigma**2))

def dy(sigma: float, x: float, y: float) -> float:
    return -y / (sigma**2) * point_spread_function(sigma, x, y)

def dxdx(sigma: float, x: float, y: float) -> float:
    return ((x**2 / sigma**4) - (1 / sigma**2)) * point_spread_function(sigma, x, y)

def dxdy(sigma: float, x: float, y: float) -> float:
    return (x * y / sigma**4) * point_spread_function(sigma, x, y)

def dydy(sigma: float, x: float, y: float) -> float:
    return ((y**2 / sigma**4) - (1 / sigma**2)) * point_spread_function(sigma, x, y)

def dydx(sigma: float, x: float, y: float) -> float:
    return dxdy(sigma, x, y)