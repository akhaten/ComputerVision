import numpy

def moravec(img: numpy.ndarray, i: int, j: int, size_search: int) -> float:

    N, M = img.shape
    limit = size_search // 2

    if size_search%2 != 1:
        raise AssertionError('search_size must be odd')
        
    if (i-limit < 0) or (N-1-limit < i+1) or (j-limit < 0) or (M-1-limit < j+1):
        raise AssertionError('Bord condition : (i, j) = ({}, {})'.format(i, j))

    def SSD(i, j, u, v):
        return numpy.sum((img[i-limit:i+limit+1, j-limit:j+limit+1] - img[u-limit:u+limit+1, v-limit:v+limit+1])**2)
    
    ssds: list = []

    for k in range(-limit, limit+1):
        for l in range(-limit, limit+1):
            # Problème de condition de bord
            if (k != 0) and (l != 0):
                ssds.append(SSD(i, j, i+k, j+l))
            

    return numpy.min(numpy.array(ssds))


def kitchen_rosenfeld(
    img_dx: numpy.ndarray, 
    img_dy: numpy.ndarray, 
    img_dxdx: numpy.ndarray, 
    img_dxdy: numpy.ndarray, 
    img_dydx: numpy.ndarray, 
    img_dydy: numpy.ndarray
) -> numpy.ndarray:

    img_x2 = img_dx**2
    img_y2 = img_dy**2

    a = img_dxdx*img_y2 + img_dydy*img_x2 - img_dxdy*img_dx*img_dy - img_dydx*img_dx*img_dy
    b = img_x2 + img_y2

    out = numpy.empty_like(b, dtype=numpy.float64)
    mask = (b == 0)
    out[mask] = 0.0
    out[~mask] = a[~mask]/b[~mask]

    return out

def beaudet(
    img_xx: numpy.ndarray, 
    img_xy: numpy.ndarray, 
    img_yx: numpy.ndarray, 
    img_yy: numpy.ndarray
) -> numpy.ndarray:
    return numpy.abs(img_xx*img_yy-img_xy*img_yx)



# For Harris response
def harris_plessey(Mij: numpy.ndarray, k: float = 0.04) -> float:
    # eigen_values = numpy.linalg.eigvals(Mij)
    a, b, _, c = Mij[0, 0], Mij[0, 1], Mij[1, 0], Mij[1, 1]
    lambda1: float = 0.5 * (a + c - numpy.sqrt((a-c)**2 + 4*b**2))
    lambda2: float = 0.5 * (a + c + numpy.sqrt((a-c)**2 + 4*b**2))
    detM: float = lambda1 * lambda2
    trM: float = lambda1 + lambda2
    return detM - k * trM

def noble(Mij: numpy.ndarray, epsilon: float = 1e-15) -> float:
    # eigen_values = numpy.linalg.eigvals(Mij)
    a, b, _, c = Mij[0, 0], Mij[0, 1], Mij[1, 0], Mij[1, 1]
    lambda1: float = 0.5 * (a + c - numpy.sqrt((a-c)**2 + 4*b**2))
    lambda2: float = 0.5 * (a + c + numpy.sqrt((a-c)**2 + 4*b**2))
    detM: float = lambda1 * lambda2
    trM: float = lambda1 + lambda2
    return 2 * detM / (trM + epsilon)

def shi_tomasi(Mij: numpy.ndarray) -> float:
    # eigen_values = numpy.linalg.eigvals(Mij)
    a, b, _, c = Mij[0, 0], Mij[0, 1], Mij[1, 0], Mij[1, 1]
    lambda1: float = 0.5 * (a + c - numpy.sqrt((a-c)**2 + 4*b**2))
    return lambda1



# def harris_multi_scaling():
#     pass

# def harris_laplace_detector():
#     pass

# def sift_detector():
#     # IN OpenCV
#     pass