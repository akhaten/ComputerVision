import numpy
import scipy.spatial.distance


def minkowski(v1: numpy.ndarray, v2:numpy.ndarray, p: float) -> float:
    scipy.spatial.distance.minkowski(v1, v2, p)

def manhattan(v1: numpy.ndarray, v2:numpy.ndarray) -> float:
    return scipy.spatial.distance.minkowski(v1, v2, 1)

def euclidean(v1: numpy.ndarray, v2:numpy.ndarray) -> float:
    return scipy.spatial.distance.euclidean(v1, v2)

def chebyshev(v1: numpy.ndarray, v2:numpy.ndarray) -> float:
    return scipy.spatial.distance.chebyshev(v1, v2)

def mahalanobis(v1: numpy.ndarray, v2:numpy.ndarray) -> float:
    return scipy.spatial.distance.mahalanobis(v1, v2)

def hamming(v1: numpy.ndarray, v2:numpy.ndarray) -> float:
    return scipy.spatial.distance.hamming(v1, v2)

def jaccard(v1: numpy.ndarray, v2:numpy.ndarray) -> float:
    return scipy.spatial.distance.jaccard(v1, v2)

def cosine(h1: numpy.ndarray, h2: numpy.ndarray) -> float:
    return scipy.spatial.distance.cosine(h1, h2)

def chi2(h1: numpy.ndarray, h2: numpy.ndarray) -> float:
    diff = h1-h2
    summ = h1+h2
    return (1/2) * (numpy.sum(diff**2)/numpy.sum(summ))

def intersection(h1: numpy.ndarray, h2: numpy.ndarray) -> float:
    return numpy.sum(numpy.min(numpy.array([h1, h2]), axis=0))




# def euclidean(v1: numpy.ndarray, v2: numpy.ndarray) -> float:
#     return numpy.sqrt(numpy.sum(numpy.power(v1-v2, 2)))

# def chebyshev(v1:numpy.ndarray, v2: numpy.ndarray) -> float:
#     return numpy.max(v1-v2)

# def minkowski(p: int | float, v1: numpy.ndarray, v2: numpy.ndarray) -> float:

#     if p == numpy.inf:
#         return chebyshev(v1, v2)
#     elif p == 2:
#         return euclidean(v1, v2)
    
#     return numpy.power(numpy.sum(numpy.power(numpy.abs(v1-v2), p)), 1/p)


# def mahalanobis(v1: numpy.ndarray, v2: numpy.ndarray) -> float:
#     cov_mat = numpy.cov(v1, v2)
#     diff: numpy.ndarray = v1-v2
#     return numpy.sqrt(diff.T @ numpy.linalg.inv(cov_mat) @ diff)


# def hamming(h1: numpy.ndarray, h2: numpy.ndarray) -> float:
#     pass    