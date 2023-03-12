import numpy
import numpy.linalg

def least_square(A: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
    return numpy.linalg.inv(A.T @ A) @ A.T @ b


def total_least_square(A: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
    """Ax = b => [A | -b] lambda  [x 1].T
    """
    eigen_values, eigen_vectors = numpy.linalg.eig(A.T @ A)
    indmin = numpy.argmin(eigen_values)
    solution = eigen_vectors[:, indmin]
    # solution /= solution[len(solution)-1]
    return solution
