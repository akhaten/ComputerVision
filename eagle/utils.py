import numpy
import numpy.linalg


def hartley_matrix(points: numpy.ndarray) -> numpy.ndarray:

    n = points.shape[0]
    mean = numpy.mean(points, axis=0)
    
    u_mean, v_mean = mean

    center_points = points-mean
    norms = numpy.sqrt(numpy.sum(center_points**2, axis=1))
    a = numpy.sqrt(2) / numpy.mean(norms)

    hartley_m = numpy.array(
        [
            [a, 0, -a*u_mean],
            [0, a, -a*v_mean],
            [0, 0, 1]
        ]
    )

    return hartley_m


def hartley_normalization(points: numpy.ndarray, hartley_m: numpy.ndarray) -> numpy.ndarray:

    a = hartley_m[0][0]
    neg_au_mean = hartley_m[0][2]
    neg_av_mean = hartley_m[1][2]
    
    u, v = points[:, 0], points[:, 1]

    u_hartley = a*u + neg_au_mean
    v_hartley = a*v + neg_av_mean

    points = numpy.array([u_hartley, v_hartley]).T

    return points


def homography_estimation(points_1: numpy.ndarray, points_2: numpy.ndarray, normalization: bool = False) -> numpy.ndarray:
    assert points_1.shape[1] == points_2.shape[1] == 2, "Points must be a row list of 2D points"
    assert points_1.shape[0] == points_2.shape[0], "Points must have the same length to be matched"
    pts1 = numpy.copy(points_1)
    pts2 = numpy.copy(points_2)

    # Hartey Normailzation

    hartley_m1 = None
    hartley_m2 = None

    if normalization:
        hartley_m1 = hartley_matrix(pts1)
        # print("hartley_m1 =", hartley_m1)
        hartley_m2 = hartley_matrix(pts2)
        pts1 = hartley_normalization(pts1, hartley_m1)
        # print("pts1 (hartley) =", pts1[0:2])
        pts2 = hartley_normalization(pts2, hartley_m2)
    
    
    # Total Least Square

    nb_points = pts1.shape[0]

    # u, v, up, vp = pts1[:, 0], pts1[:, 1], pts2[:, 0], pts2[:, 1]

    D = numpy.zeros(shape=(2*nb_points, 9))

    # print(u, v)
    for i in range(0, nb_points):
        
        u, v = pts1[i]
        up, vp = pts2[i]
        # print(2*i,2*i+1)


        D[2*i:2*(i+1)] = numpy.array(
            [
                [ u, v, 1, 0, 0, 0, -up*u, -up*v, -up ],
                [ 0, 0, 0, u, v, 1, -vp*u, -vp*v, -vp ]
            ]
        )

    # print("D =", D[0:3, 0:3])
    DTD = D.T @ D
    # print("D.T @ D =", DTD[0:3, 0:3])
    eigen_values, eigen_vectors = numpy.linalg.eig(DTD)
    indmin = numpy.argmin(eigen_values)
    v = eigen_vectors[:, indmin]

    # print(indmin)
    
    h = numpy.reshape(-v, newshape=(3, 3))

    # Denormalization

    if normalization:
        h = numpy.linalg.inv(hartley_m2) @ h @ hartley_m1

    return h