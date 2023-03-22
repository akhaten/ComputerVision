from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
from itertools import combinations
import typing as t
from typing import Generic, Iterator, Protocol, TypeAlias, TypeVar
from eagle.points.descriptor.differential import DifferentialInvariant
import matplotlib.pyplot as plt

import eagle.points.response
import eagle.utils

if t.TYPE_CHECKING:
    _T = TypeVar("_T", covariant=True, bound=np.generic)
    _T_co = TypeVar("_T_co", contravariant=True, bound=np.generic)
else:
    _T = TypeVar("_T")
    _T_co = TypeVar("_T_co", contravariant=True)

arr: TypeAlias = np.ndarray[tuple[int], np.dtype[_T]]
image: TypeAlias = np.ndarray[tuple[int, int], np.dtype[_T]]
vec: TypeAlias = image[_T]


class Detector(Protocol[_T_co]):
    def __call__(self, image: image[_T_co]) -> image[np.float_]:
        ...


@dataclass
class Beaudet(Detector[np.float_]):
    sigma = np.sqrt(2) / 2

    def __call__(self, image: image[np.float_]) -> image[np.float_]:
        return eagle.points.response.beaudet(image, self.sigma)


@dataclass
class KitchenRosenfeld(Detector[np.float_]):
    sigma = np.sqrt(2) / 2

    def __call__(self, image: image[np.float_]) -> image[np.float_]:
        return eagle.points.response.kitchen_rosenfeld(image, self.sigma)


class Metric(Protocol[_T_co]):
    def prepare(self, ix: int, image: image[_T_co]) -> None:
        ...

    def __call__(self, a: vec[np.int_], b: vec[np.int_], ima: int, imb: int) -> vec[np.float_]:
        ...


@dataclass
class EuclideanDifferentialInvariant(Metric[_T_co]):
    sigma = np.sqrt(2) / 2
    differentials: dict[int, DifferentialInvariant] = field(init=False, default_factory=dict)

    def prepare(self, ix: int, image: image[_T]) -> None:
        if ix not in self.differentials:
            self.differentials[ix] = DifferentialInvariant(image, self.sigma)
        
    def __call__(self, a: vec[np.int_], b: vec[np.int_], ima: int, imb: int) -> vec[np.float_]:
        da = self.differentials[ima]
        db = self.differentials[imb]

        pa = np.array([da[i] for i in a])
        pb = np.array([db[i] for i in b])

        return np.sqrt((pa-pb)**2)


def merge(img1: np.ndarray, img2: np.ndarray, homography: np.ndarray) -> np.ndarray:

    """
    Parameters:
        - img1: image
        - img2: image
        - homography : homography to tranfert pixel from img1 to img2
    """

    nb_rows, nb_cols = img1.shape

    corners = np.transpose(
        np.array(
            [
                np.array([0, 0, 1]),
                np.array([0, nb_cols, 1]),
                np.array([nb_rows, 0, 1]),
                np.array([nb_rows, nb_cols, 1])
            ]
        )
    )

    corners_out = homography @ corners

    for i in range(0, 4):
        corners_out[:, i] /= corners_out[2, i]

    row_min = np.minimum(0, np.min(corners_out[0:2, :][0, :]))
    row_max = np.maximum(img2.shape[0], np.max(corners_out[0:2, :][0, :]))
    col_min = np.minimum(0, np.min(corners_out[0:2, :][1, :]))
    col_max = np.maximum(img2.shape[1], np.max(corners_out[0:2, :][1, :]))

    nb_rows_out, nb_cols_out = int(np.ceil(row_max)), int(np.ceil(col_max))

    T = np.array(
        [
            [ 1, 0, -row_min ],
            [ 0, 1, -col_min ],
            [ 0, 0, 1],
        ]
    )

    THinv = np.linalg.inv(T @ homography)

    result = np.zeros(shape=(nb_rows_out, nb_cols_out))


    for i_out in range(0, nb_rows_out):

        for j_out in range(0, nb_cols_out):

            coord_out_homgeneous = np.array([ i_out, j_out, 1 ]).T

            coord_initiale = (THinv @ coord_out_homgeneous).T
            i = int(np.round(coord_initiale[0] / coord_initiale[2]))
            j = int(np.round(coord_initiale[1] / coord_initiale[2]))
            

            if ( (0 <= i) and (i < nb_rows) ) and ( (0 <= j) and (j < nb_cols) ) :
                result[i_out, j_out] = img1[i, j]
            elif (0 <= i_out) and (i_out < img2.shape[0]) and (0 <= j_out) and (j_out < img2.shape[1]):
                result[i_out, j_out] = img2[i_out,j_out]

    return result

@dataclass
class Stitch(Generic[_T]):
    source_images: tuple[image[_T], image[_T]]
    detector: Detector
    cp_metric: Metric = EuclideanDifferentialInvariant()
    window_size: int = 32
    max_control_points: int = 50
    # select_threshold: float = 0.1

    def __call__(self) -> image[np.float_]:
        # self.cp_metric.prepare(0, self.source_images[0])
        # self.cp_metric.prepare(1, self.source_images[1])

        print("== Control points")
        cpa = self.detect_control_points(self.source_images[0])
        cpb = self.detect_control_points(self.source_images[1])
        print("== Done")

        fig = plt.figure()
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(self.source_images[0], cmap="gray")
        ax.scatter(cpa[:,1], cpa[:,0], marker="x", color="green")
        fig.savefig("cpa.png")

        fig = plt.figure()
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(self.source_images[1], cmap="gray")
        ax.scatter(cpb[:,1], cpb[:,0], marker="x", color="green")
        fig.savefig("cpb.png")

        # scores = self.cp_metric(cpa, cpb, 0, 1)
        cpn = np.minimum(cpa.shape[0], cpb.shape[0])
        homography = eagle.utils.homography_estimation(cpa[:cpn], cpb[:cpn], normalization=True)
        print(homography)
        print("== Stitching")
        return self._stitch_pair(homography, 0, 1)

    def detect_control_points(self, image: image[_T]) -> image[np.int_]:
        response = self.detector(image)
        response_sparse = eagle.points.response.remove_non_maxima(
            response, self.window_size
        )
        control_points = eagle.points.response.select_with_threshold(response_sparse, 0.8)
        # control_points = eagle.points.response.select_best(
        #     response_sparse, self.max_control_points
        # )
        return control_points.astype(int)

    def _build_match_scores(
        self, control_points: dict[int, vec[np.int_]]
    ) -> Iterator[tuple[int, int, vec[np.float_]]]:
        for i1, i2 in combinations(control_points.keys(), 2):
            self.cp_metric.prepare(i1, self.source_images[i1])
            self.cp_metric.prepare(i2, self.source_images[i2])

            cpi = control_points[i1]
            cpj = control_points[i2]
            metric = self.cp_metric(cpi, cpj, i1, i2)
            yield i1, i2, metric

    def _stitch_pair(self, h: image[np.float_], i1: int, i2: int) -> image[np.float_]:
        return merge(self.source_images[i1], self.source_images[i2], h)