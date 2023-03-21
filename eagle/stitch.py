from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
from itertools import combinations
from typing import Generic, Iterator, Protocol, TypeAlias, TypeVar
from eagle.points.descriptor.differential import DifferentialInvariant

import eagle.points.response

_T = TypeVar("_T", covariant=True)
arr: TypeAlias = np.ndarray[tuple[int], np.dtype[_T]]
image: TypeAlias = np.ndarray[tuple[int, int], np.dtype[_T]]
vec: TypeAlias = image[_T]
floatlike: TypeAlias = float | vec[float]


class Detector(Protocol[_T]):
    def __call__(self, image: image[_T]) -> image[float]:
        ...


@dataclass
class Beaudet(Detector[float]):
    sigma = np.sqrt(2) / 2

    def __call__(self, image: image[float]) -> image[float]:
        return eagle.points.response.beaudet(image, self.sigma)


@dataclass
class KitchenRosenfeld(Detector[float]):
    sigma = np.sqrt(2) / 2

    def __call__(self, image: image[float]) -> image[float]:
        return eagle.points.response.kitchen_rosenfeld(image, self.sigma)


class Metric(Protocol[_T]):
    def prepare(self, ix: int, image: image[_T]) -> None:
        ...

    def __call__(self, a: vec[int], b: vec[int], ima: int, imb: int) -> floatlike:
        ...


@dataclass
class EuclideanDifferentialInvariant(Metric[_T]):
    sigma = np.sqrt(2) / 2
    differentials: dict[int, DifferentialInvariant] = field(init=False, default_factory=dict)

    def prepare(self, ix: int, image: image[_T]) -> None:
        if ix not in self.differentials:
            self.differentials[ix] = DifferentialInvariant(image, self.sigma)
        
    def __call__(self, a: vec[int], b: vec[int], ima: int, imb: int) -> floatlike:
        da = self.differentials[ima]
        db = self.differentials[imb]

        pa = np.array([da[i] for i in a])
        pb = np.array([db[i] for i in b])

        return np.sqrt((pa-pb)**2)


@dataclass
class Stitch(Generic[_T]):
    source_images: list[image[_T]]
    detector: Detector
    cp_metric: Metric = EuclideanDifferentialInvariant()
    window_size: int = 32
    max_control_points: int = 50
    # select_threshold: float = 0.1

    def __call__(self) -> Iterator[image[float]]:
        control_points: dict[int, vec[int]] = dict()
        for i, image in enumerate(self.source_images):
            print("control points", i)
            points = self.detect_control_points(image)
            control_points[i] = points

        for i1, i2, scores in self._build_match_scores(control_points):
            print(score)

    def detect_control_points(self, image: image[_T]) -> image[int]:
        response = self.detector(image)
        response_sparse = eagle.points.response.remove_non_maxima(
            response, self.window_size
        )
        # control_points = eagle.points.response.select_with_threshold(response_sparse, self.select_threshold)
        control_points = eagle.points.response.select_best(
            response_sparse, self.max_control_points
        )
        return control_points.astype(int)

    def _build_match_scores(
        self, control_points: dict[int, vec[int]]
    ) -> Iterator[tuple[int, int, vec[float]]]:
        for i1, i2 in combinations(control_points.keys(), 2):
            self.cp_metric.prepare(i1, self.source_images[i1])
            self.cp_metric.prepare(i2, self.source_images[i2])

            cpi = control_points[i1]
            cpj = control_points[i2]
            metric = self.cp_metric(cpi, cpj, i1, i2)
            yield i1, i2, metric
