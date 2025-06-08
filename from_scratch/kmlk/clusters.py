"""
Generates data for classification models.

The general idea is that each cluster will be some sort of closed shape in the form of:
    a(x-b)^2 + c(y-d)^2 = e
"""

import numpy as np
from random import uniform
from math import sin, cos, pi

def gen_cluster(n: int = 10, a: float = 1, b: float = 1, r: float = 1, variance: float = 0, passes: int = 1) -> np.ndarray:
    """
    Generates a single cluster and returns an array with the resulting points.

    Args:
        n: integer containing the number of points in the cluster
        a: x value of the center coordinate
        b: y value of the center coordinate
        r: radius size
        variance: maximum percentage of error that each point can experience
        passes: number of increments of radius size the algorithm experiences. 
            More passes leads to higher density of points at the center of the cluster
    """

    arr = np.empty((n, 2))

    radius = 0
    num_points = n//passes

    for p in range(passes):
        radius += r/passes

        for i in range(num_points):
            theta = uniform(0, 2*pi)
            p_radius = radius * uniform(0, 1)
            err = 1 + uniform(-variance, variance)

            arr[i] = np.array([(a + p_radius*cos(theta)) * err, (b + p_radius*sin(theta)) * err])

    return arr

def gen_data(centers: list, points_per_cluster: int = 10, min_radius: int = 1, max_radius: int = 10, variance: float = 0, passes: int = 1) -> tuple:
    """
    Generates all the clusters.

    Args:
        centers: list of all center coordinate pairs
        points_per_cluster: number of points for each cluster
        min_radius: minimum radius size
        max_radius: maximum radius size
        variance: maximum percentage of error that each point can experience
        passes: number of increments of radius size the algorithm experiences.
            More passes leads to higher density of points at the center of the cluster
    """

    arr = np.empty((0, 2))
    labels = np.empty(0)

    for c in range(len(centers)):
        arr = np.concatenate((arr, gen_cluster(points_per_cluster, centers[c][0], centers[c][1], uniform(min_radius, max_radius), variance, passes)), axis=0)
        labels = np.concatenate([labels, np.full(points_per_cluster, c)], axis=0)

    return arr, labels
