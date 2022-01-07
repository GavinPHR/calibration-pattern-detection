from PIL import Image
import os
import math
import logging

import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

from custom_types import *

LOGGER = logging.getLogger('global_logger')


def harris_corner(image: np.ndarray) -> Points:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # magic numbers are blockSize, ksize, k
    response = cv2.cornerHarris(image, 2, 3, 0.04)
    # Filter out weak responses
    # threshold of (0.01 * max) should return more points than the wanted corners
    response = response > 0.01 * response.max()
    # Extract points
    points = list(zip(*np.where(response == True)))
    LOGGER.info(f'Harris corner detector found {len(points)} corners')
    return np.array(points)


def clustering_filter(corners: Points) -> Points:
    dist = distance_matrix(corners, corners)
    kept = set()
    for i in range(len(corners)):
        close = False
        if len(close_point_idx := np.argwhere(dist[i] < 10).flatten()) != 0:
            for j in close_point_idx:
                if j in kept:
                    close = True
                    break
        if not close:
            kept.add(i)
    LOGGER.info(f'Clustering filter: {len(corners)}->{len(kept)} corners')
    return corners[list(kept)]


def connected_component_filter(corners: Points) -> Points:
    graph = distance_matrix(corners, corners) < 30
    # Only keep the components with <2 nodes
    kept = corners[np.sum(graph, axis=0) <= 1]
    LOGGER.info(f'Connected component filter: {len(corners)}->{len(kept)} corners')
    return kept

















