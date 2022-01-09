import logging
import sys
from collections import Counter
from typing import List, Tuple

import numpy as np
from scipy.spatial import distance_matrix
import cv2

from common import Point, Points, approx_quadrilateral_hull, ChessboardFiller
import utils

LOGGER = logging.getLogger('my_logger')


def harris_corner(image: np.ndarray) -> Points:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # magic numbers are blockSize, ksize, k
    response = cv2.cornerHarris(image, 2, 3, 0.04)
    # Filter out weak responses
    # threshold of (0.01 * max) should return more points than the wanted corners
    response = response > 0.01 * response.max()
    # Extract points
    points = list(zip(*np.where(response == True)))
    LOGGER.debug(f'Harris corner detector found {len(points)} corners')
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
    LOGGER.debug(f'Clustering filter: {len(corners)}->{len(kept)} corners')
    return corners[list(kept)]


def connected_component_filter(corners: Points) -> Points:
    graph = distance_matrix(corners, corners) < 30
    # Only keep the components with <2 nodes
    kept = corners[np.sum(graph, axis=0) <= 2]  # <=2 because each node is connected to itself
    LOGGER.debug(f'Connected component filter: {len(corners)}->{len(kept)} corners')
    return kept


class SquareResponseFilter:
    def __init__(self, size=10, half_smoothing_window_size=4):
        self.size = size
        self.half_smoothing_window_size = half_smoothing_window_size

    @staticmethod
    def _OTSU_binarization(image: np.ndarray) -> np.ndarray:
        grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binarized = cv2.threshold(grayscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binarized

    def _majority_vote_smoothing(self, response: List[int]) -> np.ndarray:
        """Smooth a 1D binary array consists of 255 and 0 with majority vote."""
        response = np.array(response)
        k = self.half_smoothing_window_size
        N = len(response)
        new_response = np.zeros(N)
        for i in range(N):
            cnt = Counter(response[np.arange(i - k, i + k + 1) % N])
            new_response[i] = 255 if cnt[255] > cnt[0] else 0
        return new_response

    def _get_square_response(self, image: np.ndarray, point: Point) -> np.ndarray:
        """Iterate through a square centered at keypoint and get a response."""
        assert len(image.shape) == 2, 'Image must be binarized.'
        k = self.size
        # top-left, bottom-right
        tl, br = point - k, point + k
        # top-right, bottom-left
        tr = np.array([point[0] - 10, point[1] + 10])
        bl = np.array([point[0] + 10, point[1] - 10])

        response = []
        for j in range(tl[1], tr[1]):
            response.append(image[tl[0], j])
        for i in range(tr[0], br[0]):
            response.append(image[i, tr[1]])
        for j in range(br[1], bl[1], -1):
            response.append(image[br[0], j])
        for i in range(bl[0], tl[0], -1):
            response.append(image[i, bl[1]])
        return self._majority_vote_smoothing(response)

    @staticmethod
    def _count_segments(response: np.ndarray) -> int:
        """Count how many distinct segments of 0 or 255 exist."""
        prev = response[0]
        cnt = 1
        for cur in response[1:]:
            if cur != prev:
                cnt += 1
            prev = cur
        return cnt

    def filter(self, image: np.ndarray, corners: Points) -> Points:
        """Filter the corners based on square response characteristics.

        An internal corner's (represented by x) surrounding region should look like:
        1110000
        1110000
        111x111
        0000111
        0000111
        Recording the values on a square circumference should give us at least 4 (maybe more)
        distince segments (_count_segments()).
        """
        kept = []
        binarized = self._OTSU_binarization(image)
        for corner in corners:
            try:
                response = self._get_square_response(binarized, corner)
            except IndexError:  # If a point to too close to the border
                continue
            if self._count_segments(response) >= 4:
                kept.append(corner)
        LOGGER.debug(f'Square response filter: {len(corners)}->{len(kept)} corners')
        return np.array(kept)


def sum_distance_filter(corners: Points, target) -> Points:
    to_remove = len(corners) - target
    if to_remove == 0:
        return corners
    sum_dist = np.sum(distance_matrix(corners, corners), axis=0)
    kept = corners[np.argsort(sum_dist)[:-to_remove]]
    LOGGER.debug(f'Sum distance filter: {len(corners)}->{len(kept)} corners')
    return kept


def find_chessboard_corners(image: np.ndarray,
                            pattern_size: Tuple[int, int],  # (height, width)
                            visualize=False) -> Tuple[bool, np.ndarray]:
    target = pattern_size[0] * pattern_size[1]
    corners = harris_corner(image)
    if visualize: utils.visualize_keypoints(image, corners)
    if len(corners) < target: return False, np.array([])

    corners = clustering_filter(corners)
    if visualize: utils.visualize_keypoints(image, corners)
    if len(corners) < target: return False, np.array([])

    corners = connected_component_filter(corners)
    if visualize: utils.visualize_keypoints(image, corners)
    if len(corners) < target: return False, np.array([])

    square_response_filter = SquareResponseFilter()
    corners = square_response_filter.filter(image, corners)
    if visualize: utils.visualize_keypoints(image, corners)
    if len(corners) < target: return False, np.array([])

    corners = sum_distance_filter(corners, target)
    if visualize: utils.visualize_keypoints(image, corners)

    if len(corners) != pattern_size[0] * pattern_size[1]:
        return False, np.ndarray([])

    quad = approx_quadrilateral_hull(corners)
    if len(quad) == 0: return False, np.array([])
    if visualize: utils.visualize_hull(image, quad, corners)

    chessboard_filler = ChessboardFiller(pattern_size, corners, quad)
    if not chessboard_filler.fill():
        return False, np.array([])
    corners = chessboard_filler.get()
    if visualize: utils.visualize_chessboard(image, pattern_size, corners)

    return True, corners


if __name__ == '__main__':
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(logging.StreamHandler(stream=sys.stdout))
    images, paths = utils.read_images('chessboard_patterns/calibration-*.png')
    results = []
    success_cnt = 0
    for i, image in enumerate(images):
        retval, corners = find_chessboard_corners(image, (7, 9))
        if retval:
            results.append(corners)
            success_cnt += 1
            LOGGER.info(f'find_chessboard_corners succeeded for {paths[i]}')
        else:
            LOGGER.info(f'find_chessboard_corners failed for {paths[i]}')
            # find_chessboard_corners(image, (7, 9), visualize=True)
    LOGGER.info(f'find_chessboard_corners succeeded for {success_cnt}/{len(images)} images')
