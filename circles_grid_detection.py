import logging
import sys
from typing import Tuple

import numpy as np
import cv2

from common import Points, approx_quadrilateral_hull, ChessboardFiller
import utils

LOGGER = logging.getLogger('my_logger')


def find_blobs(image: np.ndarray, visualize_blobs=False) -> Points:
    blob_params = cv2.SimpleBlobDetector_Params()
    blob_params.filterByArea = True
    blob_params.minArea = 30 ** 2
    blob_params.maxArea = 150 ** 2
    blob_detector = cv2.SimpleBlobDetector_create(blob_params)
    keypoints = blob_detector.detect(image)
    corners = np.array([keypoint.pt for keypoint in keypoints]).astype(np.int32)
    corners = corners[:, [1, 0]]
    if visualize_blobs:
        image = cv2.drawKeypoints(image, keypoints, np.array([]),
                                  (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        utils.visualize_keypoints(image, corners)
    return corners


def find_circles_grid(image: np.ndarray,
                      pattern_size: Tuple[int, int],  # (height, width)
                      visualize=False) -> Tuple[bool, np.ndarray]:
    corners = find_blobs(image, visualize_blobs=True if visualize else False)
    if len(corners) != pattern_size[0] * pattern_size[1]:
        return False, np.array([])

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
    images, paths = utils.read_images('circles_grid_patterns/calibration-*.jpg')
    results = []
    success_cnt = 0
    visualize = True if len(sys.argv) == 2 and sys.argv[1] == 'visualize' else False
    for i, image in enumerate(images):
        retval, corners = find_circles_grid(image, (7, 9), visualize)
        if retval:
            results.append(corners)
            success_cnt += 1
            LOGGER.info(f'find_circles_grid succeeded for {paths[i]}')
        else:
            LOGGER.info(f'find_circles_grid failed for {paths[i]}')
            find_circles_grid(image, (7, 9), visualize=True)
    LOGGER.info(f'find_circles_grid succeeded for {success_cnt}/{len(images)} images')
