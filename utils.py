import glob
from copy import deepcopy

import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200


def read_images(dirname, grayscale=False):
    images = []
    paths = []
    for fpath in glob.glob(dirname):
        images.append(cv2.imread(fpath, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR))
        paths.append(fpath)
    return images, paths


def show(image):
    if image.shape[-1] == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif image.shape[-1] == 1:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    else:
        raise RuntimeError('Image should be 1 or 3 channels BGR')
    plt.axis('off')
    plt.show()


def visualize_keypoints(image, points, radius=5, no_show=False):
    image = deepcopy(image)
    for point in points:
        cv2.circle(image, (point[1], point[0]), radius, (0, 0, 255), -1)
    if no_show:
        return image
    show(image)


def visualize_hull(image, quad_vertices, corners=()):
    """Assume the points are arranged in order and that the first and last are connected."""
    image = deepcopy(image)
    points = quad_vertices[:, [1, 0]]
    points = points.reshape((-1, 1, 2))
    cv2.polylines(image, [points], True, (255, 0, 0), 3)
    if len(corners) != 0:
        image = visualize_keypoints(image, corners, no_show=True)
    show(image)


def visualize_chessboard(image, size, corners):
    image = deepcopy(image)
    corners = corners[:, [1, 0]]
    image = cv2.drawChessboardCorners(image, size, corners, True)
    show(image)
