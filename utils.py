import os
import glob
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200


def read_images(dirname, grayscale=False):
    images = []
    for fpath in glob.glob(dirname):
        images.append(cv2.imread(fpath, cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR))
    return images


def show(image):
    if image.shape[-1] == 3:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    elif image.shape[-1] == 1:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    else:
        raise RuntimeError('Image should be 1 or 3 channels BGR')
    plt.axis('off')
    plt.show()


def visualize_keypoints(image, points, radius=5):
    image = deepcopy(image)
    for point in points:
        cv2.circle(image, (point[1], point[0]), radius, (0, 0, 255), -1)
    show(image)
