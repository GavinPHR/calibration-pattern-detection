import logging

import cv2
import numpy as np
from scipy.spatial import ConvexHull

LOGGER = logging.getLogger('my_logger')

Point = np.ndarray  # (2,)
Points = np.ndarray  # (N, 2)


def approx_quadrilateral_hull(corners: Points) -> Points:
    hull = ConvexHull(corners)
    vertex_idx = hull.vertices  # counter-clockwise
    polygon = None
    for epsilon in range(1, 10):
        polygon = cv2.approxPolyDP(corners[vertex_idx], epsilon=epsilon, closed=True)
        if len(polygon) == 4:  # Achieved quadrilateral
            break
    if len(polygon) != 4:
        LOGGER.debug(f'Could not find a approximating quadrilateral hull.')
        return np.array([])
    polygon = polygon.squeeze()

    # Roll the quadrilatral points, with top-left first.
    # Assuming top-left point is closest to the origin,
    # an assumption that CAN EASILY break things.
    min_dist = float('inf')
    min_index = None
    for i, p in enumerate(polygon):
        dist = np.linalg.norm(p)
        if dist < min_dist:
            min_dist = dist
            min_index = i
    return np.roll(polygon, -min_index, axis=0)


class ChessboardFiller:
    def __init__(self, size, corners: Points, quadrilateral: Points):
        """Fills a chessboard grid with all the corners in the correct order.

        It is required that the given corners are exactly the correct ones (but unordered).
        And that we know the quadrilateral that (approximately) enclose them, where
        the quadrilateral points are the top-left, bottom-left, bottom-right, top-right corners IN ORDER.
        """
        # Initialize a chessboard to have all coordinates being (-1, -1)
        self.chessboard = np.zeros((size[0], size[1], 2), dtype=np.int32) - 1
        self.corners = corners
        self.quad = quadrilateral

    @staticmethod
    def _get_line_parameters(p1: Point, p2: Point, c=1) -> np.ndarray:
        """Returns arameter of a line a, b.

        A line is parameterized as ax + by = c (where we set c to some constant).
        Given 2 points, return a, b.
        """
        A = np.array([p1, p2])
        b = np.array([c, c])
        return np.linalg.solve(A, b)

    def _get_corners_on_line(self, a: float, b: float, epsilon: float, c: float = 1) -> Points:
        """Return corners that are (approximately) on a line parameterized by a, b"""
        res = []
        for p in self.corners:
            val = a * p[0] + b * p[1]
            if abs(val - c) < epsilon:
                res.append(p)
        return np.array(res)

    def _get_corners_between_points(self, p1: Point, p2: Point, expected_count: int):
        a, b = self._get_line_parameters(p1, p2)
        for eps in np.arange(0.01, 0.07, 0.01):
            corners = self._get_corners_on_line(a, b, eps)
            if len(corners) == expected_count:
                return corners
        LOGGER.debug(f'Exhausted epsilon range and could not find {expected_count} corners between {p1} and {p2}.')
        return np.array([])

    @staticmethod
    def _sort(points: Points, key_col) -> Points:
        order = np.argsort(points[:, key_col])
        return points[order]

    def _fill_outer(self):
        """Fills the perimeter of the chessboard."""
        h, w, _ = self.chessboard.shape

        # top-left -> bottom-left
        tl, bl = self.quad[0:2]
        corners = self._get_corners_between_points(tl, bl, expected_count=h)
        if len(corners) == 0:
            return
        corners = self._sort(corners, 0)
        for i in range(h):
            self.chessboard[i][0] = corners[i]

        # bottom-left -> bottom-right
        bl, br = self.quad[1:3]
        corners = self._get_corners_between_points(bl, br, expected_count=w)
        if len(corners) == 0:
            return
        corners = self._sort(corners, 1)
        for j in range(w):
            if self.chessboard[-1][j][0] == -1:
                self.chessboard[-1][j] = corners[j]
            elif not (self.chessboard[-1][j] == corners[j]).all():  # filled, check equality instead
                LOGGER.debug(f'Position ({-1}, {j}) filled with {self.chessboard[-1][j]},'
                             f' now trying to fill {corners[j]}')
                return

        # top-right -> bottom-right
        br, tr = self.quad[2:4]
        corners = self._get_corners_between_points(br, tr, expected_count=h)
        if len(corners) == 0:
            return
        corners = self._sort(corners, 0)
        for i in range(h):
            if self.chessboard[i][-1][0] == -1:
                self.chessboard[i][-1] = corners[i]
            elif not (self.chessboard[i][-1] == corners[i]).all():  # filled, check equality instead
                LOGGER.debug(f'Position ({i}, {-1}) filled with {self.chessboard[i][-1]},'
                             f' now trying to fill {corners[i]}')
                return

        # top-left -> top-right
        tl, tr = self.quad[[0, 3]]
        corners = self._get_corners_between_points(tl, tr, expected_count=w)
        if len(corners) == 0:
            return
        corners = self._sort(corners, 1)
        for j in range(w):
            if self.chessboard[0][j][0] == -1:
                self.chessboard[0][j] = corners[j]
            elif not (self.chessboard[0][j] == corners[j]).all():  # filled, check equality instead
                LOGGER.debug(f'Position ({0}, {j}) filled with {self.chessboard[0][j]},'
                             f' now trying to fill {corners[j]}')
                return

    def _check_outer_filled(self) -> bool:
        h, w, _ = self.chessboard.shape
        for i in range(h):
            if self.chessboard[i][0][0] == -1 or self.chessboard[i][-1][0] == -1:
                return False
        for j in range(w):
            if self.chessboard[0][j][0] == -1 or self.chessboard[-1][j][0] == -1:
                return False
        return True

    def _check_all_filled(self) -> bool:
        h, w, _ = self.chessboard.shape
        for i in range(h):
            for j in range(w):
                if self.chessboard[i][j][0] == -1:
                    return False
        return True

    def _clear_chessboard_center(self):
        h, w, _ = self.chessboard.shape
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                self.chessboard[i][j][0] = -1
                self.chessboard[i][j][1] = -1

    def fill(self) -> bool:
        self._fill_outer()
        if not self._check_outer_filled():
            LOGGER.debug(f'Chessboard perimeter not filled, aborted.')
            return False

        # Try filling each horizontal lines form top to down
        h, w, _ = self.chessboard.shape
        try:
            for i in range(1, h - 1):
                left, right = self.chessboard[i, [0, -1]]
                corners = self._get_corners_between_points(left, right, expected_count=w)
                assert len(corners) != 0
                corners = self._sort(corners, 1)
                for j in range(1, w - 1):
                    self.chessboard[i][j] = corners[j]
        except AssertionError as e:
            LOGGER.debug(e)

        if not self._check_all_filled():
            self._clear_chessboard_center()
        else:
            return True

        # If horizontal failed, try filling each vertical lines form left to right instead
        try:
            for j in range(1, w - 1):
                top, bottom = self.chessboard[[0, -1], j]
                corners = self._get_corners_between_points(top, bottom, expected_count=h)
                assert len(corners) != 0
                corners = self._sort(corners, 0)
                for i in range(1, h - 1):
                    self.chessboard[i][j] = corners[i]
        except AssertionError as e:
            LOGGER.debug(e)
            return False
        return True

    def get(self, flat=True):
        """Returns the chessboard as a (h*w, 2) array if flat=True, otherwise a (h, w, 2) array."""
        if flat:
            h, w, _ = self.chessboard.shape
            return self.chessboard.reshape(h * w, -1).astype(np.float32)
        return self.chessboard.astype(np.float32)
