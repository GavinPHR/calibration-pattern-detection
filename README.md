# Calibration Pattern Detection
#### Cambridge MLMI12 Computer Vision Coursework 

This repository contains code for custom calibration pattern detection algorithms. The 4 relevant Python files are:

- `chessboard_detection.py`: Code for detecting chessboard calibration pattern, it contains the function `find_chessboard_corners()` that emulates `cv2.findChessboardCorners()`.
- `circles_grid_detection.py`: Code for detecting symmetric circles grid calibration pattern, it contains the function `find_circles_grid()` that emulates `cv2.findCirclesGrid()`.
- `common.py`: Common code used by both `chessboard_detection.py` and `circles_grid_detection.py`.
- `utils.py`: Convenient functions that load and visualize images.

The files have the following import structure:
<p align="center">
  <img width="350" src="https://user-images.githubusercontent.com/22922351/149175864-3aff4dd8-6b21-4365-9960-c75f9752cabd.png">
</p>

Both `chessboard_detection.py` and `circles_grid_detection.py` are runnable which will process the example calibration images:

```bash
# pip3 install -r requirements.txt
python3 chessboard_detection.py visualize
python3 circles_grid_detection.py visualize
```
Users can optionally append a `visualize` flag, which will visualize the algorithm step-by-step.

### Chessboard Detection Steps


### Remarks

More details can be found in `report.pdf`. 

The algorithms implemented are not robust and should only be used for academic purposes.
