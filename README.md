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

### Chessboard Detection Steps Visualization

#### 0 - Example Calibration Image
<p align="center">
  <img width="450" src="https://user-images.githubusercontent.com/22922351/149222640-bab43edf-7bd8-4f40-97da-67cc31f499de.jpg">
</p>

#### 1 - Harris Corner Detection
<p align="center">
  <img width="450" src="https://user-images.githubusercontent.com/22922351/149222189-6b6f26aa-1693-426f-9aee-3631e692fc03.jpg">
</p>

#### 2 - Multiple Corner Filtering Steps
<p align="center">
  <img width="300" src="https://user-images.githubusercontent.com/22922351/149222299-dcdfd564-e739-4eb8-9b60-98b32c2b7c87.jpg">
  <img width="300" src="https://user-images.githubusercontent.com/22922351/149222371-b65866da-5bc9-4b2f-a935-833596fea611.jpg">
  <img width="300" src="https://user-images.githubusercontent.com/22922351/149222397-42bbb09b-18d4-4fa6-be39-213af311616d.jpg">
</p>

#### 3 - Approximate Quadrilateral Hull
<p align="center">
  <img width="450" src="https://user-images.githubusercontent.com/22922351/149222750-a1053e12-1b63-4445-b6d7-acd6619dd7f7.jpg">
</p>

#### 4 - Corners Ordering
<p align="center">
  <img width="450" src="https://user-images.githubusercontent.com/22922351/149222838-73fee391-c4c2-43f9-87b6-e4cd9dd83bdc.jpg">
</p>

### Remarks

More details can be found in `report.pdf`. 

The algorithms implemented are not robust and should only be used for academic purposes.
