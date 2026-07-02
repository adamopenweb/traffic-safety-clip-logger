"""De-warp a frame with the pipeline's Undistorter and overlay a grid.

Usage: python scripts/dewarp_grid.py <src.jpg> <out.jpg> [k1]

Used to eyeball calibration / camera leveling in the same de-warped space the
analyzer operates in (straight lines should run along the gridlines).
"""
import sys

import cv2

from traffic_logger.analyze.undistort import Undistorter

src = sys.argv[1]
out = sys.argv[2]
k1 = float(sys.argv[3]) if len(sys.argv) > 3 else -0.35
roll = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0

img = cv2.imread(src)
img = Undistorter(k1, roll_degrees=roll)(img)
h, w = img.shape[:2]
for step, color in ((50, (0, 0, 255)), (100, (0, 255, 255))):
    for x in range(0, w, step):
        cv2.line(img, (x, 0), (x, h), color, 1)
    for y in range(0, h, step):
        cv2.line(img, (0, y), (w, y), color, 1)
cv2.imwrite(out, cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR))
print(out)
