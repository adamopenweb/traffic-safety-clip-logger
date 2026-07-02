"""Throwaway tuner: try cv2 radial-undistortion k1 values on a frame.

Draws a grid on the undistorted result so we can pick the k1 (OpenCV model)
that straightens the known-straight lines (sidewalk, curbs, hedge).
"""
import sys
import cv2
import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else "samples/cam-day-sub.jpg"
img = cv2.imread(src)
h, w = img.shape[:2]
fx = fy = float(w)  # focal-length assumption; tune k1 against it
cx, cy = w / 2.0, h / 2.0
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

for k1 in (-0.15, -0.25, -0.35, -0.45):
    dist = np.array([k1, 0, 0, 0, 0], dtype=np.float64)
    und = cv2.undistort(img, K, dist)
    for step, color in ((50, (0, 0, 255)), (100, (0, 255, 255))):
        for x in range(0, w, step):
            cv2.line(und, (x, 0), (x, h), color, 1)
        for y in range(0, h, step):
            cv2.line(und, (0, y), (w, y), color, 1)
    big = cv2.resize(und, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
    tag = str(k1).replace("-", "").replace(".", "")
    out = f"samples/cvund_{tag}.jpg"
    cv2.imwrite(out, big)
    print(out)
