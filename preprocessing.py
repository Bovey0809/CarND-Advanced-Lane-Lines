import numpy as np
import cv2
import glob


# Undistort
images = glob.glob('./camera_cal/calibration*.jpg')


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.
def_images = []
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, (720, 1280), None, None)
np.save('cmx_dist.npy', (cameraMatrix, distCoeffs))


# perspective
firstline = 480
secondline = 700
x1 = 554
x2 = 233
x3 = 1070
x4 = 732

first_point = (x1, firstline)
second_point = (x2, secondline)
third_point = (x3, secondline)
fourth_point = (x4, firstline)
src = np.float32([first_point, second_point, third_point, fourth_point])
dst = np.float32([(x2, 0), (x2, 720), (x3, 720), (x3, 0)])

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

np.save('M_Minv.npy', (M, Minv))
