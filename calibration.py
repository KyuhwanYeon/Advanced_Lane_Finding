
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import glob

# Read in and make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# prepare object points
nx = 9
ny = 6
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane
# Prepare object points like (0,0,0), (1, 0, 0)
objp = np.zeros((nx*ny,3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)  # x,y coordinates
# Make a list of calibration images
def calibration():
    for img_fname in images:
        fname = img_fname
        img = cv2.imread(fname)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        # If found, draw corners
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist

def cal_undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
def unwarp(img, src, dst, img_size):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    M_for_rewarp = cv2.getPerspectiveTransform(dst, src)
    return warped, M, M_for_rewarp
    
        

if __name__ == '__main__':
    img = cv2.imread(images[0])
    matrix, distortion = calibration()
    undistort_img = cal_undistort(img, matrix, distortion)
    c_rows, c_cols = undistort_img.shape[:2]
    s_LTop2, s_RTop2 = [917, 15], [1249, 46]
    s_LBot2, s_RBot2 = [917, 454], [1249, 432]
    src = np.float32([s_LBot2, s_LTop2, s_RTop2, s_RBot2])
    dst = np.float32([(0, 720), (0, 0), (720, 0), (720, 720)])
    unwarp_img , M, M_for_warp = unwarp(undistort_img, src, dst, (720, 720))
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 3))
    ax1.imshow(img)
    ax1.set_title('original Image', fontsize=10)
    ax2.imshow(undistort_img)
    ax2.set_title('undistortion', fontsize=10)
    ax3.imshow(unwarp_img, cmap='gray')
    ax3.set_title('warp_img', fontsize=10)       