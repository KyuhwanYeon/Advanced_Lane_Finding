
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from calibration import calibration, cal_undistort
from threshold_filter import grad_color_filter

input_type = 'image'
img_file_name = 'test_images/test2.jpg'
if __name__ == '__main__':
    if input_type == 'image':
        img = cv2.imread(img_file_name)
        matrix, distortion = calibration()
        print("Complete calibration")
        #undistort the image
        undistort_img = cal_undistort(img, matrix, distortion)
        print("Complete undistortion")
        filtered_img = grad_color_filter(undistort_img)
        print("Complete filter")
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 3))
        ax1.imshow(img)
        ax1.set_title('original Image', fontsize=10)
        ax2.imshow(undistort_img)
        ax2.set_title('undistortion', fontsize=10)
        ax3.imshow(filtered_img, cmap='gray')
        ax3.set_title('filtered', fontsize=10)
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(undistort_img)
        # plt.show()
        # plt.imshow(filtered_img)
        # plt.show()