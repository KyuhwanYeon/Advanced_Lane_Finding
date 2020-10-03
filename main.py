
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from calibration import calibration, cal_undistort

input_type = 'image'
img_file_name = 'test_images/test2.jpg'
if __name__ == '__main__':
    if input_type == 'image':
        img = cv2.imread(img_file_name)
        matrix, distortion = calibration()
        undistort_img = cal_undistort(img, matrix, distortion)
        plt.imshow(img)
        plt.show()
        print("Complete calibration")
        plt.imshow(undistort_img)
        plt.show()