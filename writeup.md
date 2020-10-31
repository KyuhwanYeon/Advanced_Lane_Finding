## Advanced lane finding

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


### Description

![Result Image](./output_images/result_img.png)

Above image describes the process of the lane detection.

Step 1. Find calibration matrix. 

Step 2. Undistort the image

Step 3. Filter the image by using gradient and color method

Step 4: Unwarp the image for perspective view

Step 5: Find the lane with sliding window

Step 6: Fill the lane

Step 7: Warp image again

Step 8: Show with origin image


