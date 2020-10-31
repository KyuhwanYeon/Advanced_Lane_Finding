#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 16:59:59 2020

@author: khyeon
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

image = mpimg.imread('test_images/test1.jpg')


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output
# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def gradient_filter(image):
    sobel_x_grad_binary = abs_sobel_thresh(image, orient='x', thresh_min=30, thresh_max=100)
    sobel_y_grad_binary = abs_sobel_thresh(image, orient='y', thresh_min=30, thresh_max=100)
    mag_thresh_grad_binary = mag_thresh(image ,sobel_kernel=3, mag_thresh =(50,100))
    dir_thresh_grad_binary = dir_threshold(image, sobel_kernel=3, thresh=(0.7, 1.3))
    combined = np.zeros_like(dir_thresh_grad_binary)
    combined[((sobel_x_grad_binary == 1) & (sobel_y_grad_binary == 1)) | ((mag_thresh_grad_binary == 1) & (dir_thresh_grad_binary == 1))] = 1
    return combined


def color_filter(image):
    R = image[:, :, 0]
    R_thresh = (200, 255)
    R_binary = np.zeros_like(R)
    R_binary[(R > R_thresh[0]) & (R <= R_thresh[1])] = 1
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    S = hls[:,:,2]
    H_thresh = (15, 100)
    S_thresh = (90, 255)
    H_bianry = np.zeros_like(H)
    S_binary = np.zeros_like(S)
    H_bianry[(H > H_thresh[0]) & (H <= H_thresh[1])] = 1
    S_binary[(S > S_thresh[0]) & (S <= S_thresh[1])] = 1
    color_combined = np.zeros_like(S_binary)
    color_combined[(R_binary == 1) | ((H_bianry == 1)&(S_binary == 1))] = 1
    #color_combined[(H_bianry == 1)&(S_binary == 1)] = 1
    return color_combined


def grad_color_filter(image):
    color_combined = color_filter(image)
    gradient_combined = gradient_filter(image)
    gradient_color_combined = np.zeros_like(color_combined)
    # gradient_color_combined[gradient_combined ==1|(color_combined==1)] = 1
    
    gradient_color_combined[(gradient_combined == 1)] = 50
    gradient_color_combined[(color_combined == 1)] = 255
    return gradient_color_combined
if __name__ == '__main__':
    # Run the function
    sobel_x_grad_binary = abs_sobel_thresh(image, orient='x', thresh_min=10, thresh_max=100)
    sobel_y_grad_binary = abs_sobel_thresh(image, orient='y', thresh_min=10, thresh_max=255)
    mag_thresh_grad_binary = mag_thresh(image ,sobel_kernel=3, mag_thresh =(30,100))
    dir_thresh_grad_binary = dir_threshold(image, sobel_kernel=3, thresh=(0.7, 1.0))
    gradient_combined = np.zeros_like(dir_thresh_grad_binary)
    gradient_combined[((sobel_x_grad_binary == 1) & (sobel_y_grad_binary == 1)) | ((mag_thresh_grad_binary == 1) & (dir_thresh_grad_binary == 1))] = 1
    
    R = image[:, :, 0]
    R_thresh = (200, 255)
    R_binary = np.zeros_like(R)
    R_binary[(R > R_thresh[0]) & (R <= R_thresh[1])] = 1
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    S = hls[:,:,2]
    H_thresh = (15, 100)
    S_thresh = (90, 255)
    H_bianry = np.zeros_like(H)
    S_binary = np.zeros_like(S)
    H_bianry[(H > H_thresh[0]) & (H <= H_thresh[1])] = 1
    S_binary[(S > S_thresh[0]) & (S <= S_thresh[1])] = 1
    color_combined = np.zeros_like(S_binary)
    #color_combined[(R_binary == 1) | ((H_bianry == 1)&(S_binary == 1))] = 1
    color_combined[(H_bianry == 1)&(S_binary == 1)] = 1
    
    gradient_color_combined = np.zeros_like(color_combined)
    gradient_color_combined[gradient_combined ==1|(color_combined==1)] = 1
    
    plt.close('all')
    # Plot the result
    f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, figsize=(24, 3))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('original Image', fontsize=10)
    ax2.imshow(sobel_x_grad_binary, cmap='gray')
    ax2.set_title('sobel_x', fontsize=10)
    ax3.imshow(sobel_y_grad_binary, cmap='gray')
    ax3.set_title('sobel_y', fontsize=10)
    ax4.imshow(mag_thresh_grad_binary, cmap='gray')
    ax4.set_title('mag', fontsize=10)
    ax5.imshow(dir_thresh_grad_binary, cmap='gray')
    ax5.set_title('dir', fontsize=10)
    ax6.imshow(gradient_combined, cmap='gray')
    ax6.set_title('gradient_combined ', fontsize=10)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    # Plot the result
    f, (ax1, ax2, ax3, ax4,ax5) = plt.subplots(1, 5, figsize=(24, 3))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('original Image', fontsize=10)
    ax2.imshow(R_binary, cmap='gray')
    ax2.set_title('R_filter', fontsize=10)
    ax3.imshow(H_bianry, cmap='gray')
    ax3.set_title('H_filter', fontsize=10)
    ax4.imshow(S_binary, cmap='gray')
    ax4.set_title('S_filter', fontsize=10)
    ax5.imshow(color_combined, cmap='gray')
    ax5.set_title('color_combined', fontsize=10)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('original Image', fontsize=10)
    ax2.imshow(gradient_color_combined, cmap='gray')
    ax2.set_title('Grad_Color_Combined', fontsize=10)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

