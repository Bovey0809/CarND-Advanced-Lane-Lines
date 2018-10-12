import numpy as np
import cv2


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        sobel_image = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel_image = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absolute_sobel = np.absolute(sobel_image)
    scale_sobel = np.uint8(255 * absolute_sobel / np.max(absolute_sobel))
    grad_binary = np.zeros_like(scale_sobel)
    grad_binary[(scale_sobel > thresh[0]) & (scale_sobel < thresh[1])] = 1
    return grad_binary


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale = np.uint8(gradmag * 255 / np.max(gradmag))
    mag_binary = np.zeros_like(scale)
    mag_binary[(scale >= mag_thresh[0]) & (scale <= mag_thresh[1])] = 1
    return mag_binary


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def threshold(image):
    # Choose a Sobel kernel size
    ksize = 3  # Choose a larger odd number to smooth gradient measurements
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, 'x', ksize, (10, 100))
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(50, 200))
    dir_binary = dir_threshold(image, ksize, (1.2, 1.9))
    HLS_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    S = HLS_image[:, :, 2]

    # RGB
    R = image[:, :, 2]

    # combined thresholds
    sobel_binary = np.zeros_like(dir_binary)
    sobel_binary[(gradx == 1) & (mag_binary == 1)] = 1

    S_thresh = (125, 255)
    S_binary = np.zeros_like(dir_binary)
    S_binary[(S > S_thresh[0]) & (S <= S_thresh[1])] = 1

    R_thresh = (200, 255)
    R_binary = np.zeros_like(R)
    R_binary[(R >= R_thresh[0]) | (R <= R_thresh[1])] = 1

    combined = np.zeros_like(dir_binary)
    combined[((S_binary == 1) & (sobel_binary == 1)) | ((S_binary == 1) &
                                                        (R_binary == 1)) | ((sobel_binary == 1) & (R_binary == 1))] = 1
    return combined
