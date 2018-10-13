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


def colorHSV(image, low=[0, 80, 200], high=[40, 255, 255]):
    '''return a mask.'''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_hsv_low = np.array(low)
    yellow_hsv_high = np.array(high)
    binary = np.zeros((image.shape[0], image.shape[1]))
    H = image[:, :, 0]
    S = image[:, :, 1]
    V = image[:, :, 2]

    binary[((H > yellow_hsv_low[0]) & (H < yellow_hsv_high[0]))] = 1
    binary[((S > yellow_hsv_low[1]) & (S < yellow_hsv_high[1]))] = 1
    binary[((V > yellow_hsv_low[2]) & (V < yellow_hsv_high[2]))] = 1

    binary[((H == 1) & (S == 1) & (V == 1))] = 1
    return binary


def gray_color_threshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = (190, 255)
    binary = np.zeros_like(gray)
    binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1
    return binary


def red_channel(image):
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    thresh = (200, 255)
    binary = np.zeros_like(R)
    binary[(R > thresh[0]) & (R <= thresh[1])] = 1
    return binary


def hls_color(image):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]
    sthresh = (90, 255)
    hthresh = (15, 100)
    binary = np.zeros_like(H)
    binary[((H > hthresh[0]) & (H <= hthresh[1])) & (
        (S > sthresh[0]) & (S <= sthresh[1]))] = 1

    return binary


def combined(image):
    return (red_channel(image) | hls_color(image) | gray_color_threshold(image))


def threshold(image, ksize=3, grad=(30, 150), mag=(50, 200), dir_t=(1.2, 1.9), S_thresh=(125, 255), R_thresh=(175, 250)):
    # Choose a Sobel kernel size
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, 'x', ksize, grad)
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=mag)
    dir_binary = dir_threshold(image, ksize, dir_t)
    HLS_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # combined thresholds
    sobel_binary = np.zeros_like(dir_binary)
    sobel_binary[(gradx == 1) & (mag_binary == 1)] = 1

    combined = np.zeros_like(dir_binary)
    another_combine = (red_channel(image) | hls_color(image)
                       | gray_color_threshold(image))
    combined[((gradx == 1) | (another_combine == 1))]
    return combined
