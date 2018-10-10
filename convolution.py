import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


# Read in a thresholded image
from numpy.core.multiarray import ndarray

warped = mpimg.imread('warped_example.jpg')
# window settings
window_width = 50
window_height = 50  # Break image into 9 vertical layers since image height is 720
margin = 50  # How much to slide left and right for searching


def window_mask(width: int, height: int, img_ref: ndarray, center: float, level: int) -> ndarray:
    """
    Return the window mask depends on the width height, center, and level, make it white.

    :rtype: ndarray
    """
    output = np.zeros_like(img_ref)
    # the image shape is (720, 1280)
    # Level: 代码上指的是找到的centroids的个数。
    # output的【0】是一个height的高度。
    # output的【1】是一个width的长度， 需要考虑越界。
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
           max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    # 在这个height*width的部分改为白色的。
    return output


def find_window_centroids(image, window_width, window_height, margin):
    # todo: what is the usage of margin?
    window_centroids = []  # 就是windon中心的坐标的列表。
    # Create our window template that we will use for convolutions
    window = np.ones(window_width)
    # TODO what is the usage of window? [1, 1, ..., 1, 1]
    # 使用对于histogram的求和得到lane的起始位置。
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3 * image.shape[0] / 4):,
                         :int(image.shape[1] / 2)], axis=0)
    print(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)].shape)
    # l_sum 得到的是在图片左下角按照竖排的所有像素值的加和。
    print(np.convolve(window, l_sum).shape)
    print('l_sum shape', l_sum.shape)
    print(window.shape)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    # 这里用的是full conv， 就是每个每个元素的被乘的次数一样, 输出的长度是N+M-1, 20+640-1
    r_sum = np.sum(image[int(3 * image.shape[0] / 4):,
                         int(image.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - \
        window_width / 2 + int(image.shape[1] / 2)
    # TODO: 为什么这里的中心位置坐标都要减去窗口宽度？
    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int(image.shape[0] / window_height))):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :],
            axis=0)
        # 横着对图片切一层, 加和。
        conv_signal = np.convolve(window, image_layer)
        print(conv_signal.shape)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image.shape[1]))
        l_center = np.argmax(
            conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, image.shape[1]))
        r_center = np.argmax(
            conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return window_centroids


window_centroids = find_window_centroids(
    warped, window_width, window_height, margin)
# window_centroids is list of tuples of lenth 2

# If we found any window centers
if len(window_centroids) > 0:

    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # Go through each level and draw the windows
    for level in range(0, len(window_centroids)):

        # Window_mask is a function to draw window areas
        l_mask = window_mask(window_width, window_height,
                             warped, window_centroids[level][0], level)
        r_mask = window_mask(window_width, window_height,
                             warped, window_centroids[level][1], level)
        # Add graphic points from window mask here to total pixels found
        l_points[(l_points == 255) | ((l_mask == 1))] = 255
        r_points[(r_points == 255) | ((r_mask == 1))] = 255

    # Draw the results
    # add both left and right window pixels together
    template = np.array(r_points + l_points, np.uint8)
    zero_channel = np.zeros_like(template)  # create a zero color channel
    # make window pixels green
    template = np.array(
        cv2.merge((zero_channel, template, zero_channel)), np.uint8)
    # making the original road pixels 3 color channels
    warpage = np.dstack((warped, warped, warped)) * 255
    # overlay the orignal road image with window results
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)

# If no window centers found, just display orginal road image
else:
    output = np.array(cv2.merge((warped, warped, warped)), np.uint8)

# Display the final results
plt.imshow(output)
plt.title('window fitting results')
plt.show()
