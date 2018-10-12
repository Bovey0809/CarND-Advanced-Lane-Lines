import numpy as np


def generate_data(ploty, leftx, rightx, ym_per_pix, xm_per_pix):
    '''
    Generates fake data to use for calculating lane curvature.
    In your own project, you'll ignore this function and instead
    feed in the output of your lane detection algorithm to
    the lane curvature calculation.
    '''

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # Fit a second order polynomial to pixel positions in each fake lane line
    ##### Utilize `ym_per_pix` & `xm_per_pix` here #####
    leftx *= xm_per_pix
    rightx *= xm_per_pix
    ploty *= ym_per_pix
    left_fit_cr = np.polyfit(ploty, leftx, 2)
    right_fit_cr = np.polyfit(ploty, rightx, 2)

    return ploty, left_fit_cr, right_fit_cr


def measure_curvature_real(ploty, leftx, rightx):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    distance = (1280/2 - (leftx[-1] + rightx[-1]) / 2) * xm_per_pix
    ploty, left_fit_cr, right_fit_cr = generate_data(
        ploty, leftx, rightx, ym_per_pix, xm_per_pix)

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    def radius(y_eval, fit):
        A = fit[0]
        B = fit[1]
        return (1 + (2 * A * y_eval + B) ** 2) ** (3 / 2) / np.abs(2 * A)

    # Implement the calculation of the left line here
    left_curverad = radius(y_eval, left_fit_cr)
    # Implement the calculation of the right line here
    right_curverad = radius(y_eval, right_fit_cr)
    return left_curverad, right_curverad, distance
