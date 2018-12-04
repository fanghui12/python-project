import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import math


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size=5):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    top = 320
    bottom = 540

    left_x1 = []
    left_y1 = []
    left_x2 = []
    left_y2 = []
    right_x1 = []
    right_y1 = []
    right_x2 = []
    right_y2 = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), [0, 0, 255], 6)

            slope = clac_slope(x1, y1, x2, y2)
            # Left line
            if slope < 0:
                # Ignore obviously invalid lines
                if slope > -0.5 or slope < -0.8:
                    continue
                left_x1.append(x1)
                left_y1.append(y1)
                left_x2.append(x2)
                left_y2.append(y2)
            # Right line
            else:
                if slope < 0.5 or slope > 0.8:
                    continue
                right_x1.append(x1)
                right_y1.append(y1)
                right_x2.append(x2)
                right_y2.append(y2)

    try:
        avg_left_x1 = np.mean(left_x1)
        avg_left_y1 = np.mean(left_y1)
        avg_left_x2 = np.mean(left_x2)
        avg_left_y2 = np.mean(left_y2)
        left_slope = clac_slope(avg_left_x1, avg_left_y1, avg_left_x2, avg_left_y2)

        left_y1_c = top
        left_x1_c = int(avg_left_x1 + (left_y1_c - avg_left_y1) / left_slope)
        left_y2_c = bottom
        left_x2_c = int(avg_left_x2 + (left_y2_c - avg_left_y2) / left_slope)

        cv2.line(img, (left_x1_c, left_y1_c), (left_x2_c, left_y2_c), color, thickness)
    except ValueError:
        # Don't error when a line cannot be drawn
        pass

    try:
        avg_right_x1 = np.mean(right_x1)
        avg_right_y1 = np.mean(right_y1)
        avg_right_x2 = np.mean(right_x2)
        avg_right_y2 = np.mean(right_y2)
        right_slope = clac_slope(avg_right_x1, avg_right_y1, avg_right_x2, avg_right_y2)

        right_y1_c = top
        right_x1_c = int(avg_right_x1 + (right_y1_c - avg_right_y1) / right_slope)
        right_y2_c = bottom
        right_x2_c = int(avg_right_x2 + (right_y2_c - avg_right_y2) / right_slope)

        cv2.line(img, (right_x1_c, right_y1_c), (right_x2_c, right_y2_c), color, thickness)
    except ValueError:
        pass


def clac_slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    gray = grayscale(image)
    gaus = gaussian_blur(gray, 5)
    edges = canny(gaus, 50, 150)

    imshape = image.shape
    plt.imshow(edges)
    plt.show()

    vertices = np.array([[(0, imshape[0]), (450, 320), (490, 320), (imshape[1], imshape[0])]], dtype=np.int32)
    masked = region_of_interest(edges, vertices)
    plt.imshow(masked)
    plt.show()
    rho = 2
    theta = np.pi / 180
    threshold = 20
    min_line_len = 40
    max_line_gap = 20
    line_image = hough_lines(masked, rho, theta, threshold, min_line_len, max_line_gap)

    line_edge = weighted_img(line_image, image)

    return line_edge


images = os.listdir("1/")
for img_file in images:
    # print(img_file)
    # Skip all files starting with line.
    if img_file[0:4] == 'line':
        continue

    image = mpimg.imread('1/' + img_file)

    weighted = process_image(image)

    plt.imshow(weighted)
    plt.show()
    # mpimg.imsave('test_image/lines-' + img_file, weighted)


# #reading in an image
# image = mpimg.imread('1/solidWhiteRight.jpg')
#
# #printing out some stats and plotting
# print('This image is:', type(image), 'with dimensions:', image.shape)
# plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
# plt.show()