import matplotlib.pyplot as plt
import matplotlib.image as mping
import numpy as np
import cv2
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip


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

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    # Next we'll create a masked edges image using cv2.fillPoly()
    mask = np.zeros_like(edges)
    ignore_mask_color = 255

    # This time we are defining a four sided polygon to mask
    imshape = image.shape
    print(imshape[0])
    print(imshape[1])
    vertices = np.array([[(0, imshape[0]), (450, 320), (490, 320), (imshape[1], imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    line_img = np.zeros((masked_edges.shape[0], masked_edges.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)

    # # Iterate over the output "lines" and draw lines on a blank image
    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    #
    # # Create a "color" binary image to combine with line image
    # color_edges = np.dstack((edges, edges, edges))

    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(line_img, 0.8, image, 1, 0)
    # plt.imshow(lines_edges)
    # plt.show()
    return lines_edges

white_output = '../test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("../test_videos/solidYellowLeft.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)