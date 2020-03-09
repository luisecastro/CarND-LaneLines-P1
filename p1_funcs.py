import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

"""
Set of functions used in for the P1 Self Driving Car Nanodegree
Luis Castro 2020
"""


def read_image(image_path: str, verbose=False) -> np.array:
    """
    Function to read an image file
    :param image_path: path to the file to be read
    :param verbose: set to True if you want to print image information
    :return: the image as a numpy array
    """
    image = mpimg.imread(image_path)
    if verbose:
        print(f'Image type: {type(image)} - Image dimensions:{image.shape}')
    return image


def show_image(image: np.array, cmap=None, figsize=(16, 9)):
    """
    Function to show the image
    :param image: image to show
    :param cmap: change to the desired cmap, e.g. 'gray; for grayscale
    :param figsize: the size the image will be plotted
    """
    plt.figure(figsize=figsize)
    if cmap:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(image)
    plt.show()


def color_transform(image: np.array, color_space=cv2.COLOR_RGB2GRAY) -> np.array:
    """
    Function to change from a color space to another
    check: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html
    :param image: image to transform
    :param color_space: transformation method, set to RGB2GRAY by default
    :return: transformed image
    """
    return cv2.cvtColor(image, color_space)


def gaussian_blur(image: np.array, kernel_size: int) -> np.array:
    """
    Function to apply Gaussian Blur on an image (reduce noise)
    check: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html#filtering
    :param image: image to be transformed
    :param kernel_size: Kernel size, here it applies it horizontally and
    vertically with the same size, must be an odd number
    :return: image with blur applied
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0, borderType=cv2.BORDER_DEFAULT)


def canny_transform(image: np.array, low_threshold: int, high_threshold: int) -> np.array:
    """
    Function to apply the Canny transformation (find edges)
    check: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
    :param image: image to be transformed
    :param low_threshold:
    :param high_threshold:
    :return:
    """
    return cv2.Canny(image, low_threshold, high_threshold)


def apply_mask(image: np.array, vertices: np.array) -> np.array:
    """
    Function to select a region of an image an occult everything else
    :param image: image to be used
    :param vertices: The vertices of the polygon that will select the area to keep
    the format to be given to the function is a bit sketchy, and the order of the vertices
    is as follows {left-down, left-up, right-up, right-down}
    :return: The image displaying only the region selected
    """
    vertices = np.array([vertices], dtype=np.int32)
    mask = np.zeros_like(image)
    mask_color = (255,) * image.shape[2] if len(image.shape) > 2 else 255
    cv2.fillPoly(mask, vertices, mask_color)
    return cv2.bitwise_and(image, mask)


def hough_lines(image: np.array, rho: float, theta: float, threshold: int,
                min_line_length: int, max_line_gap: int, thickness: int, color=(255, 0, 0)) -> np.array:
    """
    Function to apply the Hough lines
    check: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
           https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html
    :param image: image to be used
    :param rho: The resolution of the parameter r in pixels.
    :param theta: The resolution of the parameter in radians.
    :param threshold: the minumum number of intersections to detect a line
    :param min_line_length: minimum length of the line to be considered in pixels
    :param max_line_gap: maximum length of a gap between to points to be considered part of a line
    :param thickness: thickness of the lines to be shown in the returned image
    :param color: color of the line to be show, set to red by default
    :return: image of lines drawn on it
    """
    line_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    coords = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    for coord in coords:
        for x1, y1, x2, y2 in coord:
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    return line_image


def hough_coords(image: np.array, rho: float, theta: float, threshold: int, min_line_length: int, max_line_gap: int) -> np.array:
    """
    A change on the hough_lines function modularizing it and allowing it to return interpolation coords for interpolation
    :param image: image to be used
    :param rho: The resolution of the parameter r in pixels.
    :param theta: The resolution of the parameter in radians.
    :param threshold: the minumum number of intersections to detect a line
    :param min_line_length: minimum length of the line to be considered in pixels
    :param max_line_gap: maximum length of a gap between to points to be considered part of a line
    return: set of line coordinates
    """
    return cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)


def plot_lines(image: np.array, coords: np.array, color=(255, 0, 0), thickness=2) -> np.array:
    """
    Function creates lines to be applied on an image
    :param image: reference image
    :param coords: coordinates of the lines to be plotted
    :param color: color of the lines, red by default
    :param thickness: thickness in pixels for the line
    :return: set of lines drawn in an image
    """
    line_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for coord in coords:
        for x1, y1, x2, y2 in coord:
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    return line_image


def invert_color(image: np.array) -> np.array:
    """
    Function inverts an image
    :param image: image to be transformed
    :return: image with colors inverted
    """
    return cv2.bitwise_not(image)


def weighted_image(image: np.array, lines_image: np.array, alpha: float, beta: float, gamma: float):
    """
    Function blends two imaged, the background image and the image with lines on top off it.
    check: https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html?highlight=addweighted#addweighted
    :param image: background image
    :param lines_image: lines image
    :param alpha: weight for values in image
    :param beta: weight for values in lines_image
    :param gamma: an offset added to the resulting image
    :return: blended image of background and lines
    """
    return cv2.addWeighted(image, alpha, lines_image, beta, gamma)


def write_image(image: np.array, path: str):
    """
    Function writes an image to disk
    :param image: image to be saved
    :param path: path to save the image
    """
    cv2.imwrite(path, image)

def interpolate(image_coord, lines_eq, y_min=324, y_max=540, slopes=[1, 2]):
    """
    Function takes a set of line coordinates and returns 2 sets of line coordinates created
    with the my + b line formula calculated from the line with the largest length. y is 
    used here as the independent variable. It considers a line if its slope makes sense, is between
    the values specified.
    :param image_coord: set of line coordinates from hough_coords
    :param lines_eq: the equation for the two line, here created for the y_min and y_max provided
    :param y_min: the min y to plot in the image
    :param y_max: the max y to plot in the image
    :param slopes: a min and max abs value for the slopes, slope 0 would be a straight vertical line,
    a high slope would approximate an horizontal line
    :return: set of 2 line coordinates
    """
    lines_eq[0][-1] = 0
    lines_eq[1][-1] = 0

    for line in image_coord[:, 0]:
        length = np.linalg.norm(line)
        m, b = np.polyfit([line[3], line[1]], [line[2], line[0]], 1)
        pos = int(m > 0)

        if length > lines_eq[pos][-1] and abs(m) < 2 and abs(m) > 1:
            lines_eq[pos][-1] = length
            lines_eq[pos][:2] = [m, b]

    f0 = np.poly1d(lines_eq[0][:2])
    line0 = [f0(y_min), y_min, f0(y_max), y_max]

    f1 = np.poly1d(lines_eq[1][:2])
    line1 = [f1(y_min), y_min, f1(y_max), y_max]

    return np.array([[line0], [line1]], dtype=np.int32), lines_eq


def img_pipeline(image, vertices, kernel, low_threshold, high_threshold, rho, theta, threshold,
                 min_line_length, max_line_gap, alpha, beta, gamma, thickness, lines_eq=[[-1.42,922,0],[1.65,-34,0]], 
                 show=False, write=False, image_save=None, ret=False, gray=True):
    """
    Function uses all transformations to take an image and detect its lines and display them on top.
    Ideally all the best parameters are already found and set in this function.
    :param image: image to be used
    :param vertices: vertices coordinates as a string ('x0y0|x1y1|x2y2|x3y3')
    :param kernel: kernel size for Gaussian blur
    :param low_threshold: below this, now gradient is considered a line
    :param high_threshold: above this, the gradient is considered as a line 
    :param rho: min length of line to be considered
    :param theta: min angle of line to be considered
    :param threshold: min number of intersections of a line
    :param min_line_length: min length of a line 
    :param max_line_gap: max gap between line coord
    :param alpha: strength of background
    :param beta: strength of lines
    :param gamma: offset for image in addWeighted
    :param thickness: thickness of lines in pixels
    :param show: set to True to display the image
    :param write: set to True to save an image
    :param image_save: path to save an image, only used if write is True
    :param ret: set to True to return the image
    :return: image if ret is True
    """
    image_trans = color_transform(image) if gray else image.copy()
    image_gauss = gaussian_blur(image_trans, kernel)
    image_canny = canny_transform(image_gauss, low_threshold, high_threshold)
    image_mask = apply_mask(image_canny, vertices)
    image_coord = hough_coords(image_mask, rho, theta, threshold, min_line_length, max_line_gap)
    image_lines, lines_eq = interpolate(image_coord, lines_eq, vertices[0][0][1], vertices[0][1][1])
    image_hough = plot_lines(image, image_lines, thickness=thickness)
    weight_image = weighted_image(image, image_hough, alpha, beta, gamma)
    if show: show_image(weight_image)
    if write: write_image(color_transform(weight_image, cv2.COLOR_BGR2RGB), image_save)
    if ret: return weight_image