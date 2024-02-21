import cv2
import numpy as np
from matplotlib import pyplot as plt

def find_shape(contour, rect_location_x, rect_location_y):
    """
    Function to detect shapes of targets.

    Args:
        contour (numpy.ndarray): The contour to be analyzed and for which the shape is determined.
        rect_location_x (int): The X-coordinate of the bounding rectangle's top-left corner relative to the image.
        rect_location_y (int): The Y-coordinate of the bounding rectangle's top-left corner relative to the image.

    Returns:
        shape_dict (dict): A dictionary containing the location (X, Y) of the shape in the image and the number of sides of the detected shape.
    """
    approx = cv2.approxPolyDP(contour, 0.06 * cv2.arcLength(contour, True), True)  # approximates number of sides

    # gets location of the shapes in the image and adds to dictionary of shape locations & sides
    M = cv2.moments(contour)
    if M['m00'] != 0.0:
        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])

        shape_dict = {}
        shape_dict[str(x + rect_location_x) + ", " + str(y + rect_location_y)] = len(approx)
        print(shape_dict)
        return shape_dict



def get_circle_percentages(image):
    """
    Function to detect circles in an image and calculate the percentage completeness of each detected circle.

    Args:
        image (numpy.ndarray): The input image in which circles are to be detected.

    Returns:
        List[float]: A list of percentages representing the completeness of detected circles.
    """
    dist = cv2.distanceTransform(image, cv2.DIST_L2, 0)
    rows = image.shape[0]
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, rows / 8, 50, param1=50, param2=10, minRadius=5,
                               maxRadius=2000)

    percentages = []

    if circles is not None:
        for circle in circles[0, :]:
            center = (int(circle[0]), int(circle[1]))
            radius = int(circle[2])

            inlier_count = 0
            total_points = 0

            for angle in np.arange(0, 2 * np.pi, 0.1):
                total_points += 1
                x = int(center[0] + radius * np.cos(angle))
                y = int(center[1] + radius * np.sin(angle))

                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    if dist[y, x] < (radius / 25.0):
                        inlier_count += 1

            completeness_percentage = (inlier_count / total_points) * 100.0
            percentages.append(completeness_percentage)
            print(f"{completeness_percentage:.2f}% of a circle with radius {radius} detected at ({center[0]}, {center[1]})")

    return percentages


def categorize_shapes(contour):
    """
    Function to categorize shapes based on the number of sides.

    Args:
        contour (numpy.ndarray): The contour to be categorized.

    Returns:
        str: The category of the shape (e.g., 'circle', 'semicircle', 'triangle', 'rectangle', etc.).
    """
    num_sides = len(cv2.approxPolyDP(contour, 0.06 * cv2.arcLength(contour, True), True))

    if num_sides >= 13:
        return 'circle'
    elif num_sides == 7:
        return 'semicircle'
    elif num_sides == 6:
        return 'quarter circle'
    elif num_sides == 3:
        return 'triangle'
    elif num_sides == 4:
        return 'rectangle'
    elif num_sides == 5:
        return 'pentagon'
    elif num_sides == 10:
        return 'star'
    elif num_sides == 12:
        return 'cross'
    else:
        return 'unknown'