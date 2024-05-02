import cv2
import numpy as np
from matplotlib import pyplot as plt


def k_means_detection(image_path, total_colors):
    """
    Function that runs k-means algorithm to take out the dominant colors from an image.

    Args:
        image_path (str): The path to the input image.
        total_colors (str): The amount of colors to take out of the image.

    Returns:
        target_image (numpy.ndarray): The image with the dominant colors removed.
    """
    # Locates the targets
    cartoonImage = cv2.imread(image_path)
    k = total_colors
    data = np.float32(cartoonImage).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, _, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    masks = []
    for i in range(k):
        hsv_center = cv2.cvtColor(np.uint8([[center[i]]]), cv2.COLOR_BGR2HSV)[0][0]

        hue_lower = max(0, hsv_center[0] - 180)
        hue_upper = min(180, hsv_center[0] + 180)
        saturation_lower = max(0, hsv_center[1] - 255)
        saturation_upper = min(255, hsv_center[1] + 80)
        value_lower = max(0, hsv_center[2] - 255)
        value_upper = min(255, hsv_center[2] + 50)
        
        lower_limit1 = np.array([hue_lower, saturation_lower, value_lower], dtype=np.uint8)
        upper_limit1 = np.array([hue_upper, saturation_upper, value_upper], dtype=np.uint8)

        lower_limit2 = np.array([hue_lower, saturation_lower, value_lower], dtype=np.uint8)
        upper_limit2 = np.array([hue_upper, max(0, saturation_upper-150), min(255, value_upper+120)], dtype=np.uint8)
        
        mask1 = cv2.inRange(cv2.cvtColor(cartoonImage, cv2.COLOR_BGR2HSV), lower_limit1, upper_limit1)
        mask2 = cv2.inRange(cv2.cvtColor(cartoonImage, cv2.COLOR_BGR2HSV), lower_limit2, upper_limit2)
        mask = mask1 | mask2
        masks.append(mask)
    result = cartoonImage.copy()
    for mask in masks:
        inverted_mask = cv2.bitwise_not(mask)
        result = cv2.bitwise_and(result, result, mask=inverted_mask)
    cv2.imwrite('final_images/drone.jpg', result)

    target_image = cv2.imread('final_images/drone.jpg')

    return target_image


def reject_outliers(dataset, indices_to_reject):
    """
    Helper function to take targets out of dataset if they have outlier widths & heights.

    Args:
        dataset (list): List of target coordinates and widths, heights.
        indices_to_reject (list): List of indices to run outlier detection and removal on.

    Returns:
        dataset (list): Updated list of target coordinates and widths, heights without outliers.
    """
    for index in indices_to_reject:
        mean = 0
        for list in dataset:
            mean += list[index]

        mean /= len(dataset)

        for i, list in enumerate(dataset):
            if list[index] < 0.5 * mean:
                dataset.pop(i)

    return dataset
    

def detect_targets(image_path):
    """
    Function to locate and return target coordinates.

    Args:
        image_path (str): The path to the input image.

    Returns:
        locations (list): List of x, y coordinates of each target, as well as corresponding widths and heights.
    """
    target_image = k_means_detection(image_path, 10)

    # finds contours, sets a minimum and maximum area to detect letters
    grayscale_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(grayscale_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 15
    contours = [i for i in contours if cv2.contourArea(i) > min_contour_area]

    locations = []

    # for loop to draw contours for each shape detected
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        dimensions = [x, y, w, h]

        locations.append(dimensions)

    indices_to_reject = [2, 3]
    locations = reject_outliers(locations, indices_to_reject)

    return locations, target_image


# get rid of this after full algorithm is complete
def display_graph(image):
    """
    Function to display a graph showing images.

    Args:
        image (numpy.ndarray): The image to be displayed.

    Returns:
        N/A
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _ = plt.imshow(image)
    plt.show()