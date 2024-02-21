import cv2
import numpy as np
from matplotlib import pyplot as plt


def k_means_detection(image_path, total_colors):
    # Locates the targets
    cartoonImage = cv2.imread(image_path)
    k = total_colors
    data = np.float32(cartoonImage).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
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


def contour_intersect(original_image, contour1, contour2):
    """
    Function to check if two contours intersect within an original image.
    https://stackoverflow.com/questions/55641425/check-if-two-contours-intersect

    Args:
        original_image (numpy.ndarray): The original image containing the contours.
        contour1 (numpy.ndarray): The first contour to check for intersection.
        contour2 (numpy.ndarray): The second contour to check for intersection.

    Returns:
        bool: True if the contours intersect; otherwise, False.
    """
    contours = [contour1, contour2]

    blank = np.zeros(original_image.shape[0:2])

    image1 = cv2.drawContours(blank.copy(), contours, 0, 1)
    image2 = cv2.drawContours(blank.copy(), contours, 1, 1)

    intersection = np.logical_and(image1, image2)

    return intersection.any()


def detect_targets(image_path):
    """
    Function to detect and analyze targets in an image.

    Args:
        image (str): The path to the input image.

    Returns:
        tuple: A tuple containing two images - the masked result image and the image with contours and shapes overlaid.
    """
    color_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    final_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    target_image = k_means_detection(image_path, 10)

    # finds contours, sets a minimum and maximum area to detect letters
    grayscale_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(grayscale_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 15
    contours = [i for i in contours if cv2.contourArea(i) > min_contour_area]

    locations = []

    # for loop to draw contours for each shape detected
    for cnt in contours:
        # creates bounded rectangle
        [x, y, w, h] = cv2.boundingRect(cnt)
        locations.append([x,y])
        x -= 15
        y -= 15

        # crops image to bounded rectangle and creates a grayscale image
        cropped_image = color_image[y:y+h+30, x:x+w+30]
        cropped_grayscale_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        # applies threshold algorithm
        threshold_image = cv2.threshold(cropped_grayscale_image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # finds contours in cropped image
        contours, hierarchy = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [i for i in contours if cv2.contourArea(i) > min_contour_area]

        # creates convex hulls from contours
        hull = []
        for cnt in contours:
            #for i in range(len(contours)):
            #    if not contour_intersect(cropped_image, cnt, contours[i]):
            hull.append(cv2.convexHull(cnt, False))

        # draws the convex hulls on the cropped image
        cv2.drawContours(cropped_image, hull, -1, (0, 255, 0), 3)

        # overlays cropped image onto color image
        final_image[y:y+cropped_image.shape[0], x:x+cropped_image.shape[1]] = cropped_image

    return locations, original_image, target_image, final_image


def display_graph(original_image, target_image, final_image):
    """
    Function to display a graph showing images.

    Args:
        result_image (numpy.ndarray): The masked result image.
        final_image (numpy.ndarray): The image with contours and shapes overlaid.

    Returns:
        None: This function displays the graph using Matplotlib.
    """
    titles = ['Original Image', 'Masked', 'Contours']
    images = [original_image, target_image, final_image]
    for i in range(3):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], 'gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.show()