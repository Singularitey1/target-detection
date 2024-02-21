from target_detection import *

def main():
    """
    Main method that initializes and runs everything.
    """
    IMAGE_PATH = "input_images/drone1.jpg"  # image file location
    detected_images = detect_targets(IMAGE_PATH)
    display_graph(detected_images[1], detected_images[2], detected_images[3])
    print(detected_images[0])


if __name__ == "__main__":
    main()