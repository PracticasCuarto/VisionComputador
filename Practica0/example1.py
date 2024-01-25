import cv2
import numpy as np

# crear el main
if __name__ == "__main__":
    # Read the image
    img = cv2.imread('Mari.jpeg', cv2.IMREAD_GRAYSCALE)

    # Create a named window
    cv2.namedWindow('Example1', cv2.WINDOW_NORMAL)  # WINDOW_NORMAL allows resizing

    # Resize the window to a smaller size
    cv2.resizeWindow('Example1', 1024, 1024)  # Adjust size as needed

    # Display the image
    cv2.imshow('Example1', img)

    # Wait for a key press
    cv2.waitKey(0)

    # Destroy the window
    cv2.destroyWindow('Example1')

    # Finalizar el programa
    exit(0)

    
