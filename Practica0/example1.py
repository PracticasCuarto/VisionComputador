# Example 1: read and show an image
import cv2 # Import python-supported OpenCV functions
import numpy as np # Import numpy and call it np
from matplotlib import pyplot as plt # Import pyplot and call it plt
img = cv2.imread('cilindros.png', cv2.IMREAD_GRAYSCALE)
cv2.namedWindow( 'Example1', cv2.WINDOW_AUTOSIZE )
cv2.imshow('Example1',img)
print(type(img))
print(img.shape)
cv2.waitKey(0)
cv2.destroyWindow( 'Example1' ) # cv2.destroyAllWindows()