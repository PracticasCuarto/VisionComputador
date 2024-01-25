# Example 2: Matrices in OpenCV: properties
import cv2 # Import python-supported OpenCV functions
import numpy as np # Import numpy and call it np
from matplotlib import pyplot as plt # Import pyplot and call it plt

def show_image_properties(my_img):
    cv2.namedWindow( 'Example1', cv2.WINDOW_AUTOSIZE )
    cv2.imshow('Example1',my_img)
    print("Properties of the matrix used to store the image")
    print("They are numpy arrays: type(my_img)= ", type(my_img))
    print("Rows, columns and channels: my_img.shape= ", my_img.shape)
    print("Total number of pixels: my_img.size= ", my_img.size)
    print("Image datatype: my_img.dtype = ", my_img.dtype )
    cv2.waitKey(1000) #cv2.waitKey(0)
    cv2.destroyWindow( 'Example1' ) # cv2.destroyAllWindows()
    return(0)

img = cv2.imread('cilindros.png', cv2.IMREAD_GRAYSCALE)
img_bgr = cv2.imread('cilindros.png', cv2.IMREAD_COLOR)
show_image_properties(img)
show_image_properties(img_bgr)
