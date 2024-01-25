# Example 3: Matrices in OpenCV: elements
import cv2 # Import python-supported OpenCV functions
import numpy as np # Import numpy and call it np
from matplotlib import pyplot as plt # Import pyplot and call it plt

# Create a black image (BGR: uint8, 3 channels)
img_uint8 = np.zeros((512,512,3), np.uint8)
# Draw a diagonal BLUE line with thickness of 5 px
cv2.line(img_uint8,(0,0),(511,511),(255,0,0),5)
show_image_properties(img_uint8)
# Create a black image (BGR: uint8, 3 channels)
img_uint8b = np.zeros((512,512,3), np.uint8)
# Access to elements and set them the value BLUE
img_uint8b[0:100,0:25]=[250,0,0]
# Alternatively..
#img_uint8b[0:100,0:25,0]=250
#img_uint8b[0:100,0:25]=np.array([250,0,0], np.uint8)
show_image_properties(img_uint8b)
# Try and observe the following mistakes (common mistakes; note that often
# there is not even a warning..)
img_uint8d = np.zeros((512,512,3), np.uint8)
#img_uint8d[0:100,0:25,0]= -100 #in some versions, no warning!!
#img_uint8d[0:100,0:25,0]= -500 #in some versions, no warning!!
show_image_properties(img_uint8d)