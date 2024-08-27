import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Function to capture mouse click
points = []

def select_pts(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x,y))
        cv.circle(image, (x,y), 5, -1)

# Load the image 
image = cv.imread('./myImage.png', cv.IMREAD_COLOR)

if image is None:
    print('Error: Image not found.')
else:
    # Set the mouse callback function
    cv.imshow('Select ROI', image)
    cv.setMouseCallback('Select ROI', select_pts)

    while len(points) < 4:
        cv.waitKey(1)

    cv.destroyAllWindows()
    
    for i, point in enumerate(points):
        print(f'Point {i+1}: {point}')

