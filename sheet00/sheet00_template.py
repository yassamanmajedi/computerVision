import cv2 as cv
import numpy as np
import random
import sys
from numpy.random import randint



def display_image(window_name, img):
    
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows() 


if __name__ == '__main__':

    # set image path
 
    img_path = '/Users/yasamanmajedi/Documents/Studies/Master/Semester 1/CV/sheet00/bonn.png' 
    img = cv.imread(img_path)

    # 2a: read and display the image 

    display_image('2 - a - Original Image', img)

    # 2b: display the intensity image

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    display_image('2 - b - Intensity Image', img_gray)

    # 2c: for loop to perform the operation

    img_gray_05 = 0.5 * img_gray
    rows, cols, _ = img.shape
    img_cpy = img.copy()
    for i in range(rows):
        for j in range(cols):
            B, G, R = img[i, j]
            B_new = max(B - img_gray_05[i, j], 0)
            G_new = max(G - img_gray_05[i, j], 0)
            R_new = max(R - img_gray_05[i, j], 0)

            # Update the result image with new values
            img_cpy[i, j] = [B_new, G_new, R_new]

    display_image('2 - c - Reduced Intensity Image', img_cpy)
    # 2d: one-line statement to perfom the operation above
    img_cpy = np.maximum(img - np.expand_dims(0.5 * img_gray, axis=2), 0).astype(np.uint8)

    display_image('2 - d - Reduced Intensity Image One-Liner', img_cpy)    
    
    # 2e: Extract the center patch and place randomly in the image
   
    patch_size = 16
    center_x, center_y = cols // 2, rows // 2
    img_patch = img[center_y - patch_size // 2 : center_y + patch_size // 2, 
                center_x - patch_size // 2 : center_x + patch_size // 2]


    display_image('2 - e - Center Patch', img_patch)  
    
    # Random location of the patch for placement
    random_x = random.randint(0, cols - patch_size)
    random_y = random.randint(0, rows - patch_size)
    img_cpy[random_y : random_y + patch_size, random_x : random_x + patch_size] = img_patch
    display_image('2 - e - Center Patch Placed Random %d, %d' % (random_x, random_y), img_cpy)  

    # 2f: Draw random rectangles and ellipses
    
    for _ in range(10):
        # Random top-left and bottom-right corners for rectangles
        top_left = (random.randint(0, cols-1), random.randint(0, rows-1))
        bottom_right = (random.randint(top_left[0], cols-1), random.randint(top_left[1], rows-1))
        
        # Random color for rectangles
        rect_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv.rectangle(img_cpy, top_left, bottom_right, rect_color, -1)  # Fill the rectangle

    for _ in range(10):
        # Random center, axes, and angle for ellipses
        center = (random.randint(0, cols-1), random.randint(0, rows-1))
        axes = (random.randint(10, 100), random.randint(10, 100))  # Major and minor axes
        angle = random.randint(0, 360)
        
        # Random color for ellipses
        ellipse_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv.ellipse(img_cpy, center, axes, angle, 0, 360, ellipse_color, -1) 
        display_image('2 - f - Rectangles and Ellipses', img_cpy)
       
    # destroy all windows
    cv.destroyAllWindows()
