import cv2 as cv
import numpy as np
import time
from numpy.random import randint


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows() 

def get_integral_img(img):
    """
    Computes the integral image iteratively.
    :param img: input image
    :return: integral image
    """
    img_integral = np.zeros((img.shape[0] + 1, img.shape[1] + 1), dtype=int)

    # Compute integral image
    for i in range(1, img_integral.shape[0]):
        for j in range(1, img_integral.shape[1]):
            img_integral[i, j] = (img[i - 1, j - 1] +
                                   img_integral[i - 1, j] +
                                   img_integral[i, j - 1] -
                                   img_integral[i - 1, j - 1])

    return img_integral

def get_mean_val_naive(img, rand_coords=[[0, 0]], size=None, verbose=True):
    if size is None:
        size = (img.shape[0], img.shape[1])

    # Run through randomized coords
    for (x, y) in rand_coords:
        sum_val = 0
        for row in img[x: x + size[0], y: y + size[1]]:
            sum_val += np.sum(row)
        mean = sum_val / (size[0] * size[1])
        if verbose:
            print(f"Mean using simple summation ({x}, {y}): {mean}")

def get_mean_val_own(img, rand_coords=[[0, 0]], size=None, verbose=True):
    if size is None:
        size = (img.shape[0], img.shape[1])

    # Get integral image
    img_integral = get_integral_img(img)

    # Run through randomized coords
    for (x, y) in rand_coords:
        sum_val = img_integral[x + size[0], y + size[1]]
        if x > 0:
            sum_val -= img_integral[x, y + size[1]]
        if y > 0:
            sum_val -= img_integral[x + size[0], y]
        if x > 0 and y > 0:
            sum_val += img_integral[x, y]
        mean = sum_val / (size[0] * size[1])
        if verbose:
            print(f"Mean using own integral implementation ({x}, {y}): {mean}")

def get_mean_val_cv(img, rand_coords=[[0, 0]], size=None, verbose=True):
    if size is None:
        size = (img.shape[0], img.shape[1])

    # Get integral image
    img_integral = cv.integral(img)

    # Run through randomized coords
    for (x, y) in rand_coords:
        sum_val = img_integral[x + size[0], y + size[1]]
        if x > 0:
            sum_val -= img_integral[x, y + size[1]]
        if y > 0:
            sum_val -= img_integral[x + size[0], y]
        if x > 0 and y > 0:
            sum_val += img_integral[x, y]
        mean = sum_val / (size[0] * size[1])
        if verbose:
            print(f"Mean using cv integral implementation ({x}, {y}): {mean}")

def time_function(func, img):
    # Get randomized patches
    np.random.seed(19)
    patch_size = (100, 100)
    n = 7
    rand_coords = np.dstack(
        [randint(low=0, high=img.shape[0] - patch_size[0] + 1, size=n), 
         randint(low=0, high=img.shape[1] - patch_size[1] + 1, size=n)]
        )[0]

    # Time function
    start = time.time()
    func(img, rand_coords, patch_size, verbose=False)
    end = time.time()
    print(f"Elapsed time for {func.__name__}: {end - start:.6f} seconds")

def main():
    # Load image and convert to gray
    img_path = 'bonn.png'
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Task 3.1a
    img_integral_own = get_integral_img(img)
    img_integral_display = cv.normalize(img_integral_own, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    display_image("3.1 - Rectangles and Integral Images", img_integral_display)

    # Task 3.1b
    get_mean_val_naive(img, verbose=True)
    get_mean_val_cv(img, verbose=True)
    get_mean_val_own(img, verbose=True)

    # Task 3.1c
    time_function(get_mean_val_naive, img)
    time_function(get_mean_val_cv, img)
    time_function(get_mean_val_own, img)

if __name__ == "__main__":
    main()
