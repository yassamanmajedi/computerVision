import cv2
import numpy as np

# Function to display image using cv2.imshow
def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to apply Gaussian blur using GaussianBlur function
def apply_gaussian_blur(gray_image, sigma):
    return cv2.GaussianBlur(gray_image, (0, 0), sigma)

# Function to apply Gaussian blur using filter2D without getGaussianKernel
def apply_filter2d(gray_image, sigma):
    # Rule of thumb: set filter half-width to about 3σ so we use 6times of sigma to make sure the kernel is big enough 
    size = int(6 * sigma + 1)
    if size % 2 == 0:
        size += 1  # Ensure size is odd
    # Manually create a Gaussian kernel
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    for x in range(size):
        for y in range(size):
            kernel[x, y] = (1/(2 * np.pi * sigma**2)) * np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
    
    # Normalize the kernel
    kernel /= np.sum(kernel)
    
    return cv2.filter2D(gray_image, -1, kernel)

# Function to apply Gaussian blur using sepFilter2D without getGaussianKernel
def apply_sep_filter2d(gray_image, sigma):
    size = int(6 * sigma + 1)
    if size % 2 == 0:
        size += 1  # Ensure size is odd
    # Manually create a 1D Gaussian kernel
    center = size // 2
    x = np.zeros(size, dtype=np.float32)
    for i in range(size):
        x[i] = (1/(np.sqrt(2 * np.pi) * sigma)) * np.exp(-(i - center)**2 / (2 * sigma**2))
    
    # Normalize the kernel
    x /= np.sum(x)
    
    return cv2.sepFilter2D(gray_image, -1, x, x) #horizental and vertical directions 

# Function to compute the maximum pixel error between two images
def compute_max_pixel_error(img1, img2):
    diff = cv2.absdiff(img1, img2)
    return np.max(diff)

# Main function
def main():
    # Load image and convert to grayscale
    img_path = 'bonn.png'
    img = cv2.imread(img_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sigma value
    sigma = 2 * np.sqrt(2)

    # Apply Gaussian blur using different methods
    gaussian_blur = apply_gaussian_blur(gray_image, sigma)
    filter2d_result = apply_filter2d(gray_image, sigma)
    sep_filter2d_result = apply_sep_filter2d(gray_image, sigma)

    # Display the images
    display_image('Gray Image', gray_image)
    display_image('Gaussian Blur', gaussian_blur)
    display_image('Filter2D', filter2d_result)
    display_image('sepFilter2D', sep_filter2d_result)

    # Compute and print the maximum pixel errors between the filtered results
    max_error1 = compute_max_pixel_error(gaussian_blur, filter2d_result)
    max_error2 = compute_max_pixel_error(gaussian_blur, sep_filter2d_result)
    max_error3 = compute_max_pixel_error(filter2d_result, sep_filter2d_result)

    print(f'Maximum pixel error between GaussianBlur and filter2D: {max_error1}')
    print(f'Maximum pixel error between GaussianBlur and sepFilter2D: {max_error2}')
    print(f'Maximum pixel error between filter2D and sepFilter2D: {max_error3}')

# Run the program
if __name__ == "__main__":
    main()
