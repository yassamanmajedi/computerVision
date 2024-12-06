import cv2
import numpy as np
import random

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

# Function to add salt-and-pepper noise to the image
def add_salt_and_pepper_noise(gray_image, noise_ratio):
    noisy_image = gray_image.copy()
    num_salt = np.ceil(noise_ratio * gray_image.size * 0.5).astype(int)
    num_pepper = np.ceil(noise_ratio * gray_image.size * 0.5).astype(int)
    
    # Add salt noise (white pixels)
    coords_salt = [np.random.randint(0, i - 1, num_salt) for i in gray_image.shape]
    noisy_image[coords_salt[0], coords_salt[1]] = 255
    
    # Add pepper noise (black pixels)
    coords_pepper = [np.random.randint(0, i - 1, num_pepper) for i in gray_image.shape]
    noisy_image[coords_pepper[0], coords_pepper[1]] = 0

    return noisy_image

# Function to compute the mean gray value distance
def compute_mean_distance(img1, img2):
    return np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))

# Main function
def main():
    # Load image and convert to grayscale
    img_path = 'bonn.png'
    img = cv2.imread(img_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Add 30% salt-and-pepper noise
    noisy_image = add_salt_and_pepper_noise(gray_image, noise_ratio=0.30)
    display_image('Noisy Image (Salt and Pepper)', noisy_image)

    # Define filter sizes to test
    filter_sizes = [1, 3, 5, 7, 9]

    # Initialize variables to store the best result
    min_mean_distance = float('inf')
    best_filter_size = None
    best_filtered_image = None

    # Test filters with different sizes
    for filter_size in filter_sizes:
        # Apply Gaussian blur
        gaussian_blur = cv2.GaussianBlur(noisy_image, (filter_size, filter_size), 0)
        
        # Apply stack blur (approximate it using a box blur)
        stack_blur = cv2.blur(noisy_image, (filter_size, filter_size))
        
        # Apply bilateral filter
        bilateral_blur = cv2.bilateralFilter(noisy_image, filter_size, 75, 75)
        #The 75 values allow for a good balance between smoothing and edge preservation.

        # Compute mean gray value distance to the original image
        gaussian_distance = compute_mean_distance(gray_image, gaussian_blur)
        stack_blur_distance = compute_mean_distance(gray_image, stack_blur)
        bilateral_distance = compute_mean_distance(gray_image, bilateral_blur)

        print(f'Filter size {filter_size}:')
        print(f'  Gaussian blur distance: {gaussian_distance}')
        print(f'  Stack blur distance: {stack_blur_distance}')
        print(f'  Bilateral filter distance: {bilateral_distance}')
        
    
        if gaussian_distance < min_mean_distance:
            min_mean_distance = gaussian_distance
            best_filter_size = filter_size
            best_filtered_image = gaussian_blur

        if stack_blur_distance < min_mean_distance:
            min_mean_distance = stack_blur_distance
            best_filter_size = filter_size
            best_filtered_image = stack_blur

        if bilateral_distance < min_mean_distance:
            min_mean_distance = bilateral_distance
            best_filter_size = filter_size
            best_filtered_image = bilateral_blur

    # Display the best filtered image
    print(f'Best filter size: {best_filter_size} with a mean distance of {min_mean_distance}')
    display_image(f'Best Filtered Image (size={best_filter_size})', best_filtered_image)

# Run the program
if __name__ == "__main__":
    main()
