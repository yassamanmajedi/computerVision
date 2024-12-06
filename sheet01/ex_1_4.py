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

# Function to apply Gaussian blur with a given sigma
def apply_gaussian_blur(gray_image, sigma):
    return cv2.GaussianBlur(gray_image, (0, 0), sigma)

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

    # Apply Gaussian blur twice with σ = 2
    gaussian_blur_2_once = apply_gaussian_blur(gray_image, 2)
    gaussian_blur_2_twice = apply_gaussian_blur(gaussian_blur_2_once, 2)

    # Apply Gaussian blur once with σ = 2√2
    sigma_2_sqrt_2 = 2 * np.sqrt(2)
    gaussian_blur_2sqrt2_once = apply_gaussian_blur(gray_image, sigma_2_sqrt_2)

    # Display the results
    display_image('Twice Gaussian Blur', gaussian_blur_2_twice)
    display_image('Once Gaussian Blur', gaussian_blur_2sqrt2_once)

    # Compute the absolute pixel-wise difference between the two results
    max_pixel_error = compute_max_pixel_error(gaussian_blur_2_twice, gaussian_blur_2sqrt2_once)
    
    # Print the maximum pixel error
    print(f'Maximum pixel error between the two results: {max_pixel_error}')

# Run the program
if __name__ == "__main__":
    main()
