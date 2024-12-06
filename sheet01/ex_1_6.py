import cv2
import numpy as np

# Function to display images using OpenCV
def display_image(window_name, img):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to apply 2D filtering
def apply_2d_filter(image, kernel):
    return cv2.filter2D(image, -1, kernel)

# Function to apply separable 1D filtering
def apply_separable_filter(image, u, s, v):
    # Compute the outer product of the top singular vectors to get the 1D kernels
    u_1d = u[:, 0] * np.sqrt(s[0])  # First singular vector of U
    v_1d = v[0, :] * np.sqrt(s[0])  # First singular vector of V

    # Apply separable filtering using cv2.sepFilter2D
    return cv2.sepFilter2D(image, -1, u_1d, v_1d)

# Function to apply separable filter with the top two singular values
def apply_separable_filter_2sv(image, u, s, v):
    # Combine the first two singular values to create two sets of 1D filters
    u_1d_1 = u[:, 0] * np.sqrt(s[0])
    v_1d_1 = v[0, :] * np.sqrt(s[0])

    u_1d_2 = u[:, 1] * np.sqrt(s[1])
    v_1d_2 = v[1, :] * np.sqrt(s[1])

    # Apply two separable filters and add the results
    result_1 = cv2.sepFilter2D(image, -1, u_1d_1, v_1d_1)
    result_2 = cv2.sepFilter2D(image, -1, u_1d_2, v_1d_2)

    return result_1 + result_2

# Compute maximum pixel error between two images
def compute_max_pixel_error(img1, img2):
    diff = cv2.absdiff(img1, img2)
    return np.max(diff)

def main():
    # Read the image and convert to grayscale
    img_path = 'bonn.png'
    img = cv2.imread(img_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define the two kernels
    kernel_1 = np.array([[0.0113, 0.0838, 0.0113],
                         [0.0838, 0.6193, 0.0838],
                         [0.0113, 0.0838, 0.0113]])

    kernel_2 = np.array([[-1.7497, 0.3426, 1.1530, -0.2524, 0.9813],
                         [0.5142, 0.2211, -1.0700, -0.1894, 0.2550],
                         [-0.4580, 0.4351, -0.5835, 0.8168, 0.6727],
                         [0.1044, -0.5312, 1.0297, -0.4381, -1.1183],
                         [1.6189, 1.5416, -0.2518, -0.8424, 0.1845]])

    # Perform 2D filtering on the grayscale image with both kernels
    filtered_image_kernel_1 = apply_2d_filter(gray_image, kernel_1)
    filtered_image_kernel_2 = apply_2d_filter(gray_image, kernel_2)

    # Perform SVD on kernel 1 (taking only the highest singular value)
    u1, s1, v1 = np.linalg.svd(kernel_1)
    separable_image_kernel_1 = apply_separable_filter(gray_image, u1, s1, v1)

    # Perform SVD on kernel 2 (taking the first two highest singular values)
    u2, s2, v2 = np.linalg.svd(kernel_2)
    separable_image_kernel_2 = apply_separable_filter_2sv(gray_image, u2, s2, v2)

    # Display results
    display_image('Filtered Image (2D Kernel 1)', filtered_image_kernel_1)
    display_image('Separable Image (Kernel 1)', separable_image_kernel_1)

    display_image('Filtered Image (2D Kernel 2)', filtered_image_kernel_2)
    display_image('Separable Image (Kernel 2)', separable_image_kernel_2)

    # Compute maximum pixel error between direct and separable filtering
    max_error_kernel_1 = compute_max_pixel_error(filtered_image_kernel_1, separable_image_kernel_1)
    max_error_kernel_2 = compute_max_pixel_error(filtered_image_kernel_2, separable_image_kernel_2)

    print(f'Maximum pixel error for Kernel 1: {max_error_kernel_1}')
    print(f'Maximum pixel error for Kernel 2: {max_error_kernel_2}')

if __name__ == "__main__":
    main()
