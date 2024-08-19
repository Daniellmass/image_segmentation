import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class KMeans:
    """
    KMeans clustering algorithm.

    Attributes:
        k (int): The number of clusters.
        max_iters (int): Maximum number of iterations for convergence.
        tol (float): Tolerance to declare convergence.
        centroids (ndarray): Coordinates of the cluster centroids.
        cluster_assignments (ndarray): Cluster assignment for each data point.
    """

    def __init__(self, k=3, max_iters=100, tol=1e-4):
        """
        Initializes the KMeans instance with the specified parameters.

        Args:
            k (int): The number of clusters.
            max_iters (int): Maximum number of iterations for convergence.
            tol (float): Tolerance to declare convergence.
        """
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        """
        Fits the KMeans model to the data X.

        Args:
            X (ndarray): The input data array of shape (n_samples, n_features).
        """
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]

        for i in range(self.max_iters):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            cluster_assignments = np.argmin(distances, axis=1)

            new_centroids = np.array([X[cluster_assignments == j].mean(axis=0) for j in range(self.k)])

            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break

            self.centroids = new_centroids

        self.cluster_assignments = cluster_assignments

    def predict(self, X):
        """
        Predicts the closest cluster for each sample in X.

        Args:
            X (ndarray): The input data array of shape (n_samples, n_features).

        Returns:
            ndarray: Cluster assignments for each sample.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        cluster_assignments = np.argmin(distances, axis=1)
        return cluster_assignments


def load_image(file_path):
    """
    Loads an image from a file path.

    Args:
        file_path (str): The path to the image file.

    Returns:
        ndarray: The loaded image as a NumPy array.
    """
    image = Image.open(file_path)
    return np.array(image)


def bgr_to_rgb(image):
    """
    Converts an image from BGR to RGB format.

    Args:
        image (ndarray): The input image in BGR format.

    Returns:
        ndarray: The image in RGB format.
    """
    return image[:, :, ::-1]


def image_to_pixel_array(image):
    """
    Converts an image to a 2D array of pixels with 3 color values (RGB).

    Args:
        image (ndarray): The input image.

    Returns:
        ndarray: A 2D array where each row represents a pixel with 3 color values.
    """
    height, width, channels = image.shape
    pixels = image.reshape(height * width, channels)
    return pixels


def segment_image(image, num_clusters):
    """
    Segments the image into clusters using KMeans clustering and assigns colors to each cluster.

    Args:
        image (ndarray): The input image.
        num_clusters (int): The number of clusters to segment the image into.

    Returns:
        tuple: Original image and the color-segmented image.
    """
    image = bgr_to_rgb(image)
    pixels = image_to_pixel_array(image)

    kmeans = KMeans(k=num_clusters)
    kmeans.fit(pixels)

    # Define colors for each cluster
    colors = np.random.randint(0, 255, (num_clusters, 3), dtype=np.uint8)

    # Assign colors to each pixel based on cluster assignment
    color_segmented_pixels = colors[kmeans.cluster_assignments]
    color_segmented_image = color_segmented_pixels.reshape(image.shape)
    return image, color_segmented_image


def save_image(image, filename):
    """
    Saves the image to a file.

    Args:
        image (ndarray): The image to save.
        filename (str): The filename to save the image as.
    """
    image = Image.fromarray(image)
    image.save(filename)


def display_images(original, segmented, original_filename, segmented_filename):
    """
    Displays the original and segmented images side by side and saves them to files.

    Args:
        original (ndarray): The original image.
        segmented (ndarray): The segmented image.
        original_filename (str): The filename to save the original image as.
        segmented_filename (str): The filename to save the segmented image as.
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(segmented)
    plt.title("Segmented Image")
    plt.axis('off')

    save_image(original, original_filename)
    save_image(segmented, segmented_filename)

    plt.show()


def main():
    """
    Main function to load example images, segment them, and display the results.
    """
    # Local image file paths
    file_paths = [
        'images/spidey2_01.jpg'  # Update with your local image path
    ]

    # Directory to save the output images
    output_directory = 'output_images/'

    k_values = [2, 4, 8, 16]  # Different values of k to test

    for k in k_values:
        for i, file_path in enumerate(file_paths):
            try:
                image = load_image(file_path)
                original, segmented = segment_image(image, num_clusters=k)
                display_images(
                    original, segmented,
                    f'{output_directory}original_k{k}_{i}.jpg',
                    f'{output_directory}segmented_k{k}_{i}.jpg'
                )
            except Exception as e:
                print(e)


if __name__ == '__main__':
    main()
