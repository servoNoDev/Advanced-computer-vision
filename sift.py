import numpy as np
from skimage import transform
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix
class Sift:
    def __init__(self, image: np.ndarray):
        """
        Initialize SIFT object with the input image.

        Parameters:
            image (np.ndarray): Input image for SIFT processing.
        """
        self.image = image
        self.sigmas_array = [1.6,  np.sqrt(2)*1.6, 2*1.6, np.sqrt(2)*2*1.6, 3*1.6]
        self.octave_scale_multipliers_array = [2, 1, .5]
        self.scale_space_global_array = []
        self.dog_global_array = []
        self.octave_keypoints_global_array = []

    def octave_processing(self, image, octave_scale_multiplier, sigmas, visualization=False):
        """
        Process an octave of the input image to compute scale space, Difference of Gaussians (DoG),
        and detect keypoints.

        Parameters:
            image (np.ndarray): Input image for octave processing.
            octave_scale_multiplier (float): Multiplier for rescaling the image to form the octave.
            sigmas (list): List of sigma values for Gaussian blurring at different scales.
            visualization (bool, optional): Whether to visualize intermediate steps. Defaults to False.
        """
        img_rescaled = transform.rescale(image, octave_scale_multiplier)

        # Generate scale space
        scale_space_array = np.zeros(shape=(img_rescaled.shape[0], img_rescaled.shape[1], len(sigmas)))
        for i, sigma in enumerate(sigmas):
            scale_space_array[:, :, i] = gaussian_filter(img_rescaled, sigma)
        self.scale_space_global_array.append(scale_space_array)

        # Visualization of scale space
        if visualization:
            self.visualize_scale_space(scale_space_array)

        # Compute DoG
        dog_array = np.diff(scale_space_array, axis=2)
        self.dog_global_array.append(dog_array)

        # Visualization of DoGs
        if visualization:
            self.visualize_dogs(dog_array, sigmas)

        # Find keypoints
        keypoints = self.find_keypoints(dog_array)

        keypoints = transform.rescale(keypoints, 1/octave_scale_multiplier)

        # Store keypoints
        self.octave_keypoints_global_array.append(keypoints)

        # Visualization of keypoints
        if visualization:
            self.visualize_keypoints(image, keypoints)

        print("Number of keypoints candidates:", np.sum(keypoints))

    def find_keypoints(self, dog_array):
        """
        Find keypoints from the Difference of Gaussians (DoG) array.

        Parameters:
            dog_array (np.ndarray): Difference of Gaussians (DoG) array.

        Returns:
            np.ndarray: Binary mask indicating keypoints.
        """
        dog_max_filt = maximum_filter(dog_array, (3, 3, 3))
        dog_min_filt = minimum_filter(dog_array, (3, 3, 3))

        local_max = (dog_array == dog_max_filt)
        local_min = (dog_array == dog_min_filt)

        keypoints = transform.resize(
            np.logical_or(np.sum(local_max[:, :, 1:4], axis=2) > 0, np.sum(local_min[:, :, 1:4], axis=2) > 0),
            output_shape=self.image.shape)

        return keypoints

    def visualize_scale_space(self, scale_space_array):
        """
        Visualize the scale space array.

        Parameters:
            scale_space_array (np.ndarray): Scale space array.
        """
        fig, axs = plt.subplots(2, 3, figsize=(20, 10))
        for i, sigma in enumerate(self.sigmas_array):
            axs[i // 3, i % 3].imshow(scale_space_array[:, :, i], cmap="gray")
            axs[i // 3, i % 3].set_title(f"Scale {i} - Sigma at {sigma:.2f}")
        plt.suptitle("Scale space in one Octave")
        plt.show()

    def visualize_dogs(self, dog_array, sigmas):
        """
        Visualize the Difference of Gaussians (DoG) array.

        Parameters:
            dog_array (np.ndarray): Difference of Gaussians (DoG) array.
            sigmas (list): List of sigma values for Gaussian blurring.
        """
        fig, axs = plt.subplots(2, 3, figsize=(20, 10))
        for i, sigma in enumerate(sigmas[:-1]):
            axs[i // 3, i % 3].imshow(dog_array[:, :, i], cmap="gray")
            axs[i // 3, i % 3].set_title(f"DOG between {sigmas[i+1]:.2f} and {sigmas[i]:.2f}")
        plt.suptitle("Difference of Gaussians (DoG) in one Octave")
        plt.show()

    def visualize_keypoints(self, image, keypoints):
        """
        Visualize keypoints on the original image.

        Parameters:
            image (np.ndarray): Original image.
            keypoints (np.ndarray): Binary mask indicating keypoints.
        """
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(image, cmap="gray")
        rr, cc = np.where(keypoints)
        for r, c in zip(rr, cc):
            circle1 = plt.Circle((c, r), 2, color='g', clip_on=False)
            ax.add_patch(circle1)
        plt.title("Detected Keypoints")
        plt.show()

    @staticmethod
    def key_point_filter_image_frame(img_shape):
        """
        Generate a mask for filtering keypoints near the image frame.

        Parameters:
            img_shape (tuple): Shape of the input image.

        Returns:
            np.ndarray: Mask for filtering keypoints.
        """
        mask = np.zeros(shape=img_shape)
        mask[1:-1, 1:-1] = True
        return mask

    @staticmethod
    def key_point_filter_contrast(difference_of_gaussians, thr: float):
        """
        Filter keypoints based on contrast threshold.

        Parameters:
            difference_of_gaussians (np.ndarray): Difference of Gaussians (DoG) array.
            thr (float): Contrast threshold value.

        Returns:
            np.ndarray: Binary mask indicating keypoints passing the contrast threshold.
        """
        contrast_threshold = np.abs(difference_of_gaussians[:, :, 1:3]) > thr
        return contrast_threshold[:, :, 0] * contrast_threshold[:, :, 1]

#%%
