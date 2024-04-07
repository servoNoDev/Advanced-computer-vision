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
        self.octave_scale_multiplier_list = []

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
        self.octave_scale_multiplier_list.append(octave_scale_multiplier)
        # print(image.shape)
        img_rescaled = transform.rescale(image, octave_scale_multiplier)
        # print(img_rescaled.shape)
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
        # print(keypoints.shape)
        # keypoints = transform.rescale(keypoints, 1/octave_scale_multiplier)
        # print(keypoints.shape)
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
    def key_point_filter_image_frame(img_shape,scale_ratio):
        """
        Generate a mask for filtering keypoints near the image frame.

        Parameters:
            img_shape (tuple): Shape of the input image.

        Returns:
            np.ndarray: Mask for filtering keypoints.
        """
        mask = np.zeros(shape=img_shape)
        # mask = np.zeros(shape=(int(img_shape[0]*scale_ratio),int(img_shape[1]*scale_ratio)))
        # print(mask.shape)

        mask[1:-1, 1:-1] = True
        return mask

    @staticmethod
    def key_point_filter_contrast(difference_of_gaussians, octave_scale_multiplier=1, thr = 0.03):
        """
        Filter keypoints based on contrast threshold.

        Parameters:
            difference_of_gaussians (np.ndarray): Difference of Gaussians (DoG) array.
            thr (float): Contrast threshold value.

        Returns:
            np.ndarray: Binary mask indicating keypoints passing the contrast threshold.
        """
        contrast_threshold = np.abs(difference_of_gaussians[:, :, 1:3]) > thr
        ret = contrast_threshold[:, :, 0] * contrast_threshold[:, :, 1]
        return transform.rescale(ret,  1/octave_scale_multiplier)

    @staticmethod
    def key_point_filter_on_edges(difference_of_gaussians, octave_scale_multiplier=1,visualization=True):
        # Calculate Hessian matrix components (second-order partial derivatives)
        Dxx, Dxy, Dyy = hessian_matrix(difference_of_gaussians, sigma=0, order='xy')

        if visualization:
            # Visualize components of the Hessian matrix
            plt.figure(figsize=(8, 15))
            plt.subplot(311)
            plt.set_cmap('hot')
            plt.imshow(Dxx**2)  # Gradient change in the Y axis
            plt.subplot(312)
            plt.imshow(Dxy**2)  # Gradient change in the XY axis
            plt.subplot(313)
            plt.imshow(Dyy**2)  # Gradient change in the X axis

        # Calculate trace and determinant of the Hessian matrix
        trace_h = Dxx + Dyy
        det_h = Dxx * Dyy - Dxy**2

        # Compute the principal curvature ratio
        r = 10  # Paper recommendation for threshold
        principal_curv_ratio = trace_h**2 / det_h

        # Filter out points on edges or ridges based on the principal curvature ratio
        # Points with a principal curvature ratio below a certain threshold (r+1)^2 / r are considered as edge points
        # print("in function")
        # print(principal_curv_ratio.shape)
        ret = transform.rescale(principal_curv_ratio < ((r + 1)**2) / r, 1/octave_scale_multiplier)
        # print(ret.shape)
        return ret

    def filter_keypoints_in_octave(self, octave_index=0, visualise=True):
        a = self.octave_keypoints_global_array[octave_index]
        b = self.key_point_filter_image_frame(img_shape=self.octave_keypoints_global_array[octave_index].shape, scale_ratio=self.octave_scale_multiplier_list[octave_index])
        c = self.key_point_filter_contrast(self.dog_global_array[octave_index],octave_scale_multiplier=self.octave_scale_multiplier_list[octave_index], thr=0.05)
        d = self.key_point_filter_on_edges(self.dog_global_array[octave_index][:,:, 0] , octave_scale_multiplier=self.octave_scale_multiplier_list[octave_index], visualization=False)
        # print(a.shape,b.shape,c.shape,d.shape)
        my_best_keypoints = a * b * c * d


        if visualise:
            rr, cc = np.where(my_best_keypoints)
            fig, ax = plt.subplots(figsize=(30,30))
            ax.imshow(self.image, cmap="gray")
            for r,c in zip(rr,cc):
                circle1 = plt.Circle((c, r), 2, color='g', clip_on=False)
                ax.add_patch(circle1)

        return my_best_keypoints
    # @staticmethod
    def get_keypoints_histgoram(self,keypoints, octave_index=0, visualise=True):
        L = self.scale_space_global_array[octave_index]

        dLdx = np.pad((L[1:L.shape[0], :] - L[0:L.shape[0]-1, :]), ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=(0, 0))
        dLdy = np.pad((L[:, 1:L.shape[1]] - L[:, 0:L.shape[1]-1]), ((0, 0), (1, 0), (0, 0)), mode='constant', constant_values=(0, 0))

        mag = np.sqrt(dLdx ** 2 + dLdy ** 2)
        orientation = np.arctan2(dLdy, dLdx)
        if visualise:
            plt.figure(figsize=(20, 20))
            plt.subplot(121)
            plt.imshow(mag[:,:,1])
            plt.title("Magnitude")
            plt.subplot(122)
            plt.imshow(orientation[:,:,1])
            plt.title("Orientation")
            plt.show()

        rr, cc = np.where(keypoints)
        points_hist = []
        for r, c in zip(rr,cc):
            mag_around_point = mag[r-1:r+2,c-1:c+2]
            orientation_around_point = orientation[r-1:r+2,c-1:c+2]

            # mag_around_point = mag[r-3:r+4,c-3:c+4]
            # orientation_around_point = orientation[r-3:r+4,c-3:c+4]
            # plt.figure()

            hist,bin_edges = np.histogram(np.rad2deg(orientation_around_point.ravel())+180,bins=np.arange(0,360,10),weights=mag_around_point.ravel())
            # hist,bin_edges = np.histogram(np.rad2deg(orientation_around_point.ravel())+180,bins=np.arange(0,360,10))

            points_hist.append(np.array(hist, dtype=np.float32))
            # print(hist)
            # plt.bar(bin_edges[:-1], hist, width = 10)
            # plt.title("Orientation for point row: {} column: {}".format(r,c))
            # plt.axhline(0.8*np.max(hist),color='red')

        points_hist = np.array(points_hist)
        return points_hist

