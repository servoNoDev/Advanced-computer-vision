import numpy as np
from skimage import transform
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter,minimum_filter

class Sift:
    def __init__(self, image: np.ndarray):
        self.image = image
        self.sigmas_array = [np.sqrt(2) / 2,
                             1,
                             np.sqrt(2),
                             2,
                             2 * np.sqrt(2),
                             2 * 2]
        self.octave_scale_multipliers_array = [2, 1, .5]
        # print("Default sigmas value is loaded: " + self.sigmas_array)
        # print("Default octave scale multiplier value is loaded: " + self.octave_scale_multiplier)
        self.scale_space_global_array = []
        self.dog_global_array = []
        self.octave_keypoins_global_array = []

    def octave_processing(self, image, octave_scale_multiplier, sigmas, visualization=False):
        img_rescaled = transform.rescale(image,octave_scale_multiplier)

        # Generate scale space
        scale_space_array = np.zeros(shape=(img_rescaled.shape[0], img_rescaled.shape[1], len(sigmas)))
        for i, sigma in zip(range(scale_space_array.shape[2]), sigmas):
            scale_space_array[:, :, i] = gaussian_filter(img_rescaled, sigma)
        self.scale_space_global_array.append(scale_space_array)

        # Visualization of scale
        if visualization:
            ig, axs = plt.subplots(2, 3, figsize=(20, 10))
            for i, sigma in zip(range(scale_space_array.shape[2]), sigmas):
                axs[i // 3, i % 3].imshow(scale_space_array[:, :, i], cmap="gray")
                axs[i // 3, i % 3].set_title("Scale {} - Sigma at {:.2f}".format(i, sigma))
            plt.suptitle("Scale space in one Octave")

        # Calculate DoG - Difference of gaussians
        dog_array = np.zeros(shape=(scale_space_array.shape[0],scale_space_array.shape[1],scale_space_array.shape[2]-1))
        for i in range(dog_array.shape[2]):
            dog_array[:,:,i] = scale_space_array[:,:,i]-scale_space_array[:,:,i+1]
        self.dog_global_array.append(dog_array)

        # Visualization of DoGs
        if visualization:
            fig, axs = plt.subplots(2, 3,figsize=(20,10))
            for i in range(dog_array.shape[2]):
                axs[i // 3,i % 3].imshow(dog_array[:,:,i], cmap="gray")
                axs[i // 3,i % 3].set_title("DOG between {:.2f} and {:.2f}".format(sigmas[i+1],sigmas[i]))
            plt.suptitle("DoGs in one Octave")

        # Find extremas
        dog_max_filt = maximum_filter(dog_array,(3,3,3))
        dog_min_filt = minimum_filter(dog_array,(3,3,3))

        local_max = (dog_array==dog_max_filt)
        local_min = (dog_array==dog_min_filt)

        # If in position of original image exist one and more extremes, we marked this point as keypoint.
        keypoints = transform.resize(np.logical_or(np.sum(local_max[:,:,1:4],axis=2)>0, np.sum(local_min[:,:,1:4],axis=2)>0), output_shape=image.shape)
        # keypoints = transform.rescale(keypoints, 1/octave_scale_multiplier)
        self.octave_keypoins_global_array.append(keypoints)

        if visualization:
            plt.figure(figsize=(20,20))
            fig, ax = plt.subplots()
            rr, cc = np.where(keypoints)
            ax.imshow(image, cmap="gray")
            for r,c in zip(rr,cc):
                circle1 = plt.Circle((c, r), 2, color='g', clip_on=False)
                ax.add_patch(circle1)
        print("Number of keypoinst candidate: " + str(sum(sum(keypoints))))

    def key_point_filter_image_frame(self, img_shape):
        mask = np.zeros(shape=img_shape)
        mask[1:-1,1:-1] = True
        return mask

    def key_point_filter_contrast(self,difference_of_gaussians, thr: float):
        contrast_threshold = np.abs(difference_of_gaussians[:,:,1:3]) > thr
        # Later must be fixed for dynamic size array aggregation
        return contrast_threshold[:,:,0] * contrast_threshold[:,:,1]