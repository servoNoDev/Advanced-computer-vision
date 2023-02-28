# SIFT - Scale-Invariant Feature Transform


## External materials
- Distinctive Image Features from Scale-Invariant Keypoints [paper](../_materials/sift_highlighted.pdf)
- SIFT: Theory and Practice. I prefer :) [page link](https://aishack.in/tutorials/sift-scale-invariant-feature-transform-introduction/)
- OpenCV - Introduction to SIFT (Scale-Invariant Feature Transform) [link](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html)
- Analytics Vidhya - A Detailed Guide to the Powerful SIFT Technique for Image Matching [link](https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/)
- YouTube - First Principles of Computer Vision - Overview | SIFT Detector [link](https://www.youtube.com/watch?v=KgsHoJYJ4S8)

- Implementation in OpenCV and Python [GitHub repo](https://github.com/OpenGenus/SIFT-Scale-Invariant-Feature-Transform)
- Harris Corner Detection. [link](https://vovkos.github.io/doxyrest-showcase/opencv/sphinxdoc/page_tutorial_py_features_harris.html)

## Solution
1. Scale-space extrema detection
2. Keypoint localization
3. Keypoint assignment
4. Keypoint descriptor

### 1. Scale-space extrema detection
- Scale space gaussian filtering - detect keypoints using a cascade filtering approach
  that uses efficient algorithms to identify candidate locations that are then examined in further
  detail.
- Difference of gaussian - DoG

### 2. Keypoint localization
- Find scale space extremes (min, max):
In order to detect the local maxima and minima of D(x, y, σ), each sample point is compared to its eight neighbors in the current image and nine neighbors in the scale above and below. It is selected only if it is larger than all of these neighbors or smaller than all of them

#### Keypoint filtering
We are localize a many keypoints which are not relevant. We apply multiple keypoint filtering blunts:
- **Removing low contrast features:** This is simple. If the magnitude of the intensity (i.e., without sign) at the current pixel in the DoG image (that is being checked for minima/maxima) is less than a certain value, it is rejected.
<br>
<br>
- **Removing (keypoint) edges:** The idea is to calculate [(hessian)](https://en.wikipedia.org/wiki/Hessian_matrix) two gradients at the keypoint. Both perpendicular to each other. Based on the image around the keypoint, three possibilities exist. We used [Harris Corner Detection](https://vovkos.github.io/doxyrest-showcase/opencv/sphinxdoc/page_tutorial_py_features_harris.html) for detecting flat surface, edges and corners [math solution - chapter 4.1](../_materials/sift_highlighted.pdf). The image around the keypoint can be:
  - **A flat region:** If this is the case, both gradients will be small. 
  - **An edge:** Here, one gradient will be big (perpendicular to the edge) and the other will be small (along the edge)
  - **A "corner":** Here, both gradients will be big.




### 3. Keypoint assignment - orientation


### 4. Keypoint descriptor