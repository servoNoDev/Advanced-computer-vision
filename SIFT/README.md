# SIFT - Scale-Invariant Feature Transform


## External materials
- Distinctive Image Features from Scale-Invariant Keypoints [paper](../_materials/sift_highlighted.pdf)

- OpenCV - Introduction to SIFT (Scale-Invariant Feature Transform) [link](https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html)
- Analytics Vidhya - A Detailed Guide to the Powerful SIFT Technique for Image Matching [link](https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/)
- YouTube - First Principles of Computer Vision - Overview | SIFT Detector [link](https://www.youtube.com/watch?v=KgsHoJYJ4S8)

- Implementation in OpenCV and Python [GitHub repo](https://github.com/OpenGenus/SIFT-Scale-Invariant-Feature-Transform)

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
- Find scale space extremes (min, max): 
In order to detect the local maxima and minima of D(x, y, σ), each sample point is compared to its eight neighbors in the current image and nine neighbors in the scale above and below. It is selected only if it is larger than all of these neighbors or smaller than all of them