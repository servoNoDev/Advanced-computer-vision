# Viola-Jones Object face Framework
This is face / this is not face :)

tags:  Haar-like Features, Integral Images, the AdaBoost Algorithm, and the Cascade Classifier

## External materials
- [Video - Detecting Faces (Viola Jones Algorithm) - Computerphile](https://www.youtube.com/watch?v=uEJ71VlUmMQ)
- [Paper - Rapid Object Detection using a Boosted Cascade of Simple Features](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)
- [Blog - Understanding Face Detection with the Viola-Jones Object Detection Framework](https://towardsdatascience.com/understanding-face-detection-with-the-viola-jones-object-detection-framework-c55cc2a9da14)
- [Code - Face Detection using Viola Jones Algorithm](https://www.mygreatlearning.com/blog/viola-jones-algorithm/)
- [Code - Face classification using Haar-like feature descriptor](https://scikit-image.org/docs/dev/auto_examples/applications/plot_haar_extraction_selection_classification.html)

- [Video - AdaBoost Classifier](https://www.youtube.com/watch?v=BoGNyWW9-mE)
- https://pytorch.org

# Solution
 - it is not deep learning based solution (year 2001) :)

## Haar like futures - Manual feature extraction 
<img src="https://docs.opencv.org/3.4/haar_features.jpg">

## Integral image
<img src="https://i.stack.imgur.com/mtPHG.png">
<br>
<img src="https://miro.medium.com/max/700/1*HhO9vGKpbx9p8x7uS49v-g.png">

## AdaBoost
### Generate dataset
### Train classifiers 


## Cascade classifier

<img src="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*Hxwci_y3MQE81lmr2No3Ag.png">

- Strong and weak classificator -> Cumulative increasing of feature number.
- First level: If in tile is not detect one feature, tile will reject.
- Second lever: If in tile is not detect more then ten feature, tile wil recejt.
- Chain rule.

