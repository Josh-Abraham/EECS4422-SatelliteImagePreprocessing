# Edge Detection for Satellite Images without Deep Networks

## Joshua Abraham, Calden Wloka


###### Satellite imagery is widely used in many application sectors, including agriculture, navigation, and urban planning. Frequently, satellite imagery involves both large numbers of images as well as high pixel counts, making satellite datasets computationally expensive to analyze. Recent approaches to satellite image analysis have largely emphasized deep learning methods. Though extremely powerful, deep learning has some drawbacks, including the requirement of specialized computing hardware and a high reliance on training data. When dealing with large satellite datasets, the cost of both computational resources and training data annotation may be prohibitive.
###### In this paper, we demonstrate that a carefully designed image pre-processing pipeline allows traditional computer vision techniques to achieve semantic edge detection in satellite imagery that is competitive with deep learning methods at a fraction of the resource costs. We focus on the task of semantic edge detection due to its general-purpose usage in remote sensing, including the detection of natural and man-made borders, coast lines, roads, and buildings.



<p align="center">
  <img src="https://raw.githubusercontent.com/Josh-Abraham/EECS4422-SatelliteImagePreprocessing/master/Paper/stitch.png" width="350" title="hover text">
</p>


To run the following code you must have the following structure:

--compareImageCannys.py
--HEDmodel.py
--imageProcessing.oy
->hed_model
----deploy.prototxt
----hed_pretrained_bsds.caffemodel
->images
----00
----01


To run:
```
python compareImageCannys.py --edge-detector hed_model --path images/FOLDER_NAME --start STARTING_INDEX --end ENDING_INDEX --display DISPLAY_CANNY_RESULTS
```
FOLDER_NAME: should be replaced with a folder within the images directory that has images you wish to run the preprocessing on
STARTING_INDEX: Should be replaced with an integer value representing which image you wish to start from by index
ENDING_INDEX: Should be replaced with an integer value representing which image you wish to end at by index, inclusively
DISPLAY_CANNY_RESULTS: Should be replaced by 1 if you wish to show the image results and plot, or 0 if you only wish to calculate the average percent change

Credit:
The hed_model directory as well as the HEDmodel.py file are taken from the Holistically-Nested Edge Detection Paper and code by Saining Xie, and Zhuowen Tu


