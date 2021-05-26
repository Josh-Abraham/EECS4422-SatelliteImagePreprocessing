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
python compareImageCannys.py --edge-detector hed_model --path images/FOLDER_NAME --start STARTING_INDEX --end ENDING_INDEX --display DISPLAY_CANNY_RESULTS

FOLDER_NAME: should be replaced with a folder within the images directory that has images you wish to run the preprocessing on
STARTING_INDEX: Should be replaced with an integer value representing which image you wish to start from by index
ENDING_INDEX: Should be replaced with an integer value representing which image you wish to end at by index, inclusively
DISPLAY_CANNY_RESULTS: Should be replaced by 1 if you wish to show the image results and plot, or 0 if you only wish to calculate the average percent change

Credit:
The hed_model directory as well as the HEDmodel.py file are taken from the Holistically-Nested Edge Detection Paper and code by Saining Xie, and Zhuowen Tu

![alt text](https://github.com/Josh-Abraham/EECS4422-SatelliteImagePreprocessing/blob/master/Paper/CANNYHED.jpg?raw=true)
