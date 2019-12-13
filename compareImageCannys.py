#########################################################################
#                                                                       #
#                       Final Project: EECS 4422                        #
#                           By: Joshua Abraham                          #
#                                                                       #
#########################################################################

# Command to run: where 00 is replaced with whichever folder subclass you want
# 2 - 5 represent the starting and ending index of which images to test within the subclass
# display == 1 means show the output stitches, display == 0 means don't display the stitch but still do the SSIM calculation
# python compareImageCannys.py --edge-detector hed_model --path images/00 --start 2 --end 5 --display 1

# import packages
import argparse
import cv2
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import imageProcessing
import HEDmodel
import statistics
# import warnings filter
from warnings import simplefilter
# ignore all future warnings because of medpy
simplefilter(action='ignore', category=FutureWarning)

def loadImage(location):
	# Import image from loaction, as well as the shape of matrix
	print("\tLoading Image")
	image = cv2.imread(location)
	(H, W) = image.shape[:2]

	return [image, H, W]

def plotSimilarities(similarity):
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	SMALL_SIZE = 12
	SMALL_MED = 13
	MEDIUM_SIZE = 14
	BIGGER_SIZE = 16

	plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
	plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
	plt.rc('axes', labelsize=SMALL_MED)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
	x = []
	y = []
	for i in range(len(similarity[0])):
		x.append(i)
	for i in range(len(similarity[0])):
		y.append(i+0.35)
	
	similarity = np.flip(similarity)
	colormap = ['darkturquoise', 'thistle']
	labelmap = ['Preprocessed Image', 'Original Image']
	alpha = [0.8, 1]
	points = [x, y]
	width = 0.35
	i = 0
	for similaritySet in similarity:
		ax.bar(points[i], similaritySet, width, align='center', color=colormap[i], alpha=alpha[i], label=labelmap[i])
		i = i + 1
	# Plot
	ax.legend(loc=0)
	plt.title('Similarity Percentage Vs Image')
	plt.xlabel('Image Number')
	plt.ylabel('Similarity (%)')
	plt.show()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--edge-detector", type=str, required=True,
	help="path to OpenCV's deep learning edge detector")
ap.add_argument("-i", "--path", type=str, required=True,
	help="path to input file")
ap.add_argument("-s", "--start", type=int, required=True,
	help="start index for images")
ap.add_argument("-e", "--end", type=int, required=True,
	help="end index")
ap.add_argument("-p", "--display", type=int, required=True,
	help="end index")
args = vars(ap.parse_args())

# Create Network Connections based off trained model
protoPath = os.path.sep.join([args["edge_detector"],
"deploy.prototxt"])
modelPath = os.path.sep.join([args["edge_detector"],
	"hed_pretrained_bsds.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
# register our new layer with the model holding our desired image size
cv2.dnn_registerLayer("Crop", HEDmodel.HEDmodel)


path = args["path"] + '/*.jpg'
verticalStitch = []
start = args["start"]
end = args["end"]
shouldShow = args["display"]
diff = end - start
similaritySets = [[], []]
imageObjects = []
i = 0
j = 0
x = 0
for filename in glob.glob(path): #assuming jpg
	if (i >= start and i <= end):
		print("\nEdge detection for image: " + str(j + 1) + " of " + str(diff + 1))
		[image, H, W] = loadImage(filename)
		imageObjects.append(imageProcessing.image(image, H, W, net))
		newSimilarity = imageObjects[j].runPreprocessing()
		imageStitch = imageObjects[j].imageStitching(j + 1)
		similaritySets[0].append(imageObjects[j].imageSimilarity[0])
		similaritySets[1].append(newSimilarity)
		print("_____________________________________________________")
		j = j + 1
		if (shouldShow == 1):
			if (x >= len(verticalStitch)):
				verticalStitch.append(imageStitch)
			else:
				verticalStitch[x] = np.concatenate((verticalStitch[x], imageStitch), axis=1)
		if (j % 9 == 0):
			x = x + 1

		
	i = i + 1


# Similarities
original = statistics.mean(similaritySets[0])
preprocessed = statistics.mean(similaritySets[1])

increasePercent = round((preprocessed - original)/original * 100, 3)
amountFail = 0
for i in range(len(similaritySets[0])):
	if (similaritySets[0][i] > similaritySets[1][i]):
		amountFail += 1

print('Preprocessing Similarity Increase Percent:', increasePercent,'%')
print('Total Worse:', amountFail, 'out of', len(similaritySets[0]))
if (shouldShow == 1):
	for i in range(len(verticalStitch)):
		name = "Image Set:" + str(i)
		cv2.imshow(name, verticalStitch[i])
plotSimilarities(similaritySets)



