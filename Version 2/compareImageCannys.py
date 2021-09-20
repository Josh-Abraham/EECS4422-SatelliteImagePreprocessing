#########################################################################
#                                                                       #
#                       Final Project: EECS 4422                        #
#                           By: Joshua Abraham                          #
#                                                                       #
#########################################################################

# Command to run: where 00 is replaced with whichever folder subclass you want
# 2 - 5 represent the starting and ending index of which images to test within the subclass
# display == 1 means show the output stitches, display == 0 means don't display the stitch but still do the SSIM calculation
# python compareImageCannys.py --edge-detector hed_model --path images/00 --start 2 --end 3 --display 1

# import packages
import argparse
import cv2
import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import imageProcessing
import HEDmodel
import HED_image
import statistics
import copy 
import gc
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import plotly.graph_objects as go
# import warnings filter
from warnings import simplefilter
# ignore all future warnings because of medpy
simplefilter(action='ignore', category=FutureWarning)

def loadImage(location):
	# Import image from loaction, as well as the shape of matrix
	image = cv2.imread(location)
	(H, W) = image.shape[:2]

	scale_percent = 100
	width = int(W * scale_percent / 100)
	height = int(H * scale_percent / 100)
	dim = (width, height)

	# resize image
	image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)	

	return [image, height, width]

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
	z = []
	for i in range(len(similarity[0])):
		x.append(i)
		y.append(i+0.5)
		z.append(i+1)
	
	colormap = ['seagreen', 'darkslateblue', 'silver']
	labelmap = ['Unprocessed', 'HED', 'Preprocessed']
	alpha = [1, 1, 1]
	points = [x, y, z]
	width = 0.15
	i = 0
	for similaritySet in similarity:
		ax.bar(points[i], similaritySet, width, align='center', color=colormap[i], alpha=alpha[i], label=labelmap[i])
		i = i + 1
	# Plot
	my_xticks = []
	plt.xticks(x, my_xticks)
	ax.legend(loc=0)
	plt.title('Edge Detection Algorithm Comparison')
	plt.xlabel('Edge Detection')
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


path = args["path"] + '/*.png'
listOfGT = ['images/GT']
# avg_set = [[], [], []]
# listOfPaths = ['images/00', 'images/01', 'images/02', 'images/03', 'images/04', 'images/05', 'images/06', 'images/07', 'images/08', 'images/09', 'images/10', 'images/11', 'images/12', 'images/13', 'images/14', 'images/15', 'images/16', 'images/17', 'images/18']
listOfPaths = ['images/Test']

for pathPart in listOfPaths:
	path = pathPart + '/*.png'
	verticalStitch = []
	start = args["start"]
	end = args["end"]
	shouldShow = args["display"]
	diff = end - start
	# Unprocessed, HED, Preprocessed
	similaritySets = [[], [], []]
	imageObjects = []
	i = 0
	j = 0
	x = 0
	missed_canny = []
	missed_HED = []
	missed_SPEED = []
	extra_canny = []
	extra_HED = []
	extra_SPEED = []
	for filename in glob.glob(path): #assuming jpg
		groundTruth_filename = listOfGT[0] + '\\' + filename.split('\\')[1].split('.')[0] + '_instance_id_RGB.png'
		gc.collect()
		if (i >= start and i <= end):
			print("\nEdge detection for image: " + str(j + 1) + " of " + str(diff + 1))
			[image, H, W] = loadImage(filename)
			[gt, H, W] = loadImage(groundTruth_filename)
			imageObject = imageProcessing.image(image, gt, H, W, net, i)
			simCalc = imageObject.runPreprocessing()
			imageObject.imageStitching()
			print(simCalc)
			missed_canny.append(simCalc[0][0])
			missed_HED.append(simCalc[0])
			missed_SPEED.append(simCalc[2][0])
			extra_canny.append(simCalc[0][1])
			extra_HED.append(simCalc[1])
			extra_SPEED.append(simCalc[2][1])
			
			print("_____________________________________________________")
			j = j + 1
			# if (shouldShow == 1):
			# 	if (x >= len(verticalStitch)):
			# 		verticalStitch.append(imageStitch)
			# 	else:
			# 		verticalStitch[x] = np.concatenate((verticalStitch[x], imageStitch), axis=1)
			# if (j % 9 == 0):
			# 	x = x + 1
		i = i + 1


# # Similarities
# missed_canny = statistics.mean(missed_canny)
# missed_HED = statistics.mean(missed_HED)
# missed_SPEED = statistics.mean(missed_SPEED)
# print('MISS Percentages')
# print(missed_canny)
# print(missed_HED)
# print(missed_SPEED)

# extra_canny = statistics.mean(extra_canny)
# extra_HED = statistics.mean(extra_HED)
# extra_SPEED = statistics.mean(extra_SPEED)
# print('Extra Percentages')
# print(extra_canny)
# print(extra_HED)
# print(extra_SPEED)
# 	preprocessed = statistics.mean(similaritySets[1])

# 	increasePercent = round((preprocessed - original)/original * 100, 3)
# 	amountFail = 0
# 	# for i in range(len(similaritySets[0])):
# 	# 	if (similaritySets[0][i] > similaritySets[1][i]):
# 	# 		amountFail += 1

# 	print('Preprocessing Similarity Increase Percent:', increasePercent,'%')
# 	print('Total Worse:', amountFail, 'out of', len(similaritySets[0]))
# 	if (shouldShow == 1):
# 		for i in range(len(verticalStitch)):
# 			name = "Image Set:" + str(i)
# 			cv2.imshow(name, verticalStitch[i])
# 	avg_set[0].append(copy.deepcopy(statistics.mean(similaritySets[0])))
# 	avg_set[1].append(copy.deepcopy(statistics.mean(similaritySets[1])))
# 	avg_set[2].append(copy.deepcopy(statistics.mean(similaritySets[2])))
# print(avg_set)
# plotSimilarities(avg_set)

def runMain():
	path = 'images/Test' + '/*.png'
	i = 0
	results = []
	for filename in glob.glob(path): #assuming jpg
		print(i)
		i += 1
		gc.collect()
		groundTruth_filename = listOfGT[0] + '\\' + filename.split('\\')[1].split('.')[0] + '_instance_id_RGB.png'
		[image, H, W] = loadImage(filename)
		[gt, H, W] = loadImage(groundTruth_filename)
		orig = HED_image.preprocessed(image, gt, H, W)
		original_ssim = orig.runPreprocessing()
		print(original_ssim)
		results.append(original_ssim)
		# orig.imageStitching()
	# mean_val = statistics.mean(results)
	# print(mean_val)

def runAblation():
	path = 'images/Originals' + '/*.png'
	results_original_miss = []
	results_original_extra = []
	results_noBlur2_miss = []
	results_noBlur2_extra = []
	results_noContrast_miss = []
	results_noContrast_extra = []
	results_noFuzzy_miss = []
	results_noFuzzy_extra = []
	results_noAni_miss = []
	results_noAni_extra = []
	i = 0
	for filename in glob.glob(path): #assuming jpg
		print(i)
		i += 1
		gc.collect()
		groundTruth_filename = listOfGT[0] + '\\' + filename.split('\\')[1].split('.')[0] + '_instance_id_RGB.png'
		[image, H, W] = loadImage(filename)
		[gt, H, W] = loadImage(groundTruth_filename)
		orig = HED_image.preprocessed(image, gt, H, W)
		original_ssim = orig.runPreprocessing()
		results_original_miss.append(original_ssim[0])
		results_original_extra.append(original_ssim[1])

		noBlur2 = HED_image.preprocessed(image, gt, H, W)
		noBlur_ssim = noBlur2.runPreprocessingNoBlur2()
		results_noBlur2_miss.append(noBlur_ssim[0])
		results_noBlur2_extra.append(noBlur_ssim[1])

		noContra = HED_image.preprocessed(image, gt, H, W)
		noContra_ssim = noContra.runPreprocessingNoContrast()
		results_noContrast_miss.append(noContra_ssim[0])
		results_noContrast_extra.append(noContra_ssim[1])

		no_fuzz = HED_image.preprocessed(image, gt, H, W)
		no_fuzz_ssim = no_fuzz.runPreprocessingNoFuzzy()
		results_noFuzzy_miss.append(no_fuzz_ssim[0])
		results_noFuzzy_extra.append(no_fuzz_ssim[1])

		no_ani = HED_image.preprocessed(image, gt, H, W)
		no_ani_ssim = no_ani.runPreprocessingNoAni()
		results_noAni_miss.append(no_ani_ssim[0])
		results_noAni_extra.append(no_ani_ssim[1])
	
	print('Miss')
	print(copy.deepcopy(statistics.mean(results_original_miss)))
	print(copy.deepcopy(statistics.mean(results_noBlur2_miss)))
	print(copy.deepcopy(statistics.mean(results_noContrast_miss)))
	print(copy.deepcopy(statistics.mean(results_noFuzzy_miss)))
	print(copy.deepcopy(statistics.mean(results_noAni_miss)))

	print('Extra')
	print(copy.deepcopy(statistics.mean(results_original_extra)))
	print(copy.deepcopy(statistics.mean(results_noBlur2_extra)))
	print(copy.deepcopy(statistics.mean(results_noContrast_extra)))
	print(copy.deepcopy(statistics.mean(results_noFuzzy_extra)))
	print(copy.deepcopy(statistics.mean(results_noAni_extra)))
	return ''

def runAblationNoConds():
	path = 'images/Originals' + '/*.png'
	results_contrast_miss = []
	results_contrast_extra = []
	results_noBlur2_miss = []
	results_noBlur2_extra = []
	
	i = 0
	for filename in glob.glob(path): #assuming jpg
		print(i)
		i += 1
		gc.collect()
		groundTruth_filename = listOfGT[0] + '\\' + filename.split('\\')[1].split('.')[0] + '_instance_id_RGB.png'
		[image, H, W] = loadImage(filename)
		[gt, H, W] = loadImage(groundTruth_filename)
		orig = HED_image.preprocessed(image, gt, H, W)
		original_ssim = orig.runPreprocessingContrast()
		results_contrast_miss.append(original_ssim[0])
		results_contrast_extra.append(original_ssim[1])

		noBlur2 = HED_image.preprocessed(image, gt, H, W)
		noBlur_ssim = noBlur2.runPreprocessingBlurAlways()
		results_noBlur2_miss.append(noBlur_ssim[0])
		results_noBlur2_extra.append(noBlur_ssim[1])
	
	print('Miss')
	print(copy.deepcopy(statistics.mean(results_contrast_miss)))
	print(copy.deepcopy(statistics.mean(results_noBlur2_miss)))


	print('Extra')
	print(copy.deepcopy(statistics.mean(results_contrast_extra)))
	print(copy.deepcopy(statistics.mean(results_noBlur2_extra)))

	return ''


def runOrders():
	path = 'images/Originals' + '/*.png'
	results_1_miss = []
	results_1_extra = []
	results_2_miss = []
	results_2_extra = []
	results_3_miss = []
	results_3_extra = []
	results_4_miss = []
	results_4_extra = []
	
	i = 0
	for filename in glob.glob(path): #assuming jpg
		print(i)
		i += 1
		gc.collect()
		groundTruth_filename = listOfGT[0] + '\\' + filename.split('\\')[1].split('.')[0] + '_instance_id_RGB.png'
		[image, H, W] = loadImage(filename)
		[gt, H, W] = loadImage(groundTruth_filename)
		orig = HED_image.preprocessed(image, gt, H, W)
		original_ssim = orig.runPreprocessing1()
		results_1_miss.append(original_ssim[0])
		results_1_extra.append(original_ssim[1])

		noBlur2 = HED_image.preprocessed(image, gt, H, W)
		noBlur_ssim = noBlur2.runPreprocessing2()
		results_2_miss.append(noBlur_ssim[0])
		results_2_extra.append(noBlur_ssim[1])

		noContra = HED_image.preprocessed(image, gt, H, W)
		noContra_ssim = noContra.runPreprocessing3()
		results_3_miss.append(noContra_ssim[0])
		results_3_extra.append(noContra_ssim[1])

		no_fuzz = HED_image.preprocessed(image, gt, H, W)
		no_fuzz_ssim = no_fuzz.runPreprocessing4()
		results_4_miss.append(no_fuzz_ssim[0])
		results_4_extra.append(no_fuzz_ssim[1])
	
	print('Miss')
	print(copy.deepcopy(statistics.mean(results_1_miss)))
	print(copy.deepcopy(statistics.mean(results_2_miss)))
	print(copy.deepcopy(statistics.mean(results_3_miss)))
	print(copy.deepcopy(statistics.mean(results_4_miss)))

	print('Extra')
	print(copy.deepcopy(statistics.mean(results_1_extra)))
	print(copy.deepcopy(statistics.mean(results_2_extra)))
	print(copy.deepcopy(statistics.mean(results_3_extra)))
	print(copy.deepcopy(statistics.mean(results_4_extra)))
	return ''
	# Set the blob as the input to the network and compute the edges
def fetchHEDImage(image, net, W, H):
	
	print("\tPerforming holistically-nested edge detection")
	blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H), mean=(104.00698793, 116.66876762, 122.67891434), swapRB=False, crop=False)
	t1 = time.perf_counter()
	net.setInput(blob)
	hed = net.forward()
	hed = cv2.resize(hed[0, 0], (W, H))
	t2 = time.perf_counter()
	
	return t2 - t1

def timeTest():
	path = 'images/test' + '/*.png'
	listOfGT = ['images/GT']
	times_HED = []
	times_pipeline = []
	for filename in glob.glob(path): #assuming jpg
		gc.collect()
		[image, H, W] = loadImage(filename)
		
		t_hed = fetchHEDImage(image, net, W, H)
		times_HED.append(t_hed)

		#  Pipeline timing
		groundTruth_filename = listOfGT[0] + '\\' + 'P0126_instance_id_RGB.png'
		[image, H, W] = loadImage(filename)
		[gt, H, W] = loadImage(groundTruth_filename)
		orig = HED_image.preprocessed(image, gt, H, W)
		t_pipeline = orig.runTimePreprocessing()
		times_pipeline.append(t_pipeline)
		
	return [times_HED, times_pipeline]


def runPartialAblation():
	path = 'images/Originals' + '/*.png'
	results_original = []
	results_noBlur2 = []
	results_noContrast = []
	results_noFuzzy = []
	results_noAni = []
	i = 0
	for filename in glob.glob(path): #assuming jpg
		print(i)
		i += 1
		gc.collect()
		groundTruth_filename = listOfGT[0] + '\\' + filename.split('\\')[1].split('.')[0] + '_instance_id_RGB.png'
		[image, H, W] = loadImage(filename)
		[gt, H, W] = loadImage(groundTruth_filename)
		orig = HED_image.preprocessed(image, gt, H, W)
		original_ssim = orig.runPreprocessingCheck()
		# results_original.append(original_ssim)
		print(original_ssim)
		if (not original_ssim == False):
			results_original.append(original_ssim)

			noBlur2 = HED_image.preprocessed(image, gt, H, W)
			noBlur_ssim = noBlur2.runPreprocessingNoContrast()
			results_noBlur2.append(noBlur_ssim)
	if (len(results_original) > 0):
		return [len(results_noBlur2), copy.deepcopy(statistics.mean(results_original)), copy.deepcopy(statistics.mean(results_noBlur2))]
	return [0, 0, 0]

# avg_set = [38.39018, 33.3815, 66.18043, 83.47961] # edge detectors

# colors = ['lightslategray', '#a8326f']

# colors = ['#34ebba', '#f5bf42', '#5c809c', '#a8326f']
# x = ['Original Pipeline', 'Alternate Order 1', 'Alternate Order 2', 'Alternate Order 3', 'Alternate Order 4']
# x2 = [145666, 777600, 952450, 1217712, 1745824, 2314035, 2854336, 3236464]
# x = ['Canny', 'HED Model', 'HED Model <br> & Canny', 'SPEED <br> & Canny']

# fig = go.Figure()
# fig.add_trace(go.Bar(x=x, y=avg_set, marker_color=colors))

# fig.update_layout(title='Edge Detection Algorithm Comparison', yaxis_title='SSIM to Ground Truth (%)', font=dict(
#         family="Times, monospace",
#         size=20
#     ))

# fig.update_xaxes(
#         title_text = "Edge Detector",
# 		tickfont=dict(size=22),
#         title_standoff = 25)

# fig.update_yaxes(
# 		tickfont=dict(size=22),
#         title_standoff = 25)
# # fig.update_layout(title='Pseudo-Ablation Study', xaxis_title='Augmented Pipeline', yaxis_title='SSIM to Ground Truth (%)')
# fig.show()

# runMain()


### Time test
# y1 = [1.2510403, 4.1390487, 6.0687487, 7.8344274, 10.7161304, 14.4123142, 17.5813502, 22.486492, 30.9318405]
# y2 = [0.1888579, 0.1925515, 0.1935823, 0.1969279, 0.2042058, 0.2112256, 0.2166236, 0.2500584, 0.3208138]
# x = [145666, 529177, 861462, 1071799, 1511193, 2208316, 2854336, 3444336, 5289074]
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='HED Model', line_shape='spline'))
# fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='SPEED + Canny', line_shape='spline'))
# fig.update_layout(title='Computation Time of HED Model vs. SPEED Pipeline', yaxis_title='Computation Time (seconds)', font=dict(
#         family="Times, monospace",
#         size=20
#     ))

# fig.update_xaxes(
#         title_text = "Image Size (Pixels)",
# 		tickfont=dict(size=25),
#         title_standoff = 25)

# fig.update_yaxes(
# 		tickfont=dict(size=22),
#         title_standoff = 25)

# fig.update_layout(legend=dict(
#     yanchor="top",
#     y=0.99,
#     xanchor="left",
#     x=0.01,
# ))

# fig.show()


# colors = ['#a8326f', 'lightslategray',  'lightslategray',  'lightslategray',  'lightslategray']


# x = ['Full Pipeline', 'No Conditional <br> Blur', 'No Conditional <br>Contrast', 'No FHH', 'No Anisotropic <br> Diffusion']
# y = [83.47961, 79.79742, 82.56239, 72.08572, 73.98236]
# fig = go.Figure()
# fig.add_trace(go.Bar(x=x, y=y, marker_color=colors))

# fig.update_layout(title='Ablation Study', yaxis_title='SSIM to Ground Truth (%)', font=dict(
#         family="Times, monospace",
#         size=20
#     ))

# fig.update_xaxes(
#         title_text = "Augmented Pipeline",
# 		tickfont=dict(size=23),
#         title_standoff = 25)

# fig.update_yaxes(
# 		tickfont=dict(size=22),
#         title_standoff = 25)
# # fig.update_layout(title='Pseudo-Ablation Study', xaxis_title='Augmented Pipeline', yaxis_title='SSIM to Ground Truth (%)')
# fig.show()



# Configuration Test


# x = ['Original Pipeline', 'Alternate Order 1', 'Alternate Order 2', 'Alternate Order 3', 'Alternate Order 4']
# x2 = [145666, 777600, 952450, 1217712, 1745824, 2314035, 2854336, 3236464]
# x = ['Canny', 'HED Model', 'HED Model <br> & Canny', 'SPEED <br> & Canny']

# fig = go.Figure()
# fig.add_trace(go.Bar(x=x, y=avg_set, marker_color=colors))

# fig.update_layout(title='Edge Detection Algorithm Comparison', yaxis_title='SSIM to Ground Truth (%)', font=dict(
#         family="Times, monospace",
#         size=20
#     ))

# fig.update_xaxes(
#         title_text = "Edge Detector",
# 		tickfont=dict(size=22),
#         title_standoff = 25)

# fig.update_yaxes(
# 		tickfont=dict(size=22),
#         title_standoff = 25)
# # fig.update_layout(title='Pseudo-Ablation Study', xaxis_title='Augmented Pipeline', yaxis_title='SSIM to Ground Truth (%)')
# fig.show()

# runMain()
# runAblation()
# runAblationNoConds()