#########################################################################
#                                                                       #
#                       Final Project: EECS 4422                        #
#                           By: Joshua Abraham                          #
#                                                                       #
#########################################################################

# Imports
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from medpy.filter.smoothing import anisotropic_diffusion
import math


# Define Image class that will hold the image object
class image:
	def __init__(self, image, H, W, net): #vars passed in
		# Original Image
		self.image = image
		self.H = H
		self.W = W
		# Preprocessed Image - Initally
		self.processedImage = self.image

		# HED Output
		self.hed = self.fetchHEDImage(net)
		# Original Canny
		self.canny = self.fetchCannyImage(self.processedImage)
		# Preprocessed Canny - Will be overwritten once preprocessing occurs
		self.cannyProcessedImage = self.canny
		# Transfering the HED to a Canny-like Space
		self.cannyHED = self.fetchHEDToCanny()
		# Original Image Similarity Calculation, will append the
		# preprocessed in after running preprocessing steps
		self.imageSimilarity = [self.imageSimilarityCalculation()]
		
	def fetchCannyImage(self, image):
		# Convert the image to grayscale from BGR, blur it, and perform Canny
		print("\tPerforming Canny edge detection")
		blurred = cv2.GaussianBlur(image, (3, 3), 0)
		canny = cv2.Canny(blurred, 30, 150)

		return canny

	def fetchHEDImage(self, net):
		# Create Network
		# construct a blob out of the input image for the Holistically-Nested Edge Detector
		# Code Below adapted from: Holistically-Nested Edge Detection Paper
		blob = cv2.dnn.blobFromImage(self.image, scalefactor=1.0, size=(self.W, self.H),
		mean=(104.00698793, 116.66876762, 122.67891434),
		swapRB=False, crop=False)

		# Set the blob as the input to the network and compute the edges
		print("\tPerforming holistically-nested edge detection")
		net.setInput(blob)
		hed = net.forward()
		hed = cv2.resize(hed[0, 0], (self.W, self.H))

		return hed


	def fetchHEDToCanny(self):
		# Convert HED into int type and erode to decrease image thickness
		hed = (255 * self.hed).astype("uint8")
		morph_size = 5
		# Create a structuring element for the erosion
		morph_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size,morph_size))
		# Erode to decrease HED thickness for 2nd Canny
		erodedHED = cv2.erode(hed.astype(np.uint8), morph_kern, iterations=2)
		cannyHED = cv2.Canny(erodedHED, 1, 255)

		return cannyHED

	# The addBorder function is soley used for visualizing the output
	# but is not necessary to obtain values
	def addBorder(self, image, text, offset):
		# Add border to image for visualization
		newImage =cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[1, 1, 1])
		title = np.zeros((50, newImage.shape[1], 3), np.uint8)
		title[:] = (1, 1, 1)
		# Add matrix space for a title
		newImage = np.concatenate((title, newImage), axis = 0)
		font = cv2.FONT_HERSHEY_DUPLEX
		(H, W) = newImage.shape[:2]
		W = int(W/2) - offset
		newImage = cv2.putText(newImage, text, (W,50), font, 2,(0,0,0), 3, 0)

		return newImage

	# The Image stitch functionality is only needed for visualization
	# The purpose of it is to append the various versions of the image into one larger image
	def imageStitching(self, number):
		# Stitch together the images: original, base Canny, HED, then the eroded HED into Canny
		cannyGrey = cv2.cvtColor(self.canny, cv2.COLOR_GRAY2RGB)
		cannyGrey = self.addBorder(cannyGrey, 'Original Canny', 230)
		hedGrey = cv2.cvtColor(self.hed, cv2.COLOR_GRAY2RGB)
		hedGrey = self.addBorder(hedGrey, 'Holistic', 150)
		cannyHEDGrey = cv2.cvtColor(self.cannyHED, cv2.COLOR_GRAY2RGB)
		cannyHEDGrey = self.addBorder(cannyHEDGrey, 'Holistic Edges', 250)
		cannyProcessedGrey = cv2.cvtColor(self.cannyProcessedImage, cv2.COLOR_GRAY2RGB)
		cannyProcessedGrey = self.addBorder(cannyProcessedGrey, 'Preprocessed', 240)
		normImage = self.image*(1/255)
		normImage = self.addBorder(normImage, 'Original Image: ' + str(number), 290)
		imageCol = np.concatenate((normImage, hedGrey, cannyHEDGrey, cannyGrey, cannyProcessedGrey), axis=0)
		# Rescale image stitch down so it fits onto the screen correctly
		ratio = 0.25
		scaledImage = cv2.resize(imageCol,None,fx=ratio,fy=ratio)

		return scaledImage

	# Perform the SSIM Calcutlation
	def imageSimilarityCalculation(self):
		canny = self.cannyProcessedImage
		(score, diff) = ssim(self.cannyHED, canny, full=True)
		return (score)*100

	# Determine the image distribution for conditional filtering
	def histogramValues(self, bins):
		imageRange = np.histogram(self.processedImage, bins, range = (0, 255))
		return imageRange

	# Image Preprocessing Techniques
	# Bluring
	def blur(self):
		kernel = np.ones((3,3),np.float32)/9
		self.processedImage = cv2.filter2D(self.processedImage,-1,kernel)

	# Conditional Secondary Blur
	def shouldBlurTwice(self):
		# Determine the Image Spread
		spreadOfImage = self.histogramValues(255)[0]
		# Determine the "noisyness" of an image
		# If there are a lot of colour values that have a small non-zero number of pixel values then the image should be reblurred
		spreadOfImage = np.where(np.logical_and(np.less(spreadOfImage, 36), np.not_equal(spreadOfImage,0)), 1, 0)
		# By testing I determined that if there are more then 20 pixels that satisfy this then a blurring should be done
		if (sum(spreadOfImage) > 25):
			self.blur()

	# Apply a median blur to the image to reduce salt and pepper noise
	def medianBlur(self):
		kernsize = 3
		self.processedImage = cv2.medianBlur(self.processedImage, kernsize).astype('uint8')
		# smooth the image with the filter kernel

	# Apply a normalizing contrast by a specific amount (to reskew the data left or right) without saturating the image
	def contrastNormalization(self, amount):
		tempImage = self.processedImage
		# Contrast the values by amount
		tempImage = tempImage * amount
		pMax = tempImage.max() + 1
		pMin = tempImage.min()
		diff = (255)/(pMax - pMin)
		tempImage = (tempImage - pMin)*diff
		self.processedImage = tempImage.astype('uint8')

	# This is a bidirectional normalization accounting for left and right skew
	# This is technically 2 independent filters
	def contrastAdjustment(self):
		histogram = self.histogramValues(20)[0]
		upper = sum(histogram[14:19])
		lower = sum(histogram[0:5])
		# Conditionally based off of upper(right) skew vs lower(left) skew
		if (upper/(lower + 1) < 0.07):
			self.contrastNormalization(1.2)
		histogram = self.histogramValues(20)[0]
		upper = sum(histogram[14:19])
		lower = sum(histogram[0:5])
		# Conditionally based off of lower skew vs upper skew
		if (lower/ (upper + 1) < 0.07):
			self.contrastNormalization(0.85)

	# White Balancing is a colour correction process using the LAB colour space
	# Sources:
	# https://pippin.gimp.org/image-processing/chapter-automaticadjustments.html 
	# http://therefractedlight.blogspot.com/2012/02/color-spaces-part-4-lab.html
	# https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption 
	def whiteBalance(self):
		tempImage = cv2.cvtColor(self.processedImage, cv2.COLOR_BGR2LAB)
		avg_a = np.mean(tempImage[:, :, 1])
		avg_b = np.mean(tempImage[:, :, 2])
		# Adjusting LAB values
		tempImage[:, :, 1] = tempImage[:, :, 1] - ((avg_a - 128) * (tempImage[:, :, 0] / 255.0) * 1.1)
		tempImage[:, :, 2] = tempImage[:, :, 2] - ((avg_b - 128) * (tempImage[:, :, 0] / 255.0) * 1.1)
		tempImage = cv2.cvtColor(tempImage, cv2.COLOR_LAB2BGR)
		# Convert the image to Gray for all other preprocessing techniques
		tempImage = cv2.cvtColor(tempImage, cv2.COLOR_BGR2GRAY)
		self.processedImage = tempImage.astype('uint8')

	# Fuzzy Histogram Hyperbolization is used as a local contrast adjustor
	# Where as the COntrast Normalization was globally based
	# Based on the following Papers:
	# https://www.researchgate.net/publication/285413959_Fuzzy_Histogram_Hyperbolization_for_Image_Enhancement
	# http://www.inase.org/library/2015/vienna/bypaper/BICHE/BICHE-09.pdf 
	def fuzzyHist(self):
		tempImage = self.processedImage
		minVal = tempImage.min()
		maxVal = tempImage.max()
		greyLevel = (tempImage - minVal)/(maxVal - minVal)
		greyLevel = np.power(greyLevel, 0.8) 
		newImage = (254/(math.exp(-1) - 1))*(np.exp(-greyLevel) - 1)
		self.processedImage = newImage.astype('uint8')

	# Apply an anisotropic diffusion to reduce image noise
	def anisotropic(self):
		img_filtered = anisotropic_diffusion(self.processedImage)
		self.processedImage = img_filtered.astype('uint8')

	# Base Call to apply all preprocessing techniques
	def runPreprocessing(self):
		self.whiteBalance()
		self.fuzzyHist()
		self.anisotropic()
		self.medianBlur()
		self.blur()
		# Only double blur if there is still a lot of variation
		self.shouldBlurTwice()
		self.contrastAdjustment()
		self.cannyProcessedImage = self.fetchCannyImage(self.processedImage)
		# Calculate a new SSIM for the preprocessed image
		newSimilarity = self.imageSimilarityCalculation()

		return newSimilarity
	