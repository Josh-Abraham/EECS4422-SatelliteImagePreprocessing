import time
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from medpy.filter.smoothing import anisotropic_diffusion
import math
import gc
import copy

class preprocessed:
    def __init__(self, image, ground_truth, H, W):
        scale_percent = 90 # percent of original size
        if (W > 3500 or H > 3500):
            scale_percent = 40 # percent of original size
        width = int(W * scale_percent / 100)
        height = int(H * scale_percent / 100)
        dim = (width, height)
  
		# resize image
        self.processedImage = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        gt = cv2.resize(ground_truth, dim, interpolation = cv2.INTER_AREA)
        self.groundTruthCanny = self.fetchCannyImage(gt)

    def fetchCannyImage(self, image):
        # Convert the image to grayscale from BGR, blur it, and perform Canny
        print("\tPerforming Canny edge detection")
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        canny = cv2.Canny(image, 30, 150)

        return canny

    # Perform the SSIM Calcutlation
    def imageSimilarityCalculation(self, image):
        (score, diff) = ssim(self.groundTruthCanny, image, gaussian_weights = True, full=True)
        for row in range (len(diff)-1):
            for col in range (len(diff[row])-1):
                if (diff[row][col] > 0):
                    diff[row][col] = 1
                else:
                    diff[row][col] = 0
        sim =  1-diff
        missed_edges = self.groundTruthCanny*sim
        image = image / 255
        extra_edges = image*sim
        sum_missed = 0
        sum_gt = 0
        sum_extra = 0
        for row in range (len(missed_edges)-1):
            sum_missed += sum(missed_edges[row])
            sum_gt += sum(self.groundTruthCanny[row])
            sum_extra += sum(extra_edges[row])
            
        sum_missed = sum_missed/sum_gt
        sum_extra = sum_extra/(sum_gt/255)
        return [(sum_missed)*100, (sum_extra)*100]
    
    # Determine the image distribution for conditional filtering
    def histogramValues(self, bins):
        imageRange = np.histogram(self.processedImage, bins, range = (0, 255))
        return imageRange

    # Image Preprocessing Techniques
    # Bluring
    def blur(self):
        kernel = np.ones((3,3),np.float32)/9
        # self.processedImage = cv2.filter2D(self.processedImage,-1,kernel)
        self.processedImage = cv2.GaussianBlur(self.processedImage,(3,3),1.5)

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
            return True
        return False

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
        histogram = self.histogramValues(40)[0]
        upper = sum(histogram[34:39])
        lower = sum(histogram[0:5])
        # Conditionally based off of upper(right) skew vs lower(left) skew
        contrast = False
        if (upper/(lower + 1) < 0.07):
            self.contrastNormalization(1.2)
            contrast = True
        histogram = self.histogramValues(40)[0]
        upper = sum(histogram[34:39])
        lower = sum(histogram[0:5])
        # Conditionally based off of lower skew vs upper skew
        if (lower/ (upper + 1) < 0.07):
            self.contrastNormalization(0.85)
            contrast = contrast and True

        return contrast
        
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
        greyLevel = np.power(greyLevel, 0.5) 
        newImage = (150/(math.exp(-1) - 1))*(np.exp(-greyLevel) - 1)
        self.processedImage = newImage.astype('uint8')

    # Apply an anisotropic diffusion to reduce image noise
    def anisotropic(self):
        img_filtered = anisotropic_diffusion(self.processedImage)
        self.processedImage = img_filtered.astype('uint8')

    def fetchCannyImage(self, image):
            # Convert the image to grayscale from BGR, blur it, and perform Canny
            # blurred = cv2.GaussianBlur(image, (3, 3), 0)
            canny = cv2.Canny(image, 30, 150)

            return canny

    # Base Call to apply all preprocessing techniques
    def runPreprocessing(self):
        self.whiteBalance()
        self.anisotropic()
        self.contrastAdjustment()
        self.fuzzyHist()
        self.medianBlur()
        self.blur()
        self.shouldBlurTwice()
        new_image = self.fetchCannyImage(self.processedImage)
        # Calculate a new SSIM for the preprocessed image
        
        return self.imageSimilarityCalculation(new_image)

    def runPreprocessingNoBlur2(self):
        self.whiteBalance()
        self.anisotropic()
        self.contrastAdjustment()
        self.fuzzyHist()
        self.medianBlur()
        self.blur()
        # self.shouldBlurTwice()
        new_image = self.fetchCannyImage(self.processedImage)
        # Calculate a new SSIM for the preprocessed image
        
        return self.imageSimilarityCalculation(new_image)

    def runPreprocessingNoContrast(self):
        self.whiteBalance()
        self.anisotropic()
        # self.contrastAdjustment()
        self.fuzzyHist()
        self.medianBlur()
        self.blur()
        self.shouldBlurTwice()
        new_image = self.fetchCannyImage(self.processedImage)
        # Calculate a new SSIM for the preprocessed image
        
        return self.imageSimilarityCalculation(new_image)

    def runPreprocessingNoFuzzy(self):
        self.whiteBalance()
        self.anisotropic()
        self.contrastAdjustment()
        # self.fuzzyHist()
        self.medianBlur()
        self.blur()
        self.shouldBlurTwice()
        new_image = self.fetchCannyImage(self.processedImage)
        # Calculate a new SSIM for the preprocessed image
        
        return self.imageSimilarityCalculation(new_image)

    def runPreprocessingNoAni(self):
        # blurring at end
        self.whiteBalance()
        # self.anisotropic()
        self.contrastAdjustment()
        self.fuzzyHist()
        self.medianBlur()
        self.blur()
        self.shouldBlurTwice()
        
        new_image = self.fetchCannyImage(self.processedImage)
        # Calculate a new SSIM for the preprocessed image
        
        return self.imageSimilarityCalculation(new_image)


    
    def runPreprocessingCheck(self):
        self.whiteBalance()
        self.anisotropic()
        if(self.contrastAdjustment()):
            self.fuzzyHist()
            self.medianBlur()
            self.blur()
            self.shouldBlurTwice()
            new_image = self.fetchCannyImage(self.processedImage)
            
            return self.imageSimilarityCalculation(new_image)
        return False

    def runTimePreprocessing(self):
        t1 = time.perf_counter()
        self.whiteBalance()
        self.anisotropic()
        self.medianBlur()
        self.blur()
        # Only double blur if there is still a lot of variation
        self.shouldBlurTwice()
        self.fuzzyHist()
        self.contrastAdjustment()
        new_image = self.fetchCannyImage(self.processedImage)
        t2 = time.perf_counter()
        # Calculate a new SSIM for the preprocessed image
        
        return t2 - t1

    def contrastAlways(self):
        self.contrastNormalization(1.2)
        self.contrastNormalization(0.85)

    # Ignoring conditionals
    def runPreprocessingContrast(self):
        self.whiteBalance()
        self.anisotropic()
        self.contrastAlways()
        self.fuzzyHist()
        self.medianBlur()
        self.blur()
        self.shouldBlurTwice()
        new_image = self.fetchCannyImage(self.processedImage)
        # Calculate a new SSIM for the preprocessed image
        
        return self.imageSimilarityCalculation(new_image)

    def runPreprocessingBlurAlways(self):
        self.whiteBalance()
        self.anisotropic()
        self.contrastAdjustment()
        self.fuzzyHist()
        self.medianBlur()
        self.blur()
        self.blur()
        new_image = self.fetchCannyImage(self.processedImage)
        # Calculate a new SSIM for the preprocessed image
        
        return self.imageSimilarityCalculation(new_image)
    
    ## Pipeline Variations
    def runPreprocessing1(self):
        self.whiteBalance()
        self.anisotropic()
        self.medianBlur()
        self.blur()
        self.shouldBlurTwice()
        self.fuzzyHist()
        self.contrastAdjustment()
        new_image = self.fetchCannyImage(self.processedImage)
        # Calculate a new SSIM for the preprocessed image
        
        return self.imageSimilarityCalculation(new_image)

    def runPreprocessing2(self):
        self.whiteBalance()
        self.contrastAdjustment()
        self.fuzzyHist()
        self.blur()
        self.shouldBlurTwice()
        self.medianBlur()
        self.anisotropic()
        new_image = self.fetchCannyImage(self.processedImage)
        # Calculate a new SSIM for the preprocessed image
        
        return self.imageSimilarityCalculation(new_image)

    def runPreprocessing3(self):
        self.whiteBalance()
        self.fuzzyHist()
        self.anisotropic()
        self.medianBlur()
        self.blur()
        self.shouldBlurTwice()
        self.contrastAdjustment()
        new_image = self.fetchCannyImage(self.processedImage)
        # Calculate a new SSIM for the preprocessed image
        
        return self.imageSimilarityCalculation(new_image)

    def runPreprocessing4(self):
        self.whiteBalance()
        self.medianBlur()
        self.blur()
        self.shouldBlurTwice()
        self.contrastAdjustment()
        self.fuzzyHist()
        self.anisotropic()
        new_image = self.fetchCannyImage(self.processedImage)
        # Calculate a new SSIM for the preprocessed image
        
        return self.imageSimilarityCalculation(new_image)
