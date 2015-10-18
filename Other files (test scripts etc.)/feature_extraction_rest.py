#!/usr/bin/env

import numpy as np
import matplotlib.pyplot as plt
import scipy
import _imaging
import cv2
import json
import os
import sys
import glob
import time

from preprocess_img import *
from annotations import *
from process_img import *
from hist_img import *

from PIL import Image
from pprint import pprint
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from collections import defaultdict
from math import pi

start_time = time.time()

nc_train_path = '/export/mlrg/salavi/Desktop/Project/Original Images/Good Images/Positive Counts/Training Set'


## Open and filter image

img_list2 = [f for f in os.listdir(nc_train_path) if f.find('.jpg') > 0]                  # creates a list of all the files with the given format
img_list2 = sort(np.array(img_list2))
nbr_list_red, nbr_list_green, nbr_list_blue = [], [], []                # list of neighbouring coordinates
first_qrtr,  second_qrtr = 112, 145          # data obtained from experiment on contrast normalized images
feature_database = []

for z in range(shape(img_list2)[0]):        # shape(img_list2)[0]

    ## Process Image

    im = cv2.imread(os.path.join(train_path, img_list2[z]), cv2.CV_LOAD_IMAGE_COLOR) # comment this while using cropped images
    im_red_eq = lcn_2d(im[:,:,0],[10, 10])
    im_green_eq = lcn_2d(im[:,:,1],[10, 10])        # 1.591
    im_blue_eq = lcn_2d(im[:,:,2],[10, 10])

##    def med_filter(channel):               # median filter
##       filtered_channel = scipy.ndimage.filters.median_filter(channel, size = (4,4))
##       return filtered_channel

    im_red = med_filter(im_red_eq)
    im_green = med_filter(im_green_eq)
    im_blue = med_filter(im_blue_eq)

    cv2.normalize(im_red, im_red, 0,255,cv2.NORM_MINMAX)
    cv2.normalize(im_green, im_green, 0,255,cv2.NORM_MINMAX)
    cv2.normalize(im_blue, im_blue, 0,255,cv2.NORM_MINMAX)

    ## Make copies for background segmentation
    im_red_back = im_red.copy()
    im_green_back = im_green.copy()
    im_blue_back = im_blue.copy()
 
    ## Convert image (current and background) from RGB to YCbCr and create HSI model
 
##    # RGB to YCbCr
##    def rgb2ycbcr(red, green, blue):   # source - wiki
##       y = 0.299*red + 0.587*green + 0.114*blue
##       cb = 128 - 0.169*red - 0.331*green + 0.5*blue
##       cr = 128 + 0.5*red - 0.419*green - 0.081*blue
##       return y, cb, cr
 
    im_y, im_cb, im_cr = rgb2ycbcr(im_red, im_green, im_blue)   
    avg_inty = np.mean(im_y)
    
    if avg_inty <= first_qrtr:
       red_avg, green_avg, blue_avg = 47.85, 52.79, 57.41      
    elif (avg_inty > first_qrtr) & (avg_inty <= second_qrtr):
       red_avg, green_avg, blue_avg = 69.76, 73.33, 75.84
    else:
       red_avg, green_avg, blue_avg = 85.51, 88.47, 89.41

    # segment background (has tunable parameters)
    def sub_backgnd(tuner):      
       im_red_back[:][im_red_back[:] <= red_avg*tuner] = 255             # avg values are initialized below             
       im_green_back[:][im_green_back[:] <= green_avg*tuner] = 255
       im_blue_back[:][im_blue_back[:] <= blue_avg*tuner] = 255
    
    sub_backgnd(1.3)       # *1.15 or 1.2 - best results with LCN
    im_y_back, im_cb_back, im_cr_back = rgb2ycbcr(im_red_back, im_green_back, im_blue_back)

##    # YCbCr to HSI
##    def ycbcr2hsi(ch1, ch2, ch3):                            # ch = channel, source = US Patent
##       inty = ch1
##       hue = np.arctan(np.divide(ch3, ch2))
##       sat = np.sqrt(np.square(ch3) + np.square(ch2))
##       return inty, hue, sat

    # Convert filtered and background image channels to HSI
    im_int, im_hue, im_sat = ycbcr2hsi(im_y, im_cb, im_cr)
    im_int_back, im_hue_back, im_sat_back = ycbcr2hsi(im_y_back, im_cb_back, im_cr_back)

    # Create image differences
    im_int_diff = abs(im_int - im_int_back)
    im_hue_diff = abs(im_hue - im_hue_back)
    im_sat_diff = abs(im_sat - im_sat_back)
 

    ## Histogram plotting
 
    pixels_int, pixels_hue, pixels_sat = [], [], []
 
    # Create histogram
    def create_hist(pixels, diff_img):
       for y in range(shape(diff_img)[0]):
          for x in range(shape(diff_img)[1]):
             pixels.append(diff_img[y,x])              # make a list all pixels
       hist, bins = np.histogram(pixels, bins = 256)
       width = 0.7*(bins[1] - bins[0])                 # just for plotting the histogram (comment it out if unnecessary)
       centre = (bins[:-1] + bins[1:])/2               # just for plotting the histogram (comment it out if unnecessary)
       return hist, bins, width, centre                # omit width and centre if unnecessary

    # omit the corresponding width and centre variables from below if these are commented out above
    hist_int, bins_int, width_int, centre_int = create_hist(pixels_int, im_int_diff)
    hist_hue, bins_hue, width_hue, centre_hue = create_hist(pixels_hue, im_hue_diff)
    hist_sat, bins_sat, width_sat, centre_sat = create_hist(pixels_sat, im_sat_diff)


    ## ADAPTIVE THRESHOLDING (see patent for algorithm)
 
    # Adaptive threshold
    def adaptive_thresh(hist, bins):
    
       # Find the total number of pixels in the image   
       N, search_thresh = 0, 0   
       for q in range(shape(hist)[0]):
          N += hist[q]

       # Set threshold to the default value bin
       num_pixels = 0
       req_pixels = N*0.15
       for i in range(shape(hist)[0]-1,-1,-1):
          if num_pixels < req_pixels:
             num_pixels += hist[i]
             if num_pixels > req_pixels:
                num_pixels -= hist[i]
                i += 1
                break
       default_thresh = bins[i]
 
       # Find peak-value bin within the 1st 30
       peak_bin = np.array(np.where(np.amax(hist[0:30])))[0][0]
 
       # Std deviation about the peak bin (have doubts)
       std_dev = np.std(hist)
    
       # Find the minimum bin size (0.15% of the total pixels)
       min_bin = N*0.0015

       # Calculate search threshold
       thresh = hist[0] + std_dev*0.5
       num_pixels2 = 0
    
       for j in range(1, shape(hist)[0]):
          if num_pixels2 < thresh:
             num_pixels2 += hist[j-1]
             if num_pixels2 >= thresh:
                num_pixels2 -= hist[j-1]
                j -= 1
                break
       dest_bin = 0
    
       for bin in range(j):
          if hist[bin+1] < min_bin:
             dest_bin = 0
             search_thresh_int = bins_int[dest_bin]
          else:
             dest_bin = bin + 1
             search_thresh = bins[dest_bin]
             break
       return N, search_thresh
 
    # Find adaptive search threshold for each of the HSI channels
    N1, search_thresh_int = adaptive_thresh(hist_int, bins_int)
    N2, search_thresh_hue = adaptive_thresh(hist_hue, bins_hue)
    N3, search_thresh_sat = adaptive_thresh(hist_sat, bins_sat)


    ## Region labeling (connected components labeling)

    inty, hue, sat = im_int_diff.copy(), im_hue_diff.copy(), im_sat_diff.copy()

    # Binary segmentation of foreground and background
    inty[:,:][inty[:,:] <= search_thresh_int] = False 
    inty[:,:][inty[:,:] > search_thresh_int] = True 
    hue[:,:][hue[:,:] <= search_thresh_hue] = False 
    hue[:,:][hue[:,:] > search_thresh_hue] = True 
    sat[:,:][sat[:,:] <= search_thresh_sat] = False 
    sat[:,:][sat[:,:] > search_thresh_sat] = True 

    im_combinedOR = np.logical_or(inty, hue, sat)   # Logical OR operation to combine all the foreground objects from different channels

    # Binary opening and closing to remove small particles and fill small holes respectively
    open_img = ndimage.binary_opening(im_combinedOR, structure = np.ones((5,5))).astype(np.int)  # (5,5) kernel works best
    close_img = ndimage.binary_closing(open_img)

    ## Remove object which are too big or too small to be our desired object; changable parameter (mask_size)
    mask = close_img > close_img.mean()                                     # create a mask based on pixel average
    label_im, nb_labels = ndimage.label(mask)                               # label mask
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))               # calculate size of all labeled regions of masked image

    # remove objects larger than 300 pixels
    mask_size = sizes >= 300                                                
    remove_pixel = mask_size[label_im]                                      
    label_im[remove_pixel] = False                                          

    # remove objects smaller than 50 pixels
    mask_size = sizes < 50 
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = False
    im_label = np.array(label_im > 0, dtype = uint8)            # plotting is weird, probably due to dtype conversion
    cnt, hierarchy = cv2.findContours(im_label,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)   

##    im_or = cv2.imread(img_list2[z], cv2.CV_LOAD_IMAGE_COLOR)
##    im_int = cv2.cvtColor(im_or,cv2.COLOR_BGR2GRAY)
    for h, cntr in enumerate(cnt):
        if shape(cnt[h])[0] < 5:
            print 'rectangular contour;  possible reason: region too noisy; image# ', img_list2[z], 'id# ', h
            continue
##       elif shape(cnt)[1] < 5:
##           print 'rectangular contour;  possible reason: region too noisy; image# ', img_list2[z], 'id# ', h
        else:

            # Draw BB over contours and get corner points of the detected bounding boxes (BBdt)

            ### Feature Extraction
            
            mask = np.zeros(im_int.shape, np.uint8)
            cv2.drawContours(mask,[cntr],0,255,-1)
            mean = cv2.mean(im, mask = mask)
            find = np.where(mask > 0)
            x_axis = find[1][:]
            y_axis = find[0][:]            
 
            ## Average intensity
                 
            intensity = []
##            def avg_int(ipixel, image, x_coord, y_coord):
##               for i in range(len(x_coord)):
##                   ipixel.append(image[y_coord[i],x_coord[i]])
##               mean = round(np.mean(np.array(ipixel)))
##               return mean

            avg_intensity = avg_int(intensity, im_int, x_axis, y_axis)


            ## Intensity histogram

            pixels = []
##            def create_hist(hpixel, image, x_coord, y_coord):
##               for i in range(len(x_coord)):
##                   hpixel.append(image[y_coord[i],x_coord[i]])
##               hist, bins = np.histogram(hpixel, bins = 64, range = (0.0, 255.0))
##               width = 0.7*(bins[1] - bins[0])                 # just for plotting the histogram (comment it out if unnecessary)
##               centre = (bins[:-1] + bins[1:])/2               # just for plotting the histogram (comment it out if unnecessary)
##               return hist, bins, width, centre 
 
            hist, bins, width, centre = create_hist(pixels, im_int, x_axis, y_axis)
 
            ### Conotur-based features
                 
            area = cv2.contourArea(cnt[h])                          # Area
            perimeter = cv2.arcLength(cnt[h], True)                 # Perimeter
            ellipse = cv2.fitEllipse(cnt[h])
            (centre, axes, orientation) = ellipse
            length = np.max(axes)                                   # Length
            width = np.min(axes)                                    # Width
            circular_fitness = (4*pi*area)/np.square(perimeter)     # Circular fitness
            elongation = length/width                               # Elongation
 
            feature_dict = {'area': area, 'perimeter': perimeter, 'length': length, 'width': width, 'circular_fitness': circular_fitness, 'elongation': elongation, 'average intensity': avg_intensity, 'intensity histogram': hist}
            feature_database.append(feature_dict)

feature_database_TrainingSet_nc = np.array(feature_database)
##feature_database_TestSet = np.array(feature_database)
##feature_database_TestSet_nc = np.array(feature_database)

print time.time() - start_time, "seconds --> Execution time"
##   # save image
##   scipy.misc.imsave('/mnt/ssd/shamir/Original Images/Good Images/No Counts/Traninig_Set_LCN_Labelled/'+ img_list2[z], im_label)

