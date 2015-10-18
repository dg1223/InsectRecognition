import numpy as np
import matplotlib.pyplot as plt
import PIL
import scipy
import ImageOps as io
import _imaging
import cv2
import json
import os
import sys
import glob
import time
import logging

from PIL import Image
from pprint import pprint
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from collections import defaultdict
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import ImageGrid
from math import pi
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn import svm
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from scipy import interp
from sklearn.pipeline import Pipeline
from time import gmtime, strftime

##from skimage import filter
##from skimage import measure
##from skimage.morphology import label, closing, square
##from skimage.measure import regionprops

cd '/mnt/data/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/Training Set'


im = Image.open('132_5196.jpg')

im_fil = scipy.ndimage.filters.median_filter(im2, size = (4,3,4))

im_cur = im_fil.copy()




## Make arrays for each channel containing respective background pixels

background = []

for y in range(shape(im_red)[0]-40, shape(im_red)[0]):
   for x in range(shape(im_red)[1]-40, shape(im_red)[1]):
      background.append(im_red[y,x])     

red_avg = mean(background[:])

for y in range(shape(im2_green)[0]-40, shape(im2_green)[0]):
   for x in range(shape(im2_green)[1]-40, shape(im2_green)[1]):
      background.append(im2_green[y,x])        

green_avg = mean(background[:])

for y in range(shape(im2_blue)[0]-40, shape(im2_blue)[0]):
   for x in range(shape(im2_blue)[1]-40, shape(im2_blue)[1]):
      background.append(im2_blue[y,x])

blue_avg = mean(background[:])









In [825]: nbr_list = []

In [826]: x_cent_nbr = np.arange(x_center[0]-3,x_center[0]+4)

In [827]: y_cent_nbr = np.arange(y_center[0]-3,y_center[0]+4)

for y in y_cent_nbr:
   for x in x_cent_nbr:
      nbr_list.append(y_cur[y,x])

In [862]: x_cent_nbr = np.arange(x_center[1]-3,x_center[1]+4)

In [863]: y_cent_nbr = np.arange(y_center[1]-3,y_center[1]+4)

In [864]: for y in y_cent_nbr:
       for x in x_cent_nbr:
        nbr_list.append(im2_fil[y,x])
   .....:         

In [865]: x_cent_nbr = np.arange(x_center[2]-3,x_center[2]+4)

In [866]: y_cent_nbr = np.arange(y_center[2]-3,y_center[2]+4)

In [867]: 

In [867]: for y in y_cent_nbr:
       for x in x_cent_nbr:
        nbr_list.append(im2_fil[y,x])
   .....:         

In [868]: x_cent_nbr = np.arange(x_center[3]-3,x_center[3]+4)

In [869]: y_cent_nbr = np.arange(y_center[3]-3,y_center[3]+4)

In [870]: for y in y_cent_nbr:
       for x in x_cent_nbr:
        nbr_list.append(im2_fil[y,x])
   .....:         

In [871]: x_cent_nbr = np.arange(x_center[4]-3,x_center[4]+4)

In [872]: y_cent_nbr = np.arange(y_center[4]-3,y_center[4]+4)

In [873]: for y in y_cent_nbr:
       for x in x_cent_nbr:
        nbr_list.append(im2_fil[y,x])

        

In [829]: nbr_list = np.array(nbr_list)

In [830]: nbr_avg = mean(nbr_list[:])

In [831]: y_cur_back[:][y_cur_back[:] <= nbr_avg] = 255

In [832]: plt.imshow(y_cur_back, cmap = 'gray')




mask = close_img > close_img.mean()
label_im, nb_labels = ndimage.label(mask)
sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
mean_vals = ndimage.sum(inty_final, label_im, range(1, nb_labels + 1))
mask_size = sizes >= 150
remove_pixel = mask_size[label_im]
label_im[remove_pixel] = True
mask_size = sizes < 150
remove_pixel = mask_size[label_im]
label_im[remove_pixel] = False
##mask_size = sizes <= 75
##remove_pixel = mask_size[label_im]
##label_im[remove_pixel] = True
##mask_size = sizes > 75
##remove_pixel = mask_size[label_im]
##label_im[remove_pixel] = False

mask = inty_final > inty_final.mean()*2
label_inty, nb_labels = ndimage.label(mask)
sizes = ndimage.sum(mask, label_inty, range(nb_labels + 1))
mean_vals = ndimage.sum(inty_final, label_inty, range(1, nb_labels + 1))
mask_size = sizes > 150
remove_pixel = mask_size[label_inty]
label_inty[remove_pixel] = True
mask_size = sizes <75
remove_pixel = mask_size[label_inty]
label_inty[remove_pixel] = True

mask = hue_final > hue_final.mean()*1.5
label_hue, nb_labels = ndimage.label(mask)
sizes = ndimage.sum(mask, label_hue, range(nb_labels + 1))
mean_vals = ndimage.sum(hue_final, label_hue, range(1, nb_labels + 1))
mask_size = sizes > 150
remove_pixel = mask_size[label_hue]
label_hue[remove_pixel] = True
mask_size = sizes <75
remove_pixel = mask_size[label_hue]
label_hue[remove_pixel] = True

mask = sat_final > sat_final.mean()*1.5
label_sat, nb_labels = ndimage.label(mask)
sizes = ndimage.sum(mask, label_sat, range(nb_labels + 1))
mean_vals = ndimage.sum(sat_final, label_sat, range(1, nb_labels + 1))
mask_size = sizes > 150
remove_pixel = mask_size[label_sat]
label_sat[remove_pixel] = True
mask_size = sizes <75
remove_pixel = mask_size[label_sat]
label_sat[remove_pixel] = True

In [1241]: plt.imshow(label_im, cmap = 'gray')

In [1604]: shape(data["Image_data"]["boundingboxes"])
Out[1604]: (5,)

In [1605]: data["Image_data"]["boundingboxes"][0]
Out[1605]: 
{u'corner_bottom_right_x': 519,
 u'corner_bottom_right_y': 27,
 u'corner_top_left_x': 504,
 u'corner_top_left_y': 3,
 u'id': 1,
 u'properties': {u'Difficult': u'False',
  u'Occluded': u'False',
  u'Truncated': u'False'},
 u'species': u'Diamondback Moth'}

In [1606]: data["Image_data"]["boundingboxes"][0]["corner_bottom_right_x"]
Out[1606]: 519

In [1613]: brx
Out[1613]: array([519, 554, 591, 360, 304])



############################################################################################################################################################

## Contrast normalization/Histogram equalization (batch processing)

##cd '/mnt/data/shamir/Annotation data set/Original Images/Bad Images'
##/mnt/data/shamir/Annotation data set/Original Images/Bad Images

img_list = glob.glob('*.jpg')
img_list = sort(np.array(img_list))
for z in range(shape(img_list)[0]):
   im = cv2.imread(img_list[z], cv2.CV_LOAD_IMAGE_COLOR)
   im_red_eq = cv2.equalizeHist(im[:,:,0])
   im_green_eq = cv2.equalizeHist(im[:,:,1])
   im_blue_eq = cv2.equalizeHist(im[:,:,2])
   im_eq = dstack([im_red_eq, im_green_eq, im_blue_eq])
   scipy.misc.imsave('/mnt/data/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/Training Set_histNorm/'+ list[z], im_eq)
##   im = Image.open(list[z])
##   im = io.autocontrast(img, cutoff = 0.1)   
##   im.save('/mnt/data/shamir/Annotation data set/Normalized Images/Labeled/'+ list[x], 'JPEG')

############################################################################################################################################################

## Local Contrast Normalization function

def lcn_2d(im, sigmas=[1.591, 1.591]):
    """ Apply local contrast normalization to a square image.
    Uses a scheme described in Pinto et al (2008)
    Based on matlab code by Koray Kavukcuoglu
    http://cs.nyu.edu/~koray/publis/code/randomc101.tar.gz

    data is 2-d
    sigmas is a 2-d vector of standard devs (to define local smoothing kernel)
    
    Example
    =======
    im_p = lcn_2d(im,[1.591, 1.591])
    """

    #assert(issubclass(im.dtype.type, np.floating))
    im = np.cast[np.float](im)

    # 1. subtract the mean and divide by std dev
    mn = np.mean(im)
    sd = np.std(im, ddof=1)

    im -= mn
    im /= sd



    lmn = gaussian_filter(im, sigmas, mode='reflect')
    lmnsq = gaussian_filter(im ** 2, sigmas, mode='reflect')

    lvar = lmnsq - lmn ** 2;
    np.clip(lvar, 0, np.inf, lvar)  # items < 0 set to 0
    lstd = np.sqrt(lvar)

    np.clip(lstd, 1, np.inf, lstd)

    im -= lmn
    im /= lstd

    return im

## Perform LCN (batch processing)

img_list = glob.glob('*.jpg')
img_list = sort(np.array(img_list))
for z in range(shape(img_list)[0]):
   im = cv2.imread(img_list[z], cv2.CV_LOAD_IMAGE_COLOR)
   im_red_eq = lcn_2d(im[:,:,0],[10, 10])
   im_green_eq = lcn_2d(im[:,:,1],[10, 10])        # 1.591
   im_blue_eq = lcn_2d(im[:,:,2],[10, 10])
   im_eq = dstack([im_red_eq, im_green_eq, im_blue_eq])
   scipy.misc.imsave('/mnt/data/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/Training Set_LCN/'+ img_list[z], im_eq)
    
############################################################################################################################################################

# Training Set

cd '/export/mlrg/salavi/shamir/Annotation data set/Original Images/Good Images/Positive Counts/Training Set'

# Training Set LCN

cd '/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/Training Set_LCN'

# Training Set LCN Labelled

cd '/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/TrainingSet_LCN_Labelled_1.2'

# Training Set LCB BB 1.2

cd '/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Positive Counts/TrainingSet_LCN_BB_1.2'

## Draw bounding boxes based on ground truth (done)

img_list = glob.glob('*.jpg')
img_list = sort(np.array(img_list))
for z in range(shape(img_list)[0]):       # shape(img_list)[0]
   im = cv2.imread(img_list[z], cv2.CV_LOAD_IMAGE_COLOR)
   # decode JSON
   json_data =  open(img_list[z][:-4])           # img_list[z][:-4]
   data = json.load(json_data)
   brx, tlx, bry, tly = [], [], [], []
   for x in range(shape(data["Image_data"]["boundingboxes"][:])[0]):
      brx.append(data["Image_data"]["boundingboxes"][x]["corner_bottom_right_x"])
      tlx.append(data["Image_data"]["boundingboxes"][x]["corner_top_left_x"])
      bry.append(data["Image_data"]["boundingboxes"][x]["corner_bottom_right_y"])
      tly.append(data["Image_data"]["boundingboxes"][x]["corner_top_left_y"])
   brx = np.array(brx)
   bry = np.array(bry)
   tly = np.array(tly)
   tlx = np.array(tlx)
   x,y,x1,y1 = tlx, tly, brx, bry
   # draw BB   
   for i in range(shape(x)[0]):            
      cv2.rectangle(im,(x[i],y[i]),(x1[i],y1[i]),(0,255,0),1)
   scipy.misc.imsave('/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Bounding Boxes_GT/'+ list[z], im)


                                                                    # Region Labeling #


## Open and filter image
                                                                    
im = Image.open('120_5096.jpg')       # img_list[z]
im_fil = scipy.ndimage.filters.median_filter(im, size = (4,3,4))
im_cur = im_fil.copy()

## Separate RGB channels

im_red = im_cur[:,:,0].copy()
im_green = im_cur[:,:,1].copy()
im_blue = im_cur[:,:,2].copy()

## Decode JSON file and store all the corner coordinates in an array

json_data =  open('120_5103')           # list[z][:-4]
data = json.load(json_data)

brx, tlx, bry, tly = [], [], [], []

for x in range(shape(data["Image_data"]["boundingboxes"][:])[0]):
   brx.append(data["Image_data"]["boundingboxes"][x]["corner_bottom_right_x"])
   tlx.append(data["Image_data"]["boundingboxes"][x]["corner_top_left_x"])
   bry.append(data["Image_data"]["boundingboxes"][x]["corner_bottom_right_y"])
   tly.append(data["Image_data"]["boundingboxes"][x]["corner_top_left_y"])     

brx = np.array(brx)
bry = np.array(bry)
tly = np.array(tly)
tlx = np.array(tlx)

x,y,x1,y1 = tlx, tly, brx, bry

## Calculate the center of each bounding box

x_centre = x + abs(x1 - x)/2
y_centre = y + abs(y1 - y)/2

## Calculate 2x2 neighbouring pixels from the center (include center) and store all the pixels in their respectice arrays


nbr_list_red, nbr_list_green, nbr_list_blue = [], [], []

def get_centre(nbr_list, colour_channel):
   no_of_moths = shape(x_centre)[0]
   for i in range(no_of_moths):
      x_cent_nbr = np.arange(x_centre[i] - 2, x_centre[i] + 3)       # tunable parameter
      y_cent_nbr = np.arange(y_centre[i] - 2, y_centre[i] + 3)       # tunable parameter
      for y in y_cent_nbr:
         for x in x_cent_nbr:
            nbr_list.append(colour_channel[y,x])

get_centre(nbr_list_red, im_red)
get_centre(nbr_list_green, im_green)
get_centre(nbr_list_blue, im_blue)         

nbr_list_red = np.array(nbr_list_red)
nbr_list_green = np.array(nbr_list_green)
nbr_list_blue = np.array(nbr_list_blue)

############################################################################################################################################################

## Perform background extraction (US Patent)

red_avg = np.mean(nbr_list_red[:])
green_avg = np.mean(nbr_list_green[:])
blue_avg = np.mean(nbr_list_blue[:])

im_red_back = im_red.copy()
im_green_back = im_green.copy()
im_blue_back = im_blue.copy()

# tunable parameters
def sub_backgnd(tuner):
   
   im_red_back[:][im_red_back[:] <= red_avg*tuner] = 255                
   im_green_back[:][im_green_back[:] <= green_avg*tuner] = 255
   im_blue_back[:][im_blue_back[:] <= blue_avg*tuner] = 255

sub_backgnd(1.2)       # *1.15 or 1.2 - best results with LCN
im_back = dstack([im_red_back, im_green_back, im_blue_back])

############################################################################################################################################################

## Convert image (current and background) from RGB to YCbCr and create HSI model

def rgb2ycbcr(image):
   y = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
   cb = 128 - 0.169*image[:,:,0] - 0.331*image[:,:,1] + 0.5*image[:,:,2]
   cr = 128 + 0.5*image[:,:,0] - 0.419*image[:,:,1] - 0.081*image[:,:,2]
   return y, cb, cr

im_y, im_cb, im_cr = rgb2ycbcr(im_cur)
im_y_back, im_cb_back, im_cr_back = rgb2ycbcr(im_back)


def ycbcr2hsi(ch1, ch2, ch3):                            # ch = channel
   inty = ch1
   hue = np.arctan(np.divide(ch3, ch2))
   sat = np.sqrt(np.square(ch3) + np.square(ch2))
   return inty, hue, sat

im_int, im_hue, im_sat = ycbcr2hsi(im_y, im_cb, im_cr)
im_int_back, im_hue_back, im_sat_back = ycbcr2hsi(im_y_back, im_cb_back, im_cr_back)

# Create image differences

im_int_diff = abs(im_int - im_int_back) # gives you an inverted image :(
im_hue_diff = abs(im_hue - im_hue_back)
im_sat_diff = abs(im_sat - im_sat_back)

## Histogram plotting (no need to consider neighbouring pixels)

pixels_int, pixels_hue, pixels_sat = [], [], []

def create_hist(pixels, diff_img):
   for y in range(shape(diff_img)[0]):
      for x in range(shape(diff_img)[1]):
         pixels.append(diff_img[y,x])
   hist, bins = np.histogram(pixels, bins = 256)
   width = 0.7*(bins[1] - bins[0])                 # just for plotting the histogram (comment it out if unnecessary)
   centre = (bins[:-1] + bins[1:])/2               # just for plotting the histogram (comment it out if unnecessary)
   return hist, bins, width, centre                # omit width and centre if unnecessary

# omit the corresponding width and centre variables from below if these are commented out above
hist_int, bins_int, width_int, centre_int = create_hist(pixels_int, im_int_diff)
hist_hue, bins_hue, width_hue, centre_hue = create_hist(pixels_hue, im_hue_diff)
hist_sat, bins_sat, width_sat, centre_sat = create_hist(pixels_sat, im_sat_diff)

##plt.figure()
##plt.subplot(131)
##plt.bar(centre_int, hist_int, align = 'center', width = width_int)
##plt.subplot(132)
##plt.bar(centre_hue, hist_hue, align = 'center', width = width_hue)
##plt.subplot(133)
##plt.bar(centre_sat, hist_sat, align = 'center', width = width_sat)


## ADAPTIVE THRESHOLDING (see patent for algorithm)

# Set threshold to the default value bin

def adaptive_thresh(hist, bins):
   
   # Find the total number of pixels in the image   
   N, search_thresh = 0, 0   
   for q in range(shape(hist)[0]):
      N += hist[q]

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

N1, search_thresh_int = adaptive_thresh(hist_int, bins_int)
N2, search_thresh_hue = adaptive_thresh(hist_hue, bins_hue)
N3, search_thresh_sat = adaptive_thresh(hist_sat, bins_sat)


## The last words

inty, hue, sat = im_int_diff.copy(), im_hue_diff.copy(), im_sat_diff.copy()

inty[:,:][inty[:,:] <= search_thresh_int] = False #True
inty[:,:][inty[:,:] > search_thresh_int] = True #False
hue[:,:][hue[:,:] <= search_thresh_hue] = False #True
hue[:,:][hue[:,:] > search_thresh_hue] = True #False
sat[:,:][sat[:,:] <= search_thresh_sat] = False #True
sat[:,:][sat[:,:] > search_thresh_sat] = True #False

im_combinedOR = np.logical_or(inty, hue, sat)

open_img = ndimage.binary_opening(im_combinedOR, structure = np.ones((5,5))).astype(np.int) # works best
close_img = ndimage.binary_closing(open_img)

mask = close_img > close_img.mean()
label_im, nb_labels = ndimage.label(mask)
sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
mean_vals = ndimage.sum(close_img, label_im, range(1, nb_labels + 1))
mask_size = sizes >= 300
remove_pixel = mask_size[label_im]
label_im[remove_pixel] = False
mask_size = sizes < 50
remove_pixel = mask_size[label_im]
label_im[remove_pixel] = False

im_final_morph_label, num_components = ndimage.label(label_im) 

##scipy.misc.imsave('/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/TrainingSet_LCN_Labelled/'+ '120_5103(test).jpg', im_final_morph_label)

scipy.misc.imsave('/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/hmm/'+ '120_5103(test).jpg', im_final_morph_label)



                                                                           # End #

############################################################################################################################################################
#### Linear Normalization
##
##im_int_diff_norm = im_int_diff.copy()
##im_hue_diff_norm = im_hue_diff.copy()
##im_sat_diff_norm = im_sat_diff.copy()
##
##newMin, newMax = 0, 255
##
##im_int_diff_norm_max = np.amax(im_int_diff_norm)
##im_hue_diff_norm_max = np.amax(im_hue_diff_norm)
##im_sat_diff_norm_max = np.amax(im_sat_diff_norm)
##
##im_int_diff_norm_min = np.amin(im_int_diff_norm)
##im_hue_diff_norm_min = np.amin(im_hue_diff_norm)
##im_sat_diff_norm_min = np.amin(im_sat_diff_norm)
##
##
for y in range(shape(im_int_diff)[0]):
   for x in range(shape(im_int_diff)[1]):
      im_int_diff_norm[y,x] = (((im_int_diff[y,x] - im_int_diff_norm_min)*(newMax - newMin))/(im_int_diff_norm_max - im_int_diff_norm_min)) + newMin
##
##for y in range(shape(im_hue_diff)[0]):
##   for x in range(shape(im_hue_diff)[1]):
##      im_hue_diff_norm[y,x] = (((im_hue_diff[y,x] - im_hue_diff_norm_min)*(newMax - newMin))/(im_hue_diff_norm_max - im_hue_diff_norm_min)) + newMin
##
##for y in range(shape(im_sat_diff)[0]):
##   for x in range(shape(im_sat_diff)[1]):
##      im_sat_diff_norm[y,x] = (((im_sat_diff[y,x] - im_sat_diff_norm_min)*(newMax - newMin))/(im_sat_diff_norm_max - im_sat_diff_norm_min)) + newMin

############################################################################################################################################################

### Colour Thresholding (already tested, didn't work well)

### Perform adaptive search

## Calculate mean

red_avg = mean(nbr_list_red[:]) # edit previous code as per necessity
green_avg = mean(nbr_list_green[:])
blue_avg = mean(nbr_list_blue[:])

## Perform colour thresholding based on the acquired threshold values

d = np.empty(shape = (shape(im_fil[:,:,0])[0], shape(im_fil[:,:,0])[1]))

for y in range(shape(im_fil[:,:,0])[0]):
   for x in range(shape(im_fil[:,:,0])[1]):
      d[y,x] = np.sqrt(np.square(im_fil[:,:,0][y,x] - red_avg) + np.square(im_fil[:,:,1][y,x] - green_avg) + np.square(im_fil[:,:,2][y,x] - blue_avg))

############################################################################################################################################################

## Adaptive search (make a function)

j, N, u_ob, u_bg, qh_ob, qh_bg, h_ob, h_bg = 0,0,0,0,0,0,0,0

## int

for q in range(shape(hist_int)[0]):
   N += hist_int[q]

T = np.zeros((shape(hist_int)[0],), dtype=np.int)

for q in range(1,shape(hist_int)[0]+1):
   T[0] += q*hist_int[q-1]
T[0] = int(T[0]/N)

while T[j] != T[j-1]:
   j += 1
   for q in range(int(T[j-1]) + 1):
      qh_ob += q*hist_int[q]
      h_ob += hist_int[q]
   u_ob = qh_ob/h_ob   
   for q in range(int(T[j-1] + 1), 256):
      qh_bg += q*hist_int[q]
      h_bg += hist_int[q]
   u_bg = qh_bg/h_bg    
   T[j] = (u_ob + u_bg)/2
   qh_ob, h_ob, qh_bg, h_bg = 0,0,0,0
   if j > 255:
      break

thres_int = T[:][T[:] > 0][-1]


##hue

for q in range(shape(hist_hue)[0]):
   N += hist_hue[q]

T = np.zeros((shape(hist_hue)[0],), dtype=np.int)

for q in range(1,shape(hist_hue)[0]+1):
   T[0] += q*hist_hue[q-1]
T[0] = int(T[0]/N)

while T[j] != T[j-1]:
   j += 1
   for q in range(int(T[j-1]) + 1):
      qh_ob += q*hist_hue[q]
      h_ob += hist_hue[q]
   u_ob = qh_ob/h_ob   
   for q in range(int(T[j-1] + 1), 256):
      qh_bg += q*hist_hue[q]
      h_bg += hist_hue[q]
   u_bg = qh_bg/h_bg    
   T[j] = (u_ob + u_bg)/2
   qh_ob, h_ob, qh_bg, h_bg = 0,0,0,0
   if j > 255:
      break

thres_hue = T[:][T[:] > 0][-1]

## sat

for q in range(shape(hist_sat)[0]):
   N += hist_sat[q]

T = np.zeros((shape(hist_sat)[0],), dtype=np.int)

for q in range(1,shape(hist_sat)[0]+1):
   T[0] += q*hist_sat[q-1]
T[0] = int(T[0]/N)

while T[j] != T[j-1]:
   j += 1
   for q in range(int(T[j-1]) + 1):
      qh_ob += q*hist_sat[q]
      h_ob += hist_sat[q]
   u_ob = qh_ob/h_ob   
   for q in range(int(T[j-1] + 1), 256):
      qh_bg += q*hist_sat[q]
      h_bg += hist_sat[q]
   u_bg = qh_bg/h_bg    
   T[j] = (u_ob + u_bg)/2
   qh_ob, h_ob, qh_bg, h_bg = 0,0,0,0
   if j > 255:
      break

thres_sat = T[:][T[:] > 0][-1]

##In [1843]: T[:][T[:] > 0]
##Out[1843]: array([68, 71, 73, 75, 77, 78, 79, 80, 80])
##
##In [1844]: np.amax(T)
##Out[1844]: 80
##
##Tmax = np.amax(T)

inty, hue, sat = im_int_diff.copy(), im_hue_diff.copy(), im_sat_diff.copy()

inty[:,:][inty[:,:] <= thres_int*0.75] = True
inty[:,:][inty[:,:] > thres_int*0.75] = False
hue[:,:][hue[:,:] <= thres_hue*0.75] = True
hue[:,:][hue[:,:] > thres_hue*0.75] = False
sat[:,:][sat[:,:] <= thres_sat*0.75] = True
sat[:,:][sat[:,:] > thres_sat*0.75] = False

im_combinedOR = np.logical_and(inty, hue, sat)



open_img = ndimage.binary_opening(im_combinedOR, structure = np.ones((5,5))).astype(np.int)


close_img = ndimage.binary_closing(open_img)

mask = close_img > close_img.mean()
label_im, nb_labels = ndimage.label(mask)
sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
mean_vals = ndimage.sum(close_img, label_im, range(1, nb_labels + 1))
mask_size = sizes >= 150
remove_pixel = mask_size[label_im]
label_im[remove_pixel] = True
mask_size = sizes < 150
remove_pixel = mask_size[label_im]
label_im[remove_pixel] = False

im_final_morph = close_img - label_im

im_final_morph_label, num_components = ndimage.label(im_final_morph)

In [382]: num_components
Out[382]: 4


##inty_final = ndimage.binary_opening(inty)
##inty_final = ndimage.binary_closing(inty_final)
##hue_final = ndimage.binary_opening(hue)
##hue_final = ndimage.binary_closing(hue_final)
##sat_final = ndimage.binary_opening(sat)
##sat_final = ndimage.binary_closing(sat_final)



##d_norm[:,:][d_norm[:,:] > Tmax] = False

plt.imshow(inty, cmap = 'gray')

##im_final = dstack([inty,hue,sat])

## Grayscale and Gaussian Filtering

im_g = gaussian_filter(im, 3)
im_norm = (im_g - im_g.min()) / (float(im_g.max()) - im_g.min())
im_norm[im_norm < 0.5] = 0
im_norm[im_norm >= 0.5] = 1

result = 255 - (im_norm * 255).astype(numpy.uint8)


############################################################################################################################################################

                                                                  # Post Processing #

## Draw bounding boxes

img_list = glob.glob('*.jpg')
img_list = sort(np.array(img_list))
for z in range(shape(img_list)[0]):
   im = cv2.imread(img_list[z], cv2.CV_LOAD_IMAGE_COLOR)
   im_can = cv2.Canny(im, 100, 200)
   cnt, hierarchy = cv2.findContours(im_can,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
   im_trainSet = cv2.imread('/export/mlrg/salavi/shamir/Annotation data set/Original Images/Good Images/Positive Counts/Training Set/' + img_list[z])
   for i in range(shape(cnt)[0]):
      x,y,w,h = cv2.boundingRect(cnt[i])
      x = x+2
      y = y+2
      w = w+2
      h = h+2
      cv2.rectangle(im_trainSet,(int(x),int(y)),(x+w,y+h),(0,255,0),1)
   scipy.misc.imsave('/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Positive Counts/TrainingSet_LCN_BB_1.2/'+ img_list[z], im_trainSet)



# for a single image

cd '/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/hmm'

egg = '120_5259(test).jpg'
im = cv2.imread(egg, cv2.CV_LOAD_IMAGE_COLOR)
im_can = cv2.Canny(im, 100, 200)
cnt, hierarchy = cv2.findContours(im_can,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
im_trainSet = cv2.imread('/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/Training Set/' + '120_5110.jpg')
for i in range(shape(cnt)[0]):
   x,y,w,h = cv2.boundingRect(cnt[i])      # the coordinates
   cv2.rectangle(im_trainSet,(int(x-2),int(y-2)),(x+w,y+h),(0,255,0),1)
scipy.misc.imsave('/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/TrainingSet_LCN_BB_1.15/'+ egg, im_trainSet)


# histogram equalization OpenCV

im_red_eq = cv2.equalizeHist(im[:,:,0])
im_green_eq = cv2.equalizeHist(im[:,:,1])
im_blue_eq = cv2.equalizeHist(im[:,:,2])

im_eq = dstack([im_red_eq, im_green_eq, im_blue_eq])

############################################################################################################################################################

#### get the pixels of all bounding boxes (ground truth) WrongApproach WrongApproach WrongApproach WrongApproach WrongApproach WrongApproach WrongApproach
##
### decode JSON file first to get x, x1, y, y1
##
##bb_gt_pixels = []
##for i in range(shape(x)[0]):
##    bb_gt_pixels.append([])
##    for m in range(x[i], x1[i]+1):
##        for n in range(y[i], y1[i]+1):
##            bb_gt_pixels[i].append(m+n)
##
##bb_gt_pixels = np.array(bb_gt_pixels)


#### get the pixels of all bounding boxes(detected) PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM PROBLEM
##
##bb_dt_pixels = []
##im = cv2.imread('120_5096.jpg', cv2.CV_LOAD_IMAGE_COLOR)
##im_can = cv2.Canny(im, 100, 200)
##cnt, hierarchy = cv2.findContours(im_can,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
##for i in range(shape(cnt)[0]):
##   bb_dt_pixels.append([])
##   x,y,w,h = cv2.boundingRect(cnt[i])
##   for m in range(x, x+w+1):
##      for n in range(y, y+h+1):
##         bb_dt_pixels[i].append(m+n)
##
##bb_dt_pixels = np.array(bb_dt_pixels)

# Alternative approach (C/C++ like for loop implementation)

bb_dt_pixels = []
im = cv2.imread('120_5103.jpg', cv2.CV_LOAD_IMAGE_COLOR)
im_can = cv2.Canny(im, 100, 200)
cnt, hierarchy = cv2.findContours(im_can,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
i, j = 0, 0
while i < shape(cnt)[0]:
##   bb_dt_pixels.append([])
   x,y,w,h = cv2.boundingRect(cnt[i])
   print x, y, w, h
##   for m in range(x, x+w+1):
##      for n in range(y, y+h+1):
##         bb_dt_pixels[j].append(m+n)
   i += 2
   j += 1

bb_dt_pixels = np.array(bb_dt_pixels)



# Get corner coordinates of the detected bouning boxes (single image)

im = cv2.imread('120_5096.jpg', cv2.CV_LOAD_IMAGE_COLOR)
im_can = cv2.Canny(im, 100, 200)
cnt, hierarchy = cv2.findContours(im_can,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
i = 0
a, b, a1, b1 = [], [], [], []

while i < shape(cnt)[0]:
   m,n,w,h = cv2.boundingRect(cnt[i])
   m = m+2
   n = n+2
   w = w+2
   h = h+2
   a.append(m)
   b.append(n)
   a1.append(m+w)
   b1.append(n+h)
   i += 2

a = np.array(a)
b = np.array(b)
a1 = np.array(a1)
b1 = np.array(b1)

# Get all the coordinates of both ground truth (gt) and  detected (dt) boxes

x_val, y_val, a_val, b_val = [], [], [], []

# gt
bb_gt_pixels = []
for k in range(shape(x)[0]):
   bb_gt_pixels.append([])
   for i in range(x[k], x1[k]+1):
      for j in range(y[k], y1[k]+1):
         x_val.append(i)
         y_val.append(j)
   x_val = np.array(x_val)
   y_val = np.array(y_val)
   bb_gt_pixels[k].append(dstack([x_val, y_val]))
   x_val, y_val = [], []

#dt
bb_dt_pixels = []
for k in range(shape(a)[0]):
   bb_dt_pixels.append([])
   for i in range(a[k], a1[k]+1):
      for j in range(b[k], b1[k]+1):
         a_val.append(i)
         b_val.append(j)
   a_val = np.array(a_val)
   b_val = np.array(b_val)
   bb_dt_pixels[k].append(dstack([a_val, b_val]))
   a_val, b_val = [], []

##for i in range(a, a1+1):
##   for j in range(b, b1+1):
##      a_val.append(i)
##      b_val.append(j)
##a_val = np.array(a_val)
##b_val = np.array(b_val)
##bb_dt_pixels = dstack([a_val, b_val])



## The ultimate evaluation algo (as fast as a slug)

comn_pix_array = []
common_pixels, overlap, match, false_pos, false_neg = 0, 0, 0, 0, 0

for i in range(shape(bb_gt_pixels)[0]):
   for j in range(shape(bb_dt_pixels)[0]):
      for m in range(shape(bb_gt_pixels[i][0][0])[0]):
         for n in range(shape(bb_dt_pixels[j][0][0])[0]):
            if (bb_gt_pixels[i][0][0][m] == bb_dt_pixels[j][0][0][n]).all() == True:
               common_pixels += 1      # intersection
      comn_pix_array.append(common_pixels)
##      print common_pixels
      common_pixels = 0
      if comn_pix_array[j] > 0:
         overlap += 1
##   print overlap
##   print comn_pix_array
   comn_pix_array = np.array(comn_pix_array)
   if overlap == 0:
      false_neg += 1
   elif overlap == 1:      
      find_index = np.where(comn_pix_array > 0)
##      print find_index
      index = find_index[0][0]
      all_pixels = shape(bb_gt_pixels[i][0][0])[0] + shape(bb_dt_pixels[index][0][0])[0] - comn_pix_array[index]  # union
      match_value = comn_pix_array[index] / float(all_pixels)
##      print comn_pix_array[index], all_pixels, match_value
##      print match_value
      if match_value > 0.15:
         match += 1
      else:
         false_neg += 1
   elif overlap > 1:
      find_index = np.where(comn_pix_array == np.amax(comn_pix_array))
      if size(find_index[0]) > 1:
         for k in range(size(find_index[0])):
            index = find_index[0][k]
            all_pixels = shape(bb_gt_pixels[i][0][0])[0] + shape(bb_dt_pixels[index][0][0])[0] - comn_pix_array[index]
            match_value = comn_pix_array[index] / float(all_pixels)
##            print match_value
            if match_value > 0.15:
               match += 1
               break
            elif k < size(find_index[0]):
               continue
            else:
               false_neg += 1
      else:
         index = find_index[0][0]
         all_pixels = shape(bb_gt_pixels[i][0][0])[0] + shape(bb_dt_pixels[index][0][0])[0] - comn_pix_array[index]
         match_value = comn_pix_array[index] / float(all_pixels)
##         print match_value
         if match_value > 0.15:
            match += 1
         else:
            false_neg += 1
   overlap = 0
   comn_pix_array = []

false_pos = shape(bb_dt_pixels)[0] - match
print 'match = ', match
print 'false_pos = ', false_pos
print 'false_neg = ', false_neg


# Find the number of common pixels (NOT EFFICIENT) INEFFICIENT INEFFICIENT INEFFICIENT INEFFICIENT INEFFICIENT INEFFICIENT INEFFICIENT INEFFICIENT

for i in range(shape(xycoordinates)[1]):
    for j in range(shape(abcoordinates)[1]):
        if (xycoordinates[0][i] == abcoordinates[0][j]).all() == True: 
            common_pixels += 1

# The first coordinate of intersection (Top Left)

for i in range(shape(xycoordinates)[1]):
   for j in range(shape(abcoordinates)[1]):
      if (xycoordinates[0][i] == abcoordinates[0][j]).all() == True:
         x = xycoordinates[0][i][0]
         y = xycoordinates[0][i][1]
   break

# The last coordinate of intersection (Bottom Right)

for i in range(shape(xycoordinates)[1]):
   for j in range(shape(abcoordinates)[1]):
      if (xycoordinates[0][i] == abcoordinates[0][j]).all() == True:
         x1 = xycoordinates[0][i][0]
         y1 = xycoordinates[0][i][1]
      break

############################################################################################################################################################

## Make an array of the 64 grids

initx, inity = 0, 0
x_cut, y_cut = 80, 60
x_val, y_val, grid = [], [], []
for k in range(64):
   if (k > 0) & ((k % 8) == 0):
      initx += 80
      x_cut += 80
      inity, y_cut = 0, 60
   grid.append([])
   for i in range(initx, x_cut):
      for j in range(inity, y_cut):
         x_val.append(i)
         y_val.append(j)
   x_val = np.array(x_val)
   y_val = np.array(y_val)
   grid[k].append(dstack([x_val, y_val]))
   inity += 60
   y_cut += 60   
   x_val, y_val = [], []
grid = np.array(grid)

# Make a dictionary of the grids
grid_ref = defaultdict(list)
for i in range(shape(grid)[0]):
   grid_ref[i].append(grid[i])
   
# Assign numbers to the corresponding Bounding Boxes

#gt
bb_gt_ref = defaultdict(list)
for i in range(shape(bb_gt_pixels)[0]):
   for j in range(shape(grid)[0]):
      if (bb_gt_pixels[i][0][0][0][0] <= grid_ref.items()[j][1][0][0,0,4799,0]) & (bb_gt_pixels[i][0][0][0][1] <= grid_ref.items()[j][1][0][0,0,4799,1]):
         bb_gt_ref[j].append(bb_gt_pixels[i][0][0])
         break

#dt
bb_dt_ref = defaultdict(list)
for i in range(shape(bb_dt_pixels)[0]):
   for j in range(shape(grid)[0]):
      if (bb_dt_pixels[i][0][0][0][0] <= grid_ref.items()[j][1][0][0,0,4799,0]) & (bb_dt_pixels[i][0][0][0][1] <= grid_ref.items()[j][1][0][0,0,4799,1]):
         bb_dt_ref[j].append(bb_dt_pixels[i][0][0])
         break
         
# Get corner coordinates of the detected bouning boxes (single image)

cd '/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Positive Counts/TrainingSet_LCN_Labelled_1.2'

im = cv2.imread('120_5174.jpg', cv2.CV_LOAD_IMAGE_COLOR)   # img_list[z]
im_can = cv2.Canny(im, 100, 200)
cnt, hierarchy = cv2.findContours(im_can,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
i = 0
a, b, a1, b1 = [], [], [], []
while i < shape(cnt)[0]:
   m,n,w,h = cv2.boundingRect(cnt[i])
   m = m+2
   n = n+2
   w = w+2
   h = h+2
   
   a.append(m)
   b.append(n)
   a1.append(m+w)
   b1.append(n+h)
   i += 2

a = np.array(a)
b = np.array(b)
a1 = np.array(a1)
b1 = np.array(b1)

## Extract insects

cd '/export/mlrg/salavi/shamir/Annotation data set/Original Images/Good Images/Positive Counts/Training Set'

insects = []
no_of_insects = 0
im = Image.open('120_5174.jpg') # 132_4554 120_5123
for i in range(len(a)):
##   insects.append([])
   cropped = im.crop((int(a[i]),int(b[i]),int(a1[i]),int(b1[i]))) # 179,198,201,226     347,132,361,155
   cropped = cropped.resize((32,32), Image.ANTIALIAS)   
   cropped = np.asarray(cropped) # weirdly auto-flipped
##   cropped = cv2.copyMakeBorder(cropped,1,1,1,1, cv2.BORDER_CONSTANT,value=(0,0,0))
   insects.append(cropped)
##   no_of_insects += 1
insects = np.array(insects)
##cropped = cv2.flip(cropped, 'flipCode' == 0)

##scipy.misc.imsave('/export/mlrg/salavi/shamir/Annotation data set/Insects/'+ img_list[z], im_trainSet)

### Subplot

fig = plt.figure(1)
grid = ImageGrid(fig, 111, # similar to subplot(111)
                nrows_ncols = (16, 16), # creates 2x2 grid of axes
                axes_pad=0.0, # pad between axes in inch.
                share_all=True,
                )

for i in range(len(a)):
    grid[i].imshow(insects[i]) # The AxesGrid object work as a list of axes.
grid.axes_llc.set_xticks([])
grid.axes_llc.set_yticks([])
plt.show()

####from mpl_toolkits.axes_grid1 import AxesGrid
##
##plt.figure()
##grid = AxesGrid(fig, 111, # similar to subplot(132)
##               nrows_ncols = (2, 5),
##               axes_pad = 0.0,
##               share_all=True,
####               label_mode = "L",               
##               )
##for i in range(10):
##   grid[i].imshow(insects[i])
##
##plt.show()
##   
##
##
##    
##icount = 0
### 2x5 subplot with
##f, axarr = plt.subplots(2, 5, sharex = True, sharey = True)
##for i in range(len(axarr)):
##   for j in range(shape(axarr)[1]):
##      icount += 1
##      axarr[i, j].axis('off')  #frame1 = plt.gca()
##      axarr[i,j].imshow(insects[icount-1])
##f.subplots_adjust(wspace=0.001, hspace=0.001)
##plt.setp([a.get_yticklabels() for a in f.axes[:]], visible=False)
##plt.setp([a.get_xticklabels() for a in f.axes[:]], visible=False)
##
##f.subplots_adjust(hspace=0, wspace=0)  # fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
##
##y = plt.imshow(insects[0])
##
##
##for i in range(2):
##   for j in range(5):
##      matshow()


# save image

insect_ID = str(i)
add_str = '_' + insect_ID + '.jpg'
filename = img_list[z][:-4] + add_str
scipy.misc.imsave('/mnt/ssd/shamir/insects_gt/' + filename, im_final_morph_label)



In [22]: avg_int = np.array(avg_int)

In [23]: avg_int
Out[23]: 
array([ 126.95586262,  105.05888647,  125.28578564,  137.48480675,
        128.06671355,  110.39882905,  134.22956113,  122.77356211,
        129.49809238,  116.30076237,  132.69532789,  113.64671633,
        112.69315892,  129.19593846,  133.44977966,  135.87590986,
        122.43601582,  137.10284221,  129.32205003,  147.10825433,
        137.47640875,  123.39432082,  121.25202964,  129.35172422,
        133.44198671,  133.27954077,  115.52231025,  126.41931389,
        111.2334617 ,  106.74701415,  153.06057357,  149.25207376,
        153.08837896,  128.55171335,  140.27861804,  133.11132054,
        139.02260133,  134.092728  ,  132.60906369,  128.8254184 ,
        131.84974623,  125.96135586,  115.81252418,  120.76735219,
        120.90539204,  128.01190048,  123.9100009 ,  133.36626507,
        133.86554285,  120.05346397,  118.86321388,  131.21783288,
        119.76916297,  114.28808873,  114.11643273,  115.1400234 ,
        143.56647923,  138.6718361 ,  119.81268462,  124.27642263,
        116.16163489,   92.31889534,  116.96065224,  116.47661694,
        111.15554179,  114.07548482,  136.72731195,  153.24112908,
        121.54665878,  126.49657082,  137.1457058 ,   98.72897262,
         99.94047563,  118.22683276,  125.90244805,  110.90161374,
        114.66655972,  162.59663143,  117.17485143,  120.99351337,
        112.41919427,  147.20026304,  165.56754456,  160.45156732,
        103.18696127,  105.94564172,  107.9759003 ,   98.86939521,
         99.11583633,   96.60010146,   73.93702804,   88.25619385,
        105.08929791,  130.59794632,  149.87291902,  125.9230628 ,
        125.65615146,  110.41632778,  111.0966967 ,  148.34749959,
        158.43674171,   98.46435953,  149.1879707 ,  118.45922802,
        109.4111418 ,  107.90687854,  103.35837346,   98.67713365,
        113.90752694,  118.21992806,  111.06344097,  109.1770675 ,
        102.74762089,  101.56754999,  110.62367668,  108.1537073 ,
        108.54466447,  116.93511639,  112.18239675,  107.42843801,
        102.62441074,   96.23865382,  108.0395281 ,  110.6649789 ,
        102.01554755,   94.96891634,  111.72285864,  103.90156768,
        129.7911734 ,  105.40064182])

In [24]: np.min(avg_int)
Out[24]: 73.937028037093057

In [25]: np.max(avg_int)
Out[25]: 165.56754455731141

In [26]: min_int = np.min(avg_int)

In [27]: max_int = np.max(avg_int)

In [28]: int_interval = (max_int - min_int) / 3

In [30]: first_qrtr = min_int + int_interval

In [31]: second_qrtr = first_qrtr + int_interval

In [32]: third_qrtr = second_qrtr + int_interval
