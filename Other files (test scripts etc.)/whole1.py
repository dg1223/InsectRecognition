import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import scipy
from scipy import ndimage
import ImageOps as io
import _imaging
import cv2
import json
import os
import sys
import glob
from pprint import pprint
from scipy.ndimage import gaussian_filter

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

list = glob.glob('*.jpg')
list = sort(np.array(list))
for z in range(shape(list)[0]):
   im = cv2.imread(list[z], cv2.CV_LOAD_IMAGE_COLOR)
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

list = glob.glob('*.jpg')
list = sort(np.array(list))
for z in range(shape(list)[0]):
   im = cv2.imread(list[z], cv2.CV_LOAD_IMAGE_COLOR)
   im_red_eq = lcn_2d(im[:,:,0],[10, 10])
   im_green_eq = lcn_2d(im[:,:,1],[10, 10])        # 1.591
   im_blue_eq = lcn_2d(im[:,:,2],[10, 10])
   im_eq = dstack([im_red_eq, im_green_eq, im_blue_eq])
   scipy.misc.imsave('/mnt/data/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/Training Set_LCN/'+ list[z], im_eq)
    
############################################################################################################################################################

# Training Set

cd '/mnt/data/shamir/Annotation data set/Original Images/Good Images/Positive Counts/Training Set'

# Training Set LCN

cd '/mnt/data/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/Training Set_LCN'

# Training Set LCN Labelled

cd '/mnt/data/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/TrainingSet_LCN_Labelled_1.2'


## Draw bounding boxes based on ground truth (done)

list = glob.glob('*.jpg')
list = sort(np.array(list))
for z in range(shape(list)[0]):
   im = cv2.imread(list[z], cv2.CV_LOAD_IMAGE_COLOR)
   # decode JSON
   json_data =  open(list[z][:-4])           # list[z][:-4]
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
   scipy.misc.imsave('/mnt/data/shamir/Annotation data set/Normalized Images/Bounding Boxes_GT/'+ list[z], im)


                                                                    # Region Labeling #


## Open and filter image
                                                                    
im = Image.open(list[z])       # list[z]
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


## Calculate the center of each bounding box

x_center = tlx + abs(brx - tlx)/2
y_center = tly + abs(bry - tly)/2

## Calculate 2x2 neighbouring pixels from the center (include center) and store all the pixels in their respectice arrays

no_of_moths = shape(x_center)[0]
nbr_list_red, nbr_list_green, nbr_list_blue = [], [], []

for i in range(no_of_moths):
   x_cent_nbr = np.arange(x_center[i] - 2, x_center[i] + 3)
   y_cent_nbr = np.arange(y_center[i] - 2, y_center[i] + 3)
   for y in y_cent_nbr:
      for x in x_cent_nbr:
         nbr_list_red.append(im_red[y,x])           

for i in range(no_of_moths):
   x_cent_nbr = np.arange(x_center[i] - 2, x_center[i] + 3)
   y_cent_nbr = np.arange(y_center[i] - 2, y_center[i] + 3)
   for y in y_cent_nbr:
      for x in x_cent_nbr:
         nbr_list_green.append(im_green[y,x])
         
for i in range(no_of_moths):
   x_cent_nbr = np.arange(x_center[i] - 2, x_center[i] + 3)
   y_cent_nbr = np.arange(y_center[i] - 2, y_center[i] + 3)
   for y in y_cent_nbr:
      for x in x_cent_nbr:
         nbr_list_blue.append(im_blue[y,x])          

nbr_list_red = np.array(nbr_list_red)
nbr_list_green = np.array(nbr_list_green)
nbr_list_blue = np.array(nbr_list_blue)

############################################################################################################################################################

## Perform background extraction (US Patent)

red_avg = mean(nbr_list_red[:])
green_avg = mean(nbr_list_green[:])
blue_avg = mean(nbr_list_blue[:])

im_red_back = im_red.copy()
im_green_back = im_green.copy()
im_blue_back = im_blue.copy()

im_red_back[:][im_red_back[:] <= red_avg*1.15] = 255                # *1.4 - best result with histogram normalization
im_green_back[:][im_green_back[:] <= green_avg*1.15] = 255
im_blue_back[:][im_blue_back[:] <= blue_avg*1.15] = 255

im_back = dstack([im_red_back, im_green_back, im_blue_back])

############################################################################################################################################################

## Convert image (current and background) from RGB to YCbCr and create HSI model

##def rgb2ycc(image): # Y u no work?
im_y = 0.299*im_cur[:,:,0] + 0.587*im_cur[:,:,1] + 0.114*im_cur[:,:,2]
im_cb = 128 - 0.169*im_cur[:,:,0] - 0.331*im_cur[:,:,1] + 0.5*im_cur[:,:,2]
im_cr = 128 + 0.5*im_cur[:,:,0] - 0.419*im_cur[:,:,1] - 0.081*im_cur[:,:,2]

im_y_back = 0.299*im_back[:,:,0] + 0.587*im_back[:,:,1] + 0.114*im_back[:,:,2]
im_cb_back = 128 - 0.169*im_back[:,:,0] - 0.331*im_back[:,:,1] + 0.5*im_back[:,:,2]
im_cr_back = 128 + 0.5*im_back[:,:,0] - 0.419*im_back[:,:,1] - 0.081*im_back[:,:,2]

im_int = im_y
im_hue = np.arctan(np.divide(im_cr,im_cb))
im_sat = np.sqrt(np.square(im_cr) + np.square(im_cb))

im_int_back = im_y_back
im_hue_back = np.arctan(np.divide(im_cr_back,im_cb_back))
im_sat_back = np.sqrt(np.square(im_cr_back) + np.square(im_cb_back))

# Create image differences

im_int_diff = abs(im_int - im_int_back) # gives you an inverted image :(
im_hue_diff = abs(im_hue - im_hue_back)
im_sat_diff = abs(im_sat - im_sat_back)

## Histogram plotting (no need to consider neighbouring pixels)

pixels_int, pixels_hue, pixels_sat = [], [], []

for y in range(shape(im_int_diff)[0]):
   for x in range(shape(im_int_diff)[1]):
      pixels_int.append(im_int_diff[y,x])

for y in range(shape(im_hue_diff)[0]):
   for x in range(shape(im_hue_diff)[1]):
      pixels_hue.append(im_hue_diff[y,x])

for y in range(shape(im_sat_diff)[0]):
   for x in range(shape(im_sat_diff)[1]):
      pixels_sat.append(im_sat_diff[y,x])

hist_int, bins_int = np.histogram(pixels_int, bins = 256)
width_int = 0.7*(bins_int[1] - bins_int[0])
center_int = (bins_int[:-1] + bins_int[1:])/2

hist_hue, bins_hue = np.histogram(pixels_hue, bins = 256)
width_hue = 0.7*(bins_hue[1] - bins_hue[0])
center_hue = (bins_hue[:-1] + bins_hue[1:])/2

hist_sat, bins_sat = np.histogram(pixels_sat, bins = 256)
width_sat = 0.7*(bins_sat[1] - bins_sat[0])
center_sat = (bins_sat[:-1] + bins_sat[1:])/2

##plt.figure()
##plt.subplot(131)
##plt.bar(center_int, hist_int, align = 'center', width = width_int)
##plt.subplot(132)
##plt.bar(center_hue, hist_hue, align = 'center', width = width_hue)
##plt.subplot(133)
##plt.bar(center_sat, hist_sat, align = 'center', width = width_sat)


## ADAPTIVE THRESHOLDING

N = 0

# Find the total number of pixels in the image

for q in range(shape(hist_int)[0]):
   N += hist_int[q]                 # only one channel will suffice

##int

# Set threshold to the default value bin

num_pixels = 0
req_pixels = N*0.15

for i in range(shape(hist_int)[0]-1,-1,-1):
   if num_pixels < req_pixels:
      num_pixels += hist_int[i]
      if num_pixels > req_pixels:
         num_pixels -= hist_int[i]
         i += 1
         break

def_thresh = bins_int[i]

# Find peak-value bin within the 1st 30

peak_bin = np.array(np.where(np.amax(hist_int[0:30])))[0][0]

# Std deviation about the peak bin (needs further clarification)

std_dev = np.std(hist_int)

# Find the minimum bin size (0.15% of the total pixels)

min_bin = N*0.0015

# Calculate search threshold

thresh = hist_int[0] + std_dev*0.5
num_pixels2 = 0

for j in range(1, shape(hist_int)[0]):
   if num_pixels2 < thresh:
      num_pixels2 += hist_int[j-1]
      if num_pixels2 >= thresh:
         num_pixels2 -= hist_int[j-1]
         j -= 1
         break

dest_bin = 0

for bin in range(j):
   if hist_int[bin+1] < min_bin:
      dest_bin = 0
##      print bin+1, hist_int[bin + 1]
   else:
      dest_bin = bin + 1
      search_thresh_int = bins_int[dest_bin]
      break


## hue
   
# Set threshold to the default value bin

num_pixels = 0
req_pixels = N*0.15

for i in range(shape(hist_hue)[0]-1,-1,-1):
   if num_pixels < req_pixels:
      num_pixels += hist_hue[i]
      if num_pixels > req_pixels:
         num_pixels -= hist_hue[i]
         i += 1
         break

def_thresh = bins_hue[i]

# Find peak-value bin within the 1st 30

peak_bin = np.array(np.where(np.amax(hist_hue[0:30])))[0][0]

# Std deviation about the peak bin (needs further clarification)

std_dev = np.std(hist_hue)

# Find the minimum bin size (0.15% of the total pixels)

min_bin = N*0.0015

# Calculate search threshold

thresh = hist_hue[0] + std_dev*0.5
num_pixels2 = 0

for j in range(1, shape(hist_hue)[0]):
   if num_pixels2 < thresh:
      num_pixels2 += hist_hue[j-1]
      if num_pixels2 >= thresh:
         num_pixels2 -= hist_hue[j-1]
         j -= 1
         break

dest_bin = 0

for bin in range(j):
   if hist_hue[bin+1] < min_bin:
      dest_bin = 0
##      print bin+1, hist_hue[bin + 1]
   else:
      dest_bin = bin + 1
      search_thresh_hue = bins_hue[dest_bin]
      break

## sat
   
# Set threshold to the default value bin

num_pixels = 0
req_pixels = N*0.15

for i in range(shape(hist_sat)[0]-1,-1,-1):
   if num_pixels < req_pixels:
      num_pixels += hist_sat[i]
      if num_pixels > req_pixels:
         num_pixels -= hist_sat[i]
         i += 1
         break

def_thresh = bins_sat[i]

# Find peak-value bin within the 1st 30

peak_bin = np.array(np.where(np.amax(hist_sat[0:30])))[0][0]

# Std deviation about the peak bin (needs further clarification)

std_dev = np.std(hist_sat)

# Find the minimum bin size (0.15% of the total pixels)

min_bin = N*0.0015

# Calculate search threshold

thresh = hist_sat[0] + std_dev*0.5
num_pixels2 = 0

for j in range(1, shape(hist_sat)[0]):
   if num_pixels2 < thresh:
      num_pixels2 += hist_sat[j-1]
      if num_pixels2 >= thresh:
         num_pixels2 -= hist_sat[j-1]
         j -= 1
         break

dest_bin = 0

for bin in range(j):
   if hist_sat[bin+1] < min_bin:
      dest_bin = 0
      print bin+1, hist_sat[bin + 1]
   else:
      dest_bin = bin + 1
      search_thresh_sat = bins_sat[dest_bin]
      break


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

scipy.misc.imsave('/mnt/data/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/TrainingSet_LCN_Labelled/'+ list[z], im_final_morph_label)

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
##for y in range(shape(im_int_diff)[0]):
##   for x in range(shape(im_int_diff)[1]):
##      im_int_diff_norm[y,x] = (((im_int_diff[y,x] - im_int_diff_norm_min)*(newMax - newMin))/(im_int_diff_norm_max - im_int_diff_norm_min)) + newMin
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

## Edge detection

list = glob.glob('*.jpg')
list = sort(np.array(list))
for z in range(shape(list)[0]):
   im = cv2.imread(list[z], cv2.CV_LOAD_IMAGE_COLOR)
   im_can = cv2.Canny(im, 100, 200)
   cnt, hierarchy = cv2.findContours(im_can,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
   im_trainSet = cv2.imread('/mnt/data/shamir/Annotation data set/Original Images/Good Images/Positive Counts/Training Set/' + list[z])
   for i in range(shape(cnt)[0]):
      x,y,w,h = cv2.boundingRect(cnt[i])      
      cv2.rectangle(im_trainSet,(int(x-2),int(y-2)),(x+w,y+h),(0,255,0),1)
   scipy.misc.imsave('/mnt/data/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/TrainingSet_LCN_BB/'+ list[z], im_trainSet)



# for a single image

im = cv2.imread('120_5096.jpg', cv2.CV_LOAD_IMAGE_COLOR)
im_can = cv2.Canny(im, 100, 200)
cnt, hierarchy = cv2.findContours(im_can,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
im_trainSet = cv2.imread('/mnt/data/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/Training Set/' + '120_5110.jpg')
for i in range(shape(cnt)[0]):
   x,y,w,h = cv2.boundingRect(cnt[i])      # the coordinates
   cv2.rectangle(im_trainSet,(int(x-2),int(y-2)),(x+w,y+h),(0,255,0),1)
scipy.misc.imsave('/mnt/data/shamir/Annotation data set/Normalized Images/Bounding Boxes_Labelled Images/'+ '120_5110_newThresh.jpg', im_trainSet)


# histogram equalization OpenCV

im_red_eq = cv2.equalizeHist(im[:,:,0])
im_green_eq = cv2.equalizeHist(im[:,:,1])
im_blue_eq = cv2.equalizeHist(im[:,:,2])

im_eq = dstack([im_red_eq, im_green_eq, im_blue_eq])

############################################################################################################################################################

## get the pixels of all bounding boxes (ground truth)

# decode JSON file first to get x, x1, y, y1

bb_gt_pixels = []
for i in range(shape(x)[0]):
    bb_gt_pixels.append([])
    for m in range(x[i], x1[i]+1):
        for n in range(y[i], y1[i]+1):
            bb_gt_pixels[i].append(m+n)

bb_gt_pixels = np.array(bb_gt_pixels)


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

im = cv2.imread('120_5103.jpg', cv2.CV_LOAD_IMAGE_COLOR)
im_can = cv2.Canny(im, 100, 200)
cnt, hierarchy = cv2.findContours(im_can,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
i = 0
a, b, a1, b1 = [], [], [], []

while i < shape(cnt)[0]:
   m,n,w,h = cv2.boundingRect(cnt[i])
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



## The ultimate evaluation algo

comn_pix_array = []
common_pixels, overlap, match, false_pos, false_neg = 0, 0, 0, 0, 0

for i in range(shape(bb_gt_pixels)[0]):
   for j in range(shape(bb_dt_pixels)[0]):
      for m in range(shape(bb_gt_pixels[i][0][0])[0]):
         for n in range(shape(bb_dt_pixels[j][0][0])[0]):
            if (bb_gt_pixels[i][0][0][m] == bb_dt_pixels[j][0][0][n]).all() == True:
               common_pixels += 1      # intersection
      comn_pix_array.append(common_pixels)
      common_pixels = 0
      if comn_pix_array[j] > 0:
         overlap += 1      
   if overlap == 0:
      false_neg += 1
   elif overlap == 1:      
      find_index = np.where(comn_pix_array > 0)
      index = find_index[0][0]
      all_pixels = shape(bb_gt_pixels[i][0][0])[0] + shape(bb_dt_pixels[index][0][0])[0] - comn_pix_array[index]  # union
      match_value = comn_pix_array[index] / float(all_pixels)
##      print match_value
      if match_value > 0.5:
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
            if match_value > 0.5:
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
         if match_value > 0.5:
            match += 1
         else:
            false_neg += 1
   overlap = 0
   comn_pix_array = []

false_pos = shape(bb_dt_pixels)[0] - match


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

# Compute the intersecting area
# area = number of pixels inside the bounded region


