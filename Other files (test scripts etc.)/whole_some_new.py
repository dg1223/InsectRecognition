#### Local Contrast Normalization
##
##def lcn_2d(im, sigmas=[1.591, 1.591]):
##    """ Apply local contrast normalization to a square image.
##    Uses a scheme described in Pinto et al (2008)
##    Based on matlab code by Koray Kavukcuoglu
##    http://cs.nyu.edu/~koray/publis/code/randomc101.tar.gz
##
##    data is 2-d
##    sigmas is a 2-d vector of standard devs (to define local smoothing kernel)
##    
##    Example
##    =======
##    im_p = lcn_2d(im,[1.591, 1.591])
##    """
##
##    #assert(issubclass(im.dtype.type, np.floating))
##    im = np.cast[np.float](im)
##
##    # 1. subtract the mean and divide by std dev
##    mn = np.mean(im)
##    sd = np.std(im, ddof=1)
##
##    im -= mn
##    im /= sd
##
##
##
##    lmn = gaussian_filter(im, sigmas, mode='reflect')
##    lmnsq = gaussian_filter(im ** 2, sigmas, mode='reflect')
##
##    lvar = lmnsq - lmn ** 2;
##    np.clip(lvar, 0, np.inf, lvar)  # items < 0 set to 0
##    lstd = np.sqrt(lvar)
##
##    np.clip(lstd, 1, np.inf, lstd)
##
##    im -= lmn
##    im /= lstd
##
##    return im

# Training Set
cd '/mnt/ssd/shamir/Original Images/Good Images/Positive Counts/Training Set'

### Test Set
##cd '/mnt/ssd/shamir/Original Images/Good Images/No Counts/Test_Set'

## Perform LCN (batch processing)


img_list1 = glob.glob('*.jpg')
img_list1 = sort(np.array(img_list1))
##nbr_list_red, nbr_list_green, nbr_list_blue = [], [], []                # list of neighbouring coordinates
##avg_int  = []
first_qrtr,  second_qrtr = 112, 145           # data obtained from experiment
for z in range(shape(img_list1)[0]):                     # shape(img_list1)[0]
   im = cv2.imread(img_list1[z], cv2.CV_LOAD_IMAGE_COLOR)
   im_red_eq = lcn_2d(im[:,:,0],[10, 10])
   im_green_eq = lcn_2d(im[:,:,1],[10, 10])        # 1.591
   im_blue_eq = lcn_2d(im[:,:,2],[10, 10])

   def med_filter(channel):               # median filter
      filtered_channel = scipy.ndimage.filters.median_filter(channel, size = (4,4))
      return filtered_channel

   im_red = med_filter(im_red_eq)
   im_green = med_filter(im_green_eq)
   im_blue = med_filter(im_blue_eq)

   ## Linear normalization
   def normalize(image, newMax, newMin):
      img_min = np.min(image)
      img_max = np.max(image)
      for y in range(shape(image)[0]):
         for x in range(shape(image)[1]):
            image[y,x] = (((image[y,x] - img_min)*(newMax - newMin))/(img_max - img_min)) + newMin
      return image

##   def normalize(arr):
##       """
##       Linear normalization
##       http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
##       """
##       arr = arr.astype('float')
##       # Do not touch the alpha channel
##       minval = arr[...,i].min()
##       maxval = arr[...,i].max()
##       if minval != maxval:
##           arr[...,i] -= minval
##           arr[...,i] *= (255.0/(maxval-minval))
##       return arr 
   
##   im_red = cv2.normalize(im_red, im_red, 0, 255)
##   im_green = cv2.normalize(im_green, im_green, 0, 255)
##   im_blue = cv2.normalize(im_blue, im_blue, 0, 255)  

   im_red = normalize(im_red, 255, 0)
   im_green = normalize(im_green, 255, 0)
   im_blue = normalize(im_blue, 255, 0)

##   im_red = normalize(im_red)
##   im_green = normalize(im_green)
##   im_blue = normalize(im_blue) 

   ## Make copies for background segmentation
   im_red_back = im_red.copy()
   im_green_back = im_green.copy()
   im_blue_back = im_blue.copy()

   ## Convert image (current and background) from RGB to YCbCr and create HSI model

   # RGB to YCbCr
   def rgb2ycbcr(red, green, blue):   # source - wiki
      y = 0.299*red + 0.587*green + 0.114*blue
      cb = 128 - 0.169*red - 0.331*green + 0.5*blue
      cr = 128 + 0.5*red - 0.419*green - 0.081*blue
      return y, cb, cr

   im_y, im_cb, im_cr = rgb2ycbcr(im_red, im_green, im_blue)   
   avg_inty = np.mean(im_y)
##   avg_int.append(avg_inty)

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
   
   sub_backgnd(1.3)       # *1.15 ~ 1.3 - best results with LCN (the lower the avg values are, the higher the tuner value should be - experimented)
##   im_back = dstack([im_red_back, im_green_back, im_blue_back])   # RGB background
   im_y_back, im_cb_back, im_cr_back = rgb2ycbcr(im_red_back, im_green_back, im_blue_back)
##   
##
      ## Decode JSON file and store all coordinates in an array
         
      json_data =  open(img_list1[z][:-4])                                 # img_list2[z][:-4]
      data = json.load(json_data)

      brx, tlx, bry, tly = [], [], [], []                                  # br = bottom right, tl = top left

      # Get coordinates
      for x in range(shape(data["Image_data"]["boundingboxes"][:])[0]):
         brx.append(data["Image_data"]["boundingboxes"][x]["corner_bottom_right_x"])
         tlx.append(data["Image_data"]["boundingboxes"][x]["corner_top_left_x"])
         bry.append(data["Image_data"]["boundingboxes"][x]["corner_bottom_right_y"])
         tly.append(data["Image_data"]["boundingboxes"][x]["corner_top_left_y"])
      brx = np.array(brx)
      bry = np.array(bry)
      tly = np.array(tly)
      tlx = np.array(tlx)

      # The Annotation Tool enables the user to draw bouning boxes beyond the image boundary which gives unexpected coordinates. To rectify this bug, the
      # following function reduces the corresponding incorrect BB coordinates to a specific value (x = 639, y = 479) within the image boundaries.

      def rectify(array, incor_val):
          if (array >= incor_val).any() == True:
              find_unwanted_val = np.where(array >= incor_val)
              for i in find_unwanted_val[0]:
                 if incor_val == 640:
                    array[i] = 639
                 elif incor_val == 480:
                    array[i] = 479
                 else:
                    return None
      rectify(brx, 640)
      rectify(tlx, 640)
      rectify(bry, 480)
      rectify(tly, 480)


      # Calculate the center of each bounding box
      x_centre = tlx + abs(brx - tlx)/2
      y_centre = tly + abs(bry - tly)/2

      # Calculate 2x2 neighbouring pixels from the centre (including center) and store all the pixels in their respectice arrays (tunable parameter)   
         

      def get_centre(nbr_list, colour_channel):
         no_of_moths = shape(x_centre)[0]
         for i in range(no_of_moths):
            x_cent_nbr = np.arange(x_centre[i] - 2, x_centre[i] + 3)
            y_cent_nbr = np.arange(y_centre[i] - 2, y_centre[i] + 3)
            for y in y_cent_nbr:
               for x in x_cent_nbr:
                  nbr_list.append(colour_channel[y,x])                  # store the pixel values of the corresponding coordinates
      get_centre(nbr_list_red, im_red)
      get_centre(nbr_list_green, im_green)
      get_centre(nbr_list_blue, im_blue)        
##
##      ##   nbr_list_red = np.array(nbr_list_red)
##      ##   nbr_list_green = np.array(nbr_list_green)
##      ##   nbr_list_blue = np.array(nbr_list_blue)
##
##      ############################################################################################################################################################
##
##         ## Perform background extraction
##
##         # This is basically a colour thresholding technique. The 2x2 neighbouring pixels values for each of the R,G and B channels from each BBgt centre has
##         # already been stored. Now, we are calculating the mean pixel values for each channel to perform a colour based background subtraction.
##
##         # Calculate average
##      red_avg = np.mean(nbr_list_red[:])
##      green_avg = np.mean(nbr_list_green[:])
##      blue_avg = np.mean(nbr_list_blue[:])
##      print red_avg, green_avg, blue_avg

############################################################################################################################################################

##   ## Convert image (current and background) from RGB to YCbCr and create HSI model
##
##   # RGB to YCbCr
##   def rgb2ycbcr(image):   # source - wiki
##      y = 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]
##      cb = 128 - 0.169*image[:,:,0] - 0.331*image[:,:,1] + 0.5*image[:,:,2]
##      cr = 128 + 0.5*image[:,:,0] - 0.419*image[:,:,1] - 0.081*image[:,:,2]
##      return y, cb, cr
##
##   im_y, im_cb, im_cr = rgb2ycbcr(im_cur)
##   avg_int.append(np.mean(im_y))
##   avg_int = np.array(avg_int)
##   print img_list2[z], '  red_avg = ', red_avg, '  avg_int = ', np.mean(im_y), '     difference = ', np.mean(im_y) - red_avg
##   print img_list2[z], '  avg_int = ', np.mean(im_y)

   # YCbCr to HSI
   def ycbcr2hsi(ch1, ch2, ch3):                            # ch = channel, source = US Patent
      inty = ch1
      hue = np.arctan(np.divide(ch3, ch2))
      sat = np.sqrt(np.square(ch3) + np.square(ch2))
      return inty, hue, sat

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

   # Remove object which are too big or too small to be our desired object; changable parameter (mask_size)
   mask = close_img > close_img.mean()                                     # create a mask based on pixel average
   label_im, nb_labels = ndimage.label(mask)                               # label mask
   sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))               # calculate size of all labeled regions of masked image
   mean_vals = ndimage.sum(close_img, label_im, range(1, nb_labels + 1))   # no idea what it does, code worked, got lazy, sorry about that

   # remove objects larger than 300 pixels
   mask_size = sizes >= 300                                                
   remove_pixel = mask_size[label_im]                                      
   label_im[remove_pixel] = False                                          

   # remove objects smaller than 50 pixels
   mask_size = sizes < 50 
   remove_pixel = mask_size[label_im]
   label_im[remove_pixel] = False

   im_final_morph_label, num_components = ndimage.label(label_im)          # label final image

   # save image
   scipy.misc.imsave('/mnt/ssd/shamir/Normalized Images/Good Images/Positive Counts/TrainingSet_LCN_new/'+ img_list1[z], im_final_morph_label)
