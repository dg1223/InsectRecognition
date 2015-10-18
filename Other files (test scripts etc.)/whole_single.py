def rlabel(image, jdata):
    
##    # Test Set LCN
##    cd '/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/Test_Set_LCN'

##    # Training Set LCN
##    cd '/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Positive Counts/Training Set_LCN'
    
    ## Open and filter image
                                                                    
    im = Image.open(image)       # img_list[z]
    im_fil = scipy.ndimage.filters.median_filter(im, size = (4,3,4))
    im_cur = im_fil.copy()

    ## Separate RGB channels

    im_red = im_cur[:,:,0].copy()
    im_green = im_cur[:,:,1].copy()
    im_blue = im_cur[:,:,2].copy()

    ## Decode JSON file and store all the corner coordinates in an array

    json_data =  open(jdata)           # list[z][:-4]
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


    nbr_list_red, nbr_list_green, nbr_list_blue = [], [], []

    def get_center(nbr_list, colour_channel):
       no_of_moths = shape(x_center)[0]
       for i in range(no_of_moths):
          x_cent_nbr = np.arange(x_center[i] - 2, x_center[i] + 3)
          y_cent_nbr = np.arange(y_center[i] - 2, y_center[i] + 3)
          for y in y_cent_nbr:
             for x in x_cent_nbr:
                nbr_list.append(colour_channel[y,x])

    get_center(nbr_list_red, im_red)
    get_center(nbr_list_green, im_green)
    get_center(nbr_list_blue, im_blue)         

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
    plt.imshow(im_final_morph_label, cmap = 'gray')

##    ##scipy.misc.imsave('/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/TrainingSet_LCN_Labelled/'+ '120_5103(test).jpg', im_final_morph_label)
##
##    scipy.misc.imsave('/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/hmm/'+ '120_5103(test).jpg', im_final_morph_label)



                                                                           # End #
