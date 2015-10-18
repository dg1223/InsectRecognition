start_time = time.time()

## Local Contrast Normalization

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

#### Make an array of the 64 grids for a 640x480 image (i.e. divide the image into 64 grids)
##
##initx, inity = 0, 0
##x_cut, y_cut = 80, 60
##x_val, y_val, grid = [], [], []
##for k in range(64):
##   if (k > 0) & ((k % 8) == 0):
##      initx += 80
##      x_cut += 80
##      inity, y_cut = 0, 60
##   grid.append([])
##   for i in range(initx, x_cut):
##      for j in range(inity, y_cut):
##         x_val.append(i)
##         y_val.append(j)
##   x_val = np.array(x_val)
##   y_val = np.array(y_val)
##   grid[k].append(dstack([x_val, y_val]))
##   inity += 60
##   y_cut += 60   
##   x_val, y_val = [], []
##grid = np.array(grid)
##
### Make a dictionary of the grids (assigning a serial number to each grid)
##grid_ref = defaultdict(list)
##for i in range(shape(grid)[0]):
##   grid_ref[i].append(grid[i])


# Training Set (nc)
cd '/export/mlrg/salavi/shamir/Annotation data set/Original Images/Good Images/No Counts/Training_Set'

### Test Set (+ve)
##cd '/export/mlrg/salavi/shamir/Annotation data set/Original Images/Good Images/Positive Counts/Test Set'

### Test Set (nc)
##cd '/mnt/ssd/shamir/Original Images/Good Images/No Counts/Test_Set'

## Open and filter image

img_list2 = glob.glob('*.jpg')                  # creates a list of all the files with the given format
img_list2 = sort(np.array(img_list2))
nbr_list_red, nbr_list_green, nbr_list_blue = [], [], []                # list of neighbouring coordinates
first_qrtr,  second_qrtr = 112, 145          # data obtained from experiment on contrast normalized images
feature_database = []
for z in range(shape(img_list2)[0]):        # shape(img_list2)[0]
##    print img_list2[z]
    ### detect corners

##    # 1. Image PreProcessing ( closing operation )
##
##    img = cv2.imread(img_list2[z], cv2.CV_LOAD_IMAGE_COLOR) # 120_5096
##    img = cv2.GaussianBlur(img,(5,5),0)
##    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
##    mask = np.zeros((gray.shape),np.uint8)
##    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)) # ellipsiodal kernel (should test using other kernels to see the difference)
##
##    close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1) # advanced morphological transformation (closing operation - removing small black areas)
##    div = np.float32(gray)/(close) # cleans up the image (no idea why!)
##    res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX)) # linear normalization
##    res2 = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR) # gray to RGB
##
##    # 2. Finding Squares and Creating Mask Image
##
##    thresh = cv2.adaptiveThreshold(res,255,0,1,19,2)
##    contour,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
##
##    max_area = 0
##    best_cnt = None
##    for cnt in contour:
##        area = cv2.contourArea(cnt)
##        if area > 1000: # need to test other values for project images
##            if area > max_area:
##                max_area = area
##                best_cnt = cnt
##
##    cv2.drawContours(mask,[best_cnt],0,255,-1)
##    cv2.drawContours(mask,[best_cnt],0,0,2)
##
##    res = cv2.bitwise_and(res,mask)
##
##    # 3. Finding Vertical lines
##
##    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10)) # a 2 by 10 rectangle
##
##    dx = cv2.Sobel(res,cv2.CV_16S,1,0)
##    dx = cv2.convertScaleAbs(dx)
##    cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
##    ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
##    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)
##
##    contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
##    for cnt in contour:
##        x,y,w,h = cv2.boundingRect(cnt)
##        if h/w > 5:
##            cv2.drawContours(close,[cnt],0,255,-1)
##        else:
##            cv2.drawContours(close,[cnt],0,0,-1)
##    close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
##    closex = close.copy()
##
##    # 4. Finding Horizontal Lines
##
##    kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
##    dy = cv2.Sobel(res,cv2.CV_16S,0,2)
##    dy = cv2.convertScaleAbs(dy)
##    cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
##    ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
##    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely)
##
##    contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
##    for cnt in contour:
##        x,y,w,h = cv2.boundingRect(cnt)
##        if w/h > 5:
##            cv2.drawContours(close,[cnt],0,255,-1)
##        else:
##            cv2.drawContours(close,[cnt],0,0,-1)
##
##    close = cv2.morphologyEx(close,cv2.MORPH_DILATE,None,iterations = 2)
##    closey = close.copy()
##
##    # 5. Finding Grid Points
##
##    res = cv2.bitwise_and(closex,closey)
##
##    ############################################################################################################################################################
##
##    ## get rid of the grids
##
##    res = cv2.bitwise_or(closex, closey)
##    x = np.where(res > 0)
##    x = np.array(x)
##
##    for i in range(shape(x)[1]):
##        gray[x[0,i], x[1,i]] = 255
##
##    a = np.where(gray == 255)
##    x_min = np.min(a[1])
##    y_min = np.min(a[0])
##    x_max = np.max(a[1])
##    y_max = np.max(a[0])
##
##    im_org = Image.open(img_list2[z])
##    cropped = im_org.crop((x_min, y_min, x_max, y_max))
##    cropped = np.asarray(cropped)
##    im = cropped.copy()
    
##    ## Decode JSON file and store all the corner coordinates of ground truth inside an array
##    
##    json_data =  open(img_list2[z][:-4])           # img_list[z][:-4]
##    data = json.load(json_data)
##    brx, tlx, bry, tly = [], [], [], []
##
##    for x in range(shape(data["Image_data"]["boundingboxes"][:])[0]):
##       brx.append(data["Image_data"]["boundingboxes"][x]["corner_bottom_right_x"])
##       tlx.append(data["Image_data"]["boundingboxes"][x]["corner_top_left_x"])
##       bry.append(data["Image_data"]["boundingboxes"][x]["corner_bottom_right_y"])
##       tly.append(data["Image_data"]["boundingboxes"][x]["corner_top_left_y"])     
##    brx = np.array(brx)
##    bry = np.array(bry)
##    tly = np.array(tly)
##    tlx = np.array(tlx)
##    x,y,x1,y1 = tlx+3, tly+3, brx, bry
##
##    # The Annotation Tool enables the user to draw bouning boxes beyond the image boundary which gives unexpected coordinates. To rectify this bug, the
##    # following function reduces the corresponding incorrect BB coordinates to a specific value (x = 639, y = 479) within the image boundaries.
##
##    def rectify(array, incor_val):
##       if (array >= incor_val).any() == True:
##          find_unwanted_val = np.where(array >= incor_val)
##          for i in find_unwanted_val[0]:
##             if incor_val == 640:
##                array[i] = 639
##             elif incor_val == 480:
##                array[i] = 479
##             else:
##                return None
##    rectify(x, 640)
##    rectify(x1, 640)
##    rectify(y, 480)
##    rectify(y1, 480)

    ## Process Image

    im = cv2.imread(img_list2[z], cv2.CV_LOAD_IMAGE_COLOR) # comment this while using cropped images
    im_red_eq = lcn_2d(im[:,:,0],[10, 10])
    im_green_eq = lcn_2d(im[:,:,1],[10, 10])        # 1.591
    im_blue_eq = lcn_2d(im[:,:,2],[10, 10])

    def med_filter(channel):               # median filter
       filtered_channel = scipy.ndimage.filters.median_filter(channel, size = (4,4))
       return filtered_channel

    im_red = med_filter(im_red_eq)
    im_green = med_filter(im_green_eq)
    im_blue = med_filter(im_blue_eq)

    cv2.normalize(im_red, im_red, 0,255,cv2.NORM_MINMAX)
    cv2.normalize(im_green, im_green, 0,255,cv2.NORM_MINMAX)
    cv2.normalize(im_blue, im_blue, 0,255,cv2.NORM_MINMAX)

##    ## Linear normalization
##    def normalize(image, newMax, newMin):
##       img_min = np.min(image)
##       img_max = np.max(image)
##       for y in range(shape(image)[0]):
##          for x in range(shape(image)[1]):
##             image[y,x] = (((image[y,x] - img_min)*(newMax - newMin))/(img_max - img_min)) + newMin
##       return image
##
##    im_red = normalize(im_red, 255, 0)
##    im_green = normalize(im_green, 255, 0)
##    im_blue = normalize(im_blue, 255, 0)

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

##            c = 0
##            a, b, a1, b1 = [], [], [], []
##
##            while c < shape(cnt)[0]:
##               m,n,w,h = cv2.boundingRect(cnt[c])                                             # bounding rectangle
##               # stretching the BB to reduce the size discrepancy between BBgt and BBdt 
##               m = m+2
##               n = n+2
##               w = w+2
##               h = h+2
##               a.append(m)
##               b.append(n)
##               a1.append(m+w)
##               b1.append(n+h)
##               c += 1                
##            a = np.array(a)
##            b = np.array(b)
##            a1 = np.array(a1)
##            b1 = np.array(b1)
##
##            ## Get all the coordinates of both ground truth (gt) and  detected (dt) boxes
##    
##            bb_gt_pixels, bb_dt_pixels = [], []                       # bb or BB = bounding box
##             
##            def get_coordinates(c1, c11, c2, c21, bb_pixels):         # c = corner
##               x_val, y_val = [], []
##               for k in range(shape(c1)[0]):
##                  bb_pixels.append([])
##                  for i in range(c1[k], c11[k]+1):
##                     for j in range(c2[k], c21[k]+1):
##                        x_val.append(i)
##                        y_val.append(j)
##                  x_val = np.array(x_val)
##                  y_val = np.array(y_val)
##                  bb_pixels[k].append(dstack([x_val, y_val]))
##                  x_val, y_val = [], []
##
##            # BBgt
##            get_coordinates(x, x1, y, y1, bb_gt_pixels)
##            bb_gt_pixels = np.array(bb_gt_pixels)
##            # BBdt
##            get_coordinates(a, a1, b, b1, bb_dt_pixels)
##            bb_dt_pixels = np.array(bb_dt_pixels)
##            
##            ## Assign numbers to the corresponding Bounding Boxes
##
##            # Make a dictionary for each type of BB which will contain the relevant pixel arrays and their corresponding grid numbers
##            bb_gt_ref = defaultdict(list)
##            bb_dt_ref = defaultdict(list)
##
##            def assign_numbers(bb_pixels, bb_ref):
##               for i in range(shape(bb_pixels)[0]):
##                  for j in range(shape(grid)[0]):
##                     if (bb_pixels[i,0,0][0,0] <= grid_ref.items()[j][1][0][0,0,4799,0]) & (bb_pixels[i,0,0][0,1] <= grid_ref.items()[j][1][0][0,0,4799,1]):
##                        bb_ref[j].append(bb_pixels[i][0][0])
##                        break
##            assign_numbers(bb_gt_pixels, bb_gt_ref)       # for BBgt
##            assign_numbers(bb_dt_pixels, bb_dt_ref)       # for BBdt


            ### Feature Extraction
            
            mask = np.zeros(im_int.shape, np.uint8)
            cv2.drawContours(mask,[cntr],0,255,-1)
            mean = cv2.mean(im, mask = mask)
            find = np.where(mask > 0)
            x_axis = find[1][:]
            y_axis = find[0][:]            
 
            ## Average intensity
                 
            intensity = []
            def avg_int(ipixel, image, x_coord, y_coord):
               for i in range(len(x_coord)):
                   ipixel.append(image[y_coord[i],x_coord[i]])
               mean = round(np.mean(np.array(ipixel)))
               return mean

            avg_intensity = avg_int(intensity, im_int, x_axis, y_axis)


            ## Intensity histogram

            pixels = []
            def create_hist(hpixel, image, x_coord, y_coord):
               for i in range(len(x_coord)):
                   hpixel.append(image[y_coord[i],x_coord[i]])
               hist, bins = np.histogram(hpixel, bins = 64, range = (0.0, 255.0))
               width = 0.7*(bins[1] - bins[0])                 # just for plotting the histogram (comment it out if unnecessary)
               centre = (bins[:-1] + bins[1:])/2               # just for plotting the histogram (comment it out if unnecessary)
               return hist, bins, width, centre 
 
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

