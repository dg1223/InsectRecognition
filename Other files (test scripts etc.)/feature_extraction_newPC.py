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

# Training Set
cd '/export/mlrg/salavi/shamir/Annotation data set/Original Images/Good Images/Positive Counts/Training Set'

### Test Set
##cd '/mnt/ssd/shamir/Original Images/Good Images/Positive Counts/Test Set'

img_list = glob.glob('*.jpg')                       # creates a list of all the files with the given format
img_list = np.sort(np.array(img_list))
first_qrtr,  second_qrtr = 112, 145
feature_database = []
for z in range(shape(img_list)[0]):                 # shape(img_list)[0], alternatively, len(...)    
##    cd '/export/mlrg/salavi/shamir/Annotation data set/Original Images/Good Images/Positive Counts/Training Set'
##    print z, img_list[z]

    
    ### Get coordinates and extract BBgt
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
    x,y,x1,y1 = tlx, tly, brx, bry  # m,n,m+w,n+h

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
    rectify(x,  640)
    rectify(x1, 640)
    rectify(y,  480)
    rectify(y1, 480)


    ## Extract BBgt

##    cd '/export/mlrg/salavi/shamir/Annotation data set/Original Images/Good Images/Positive Counts/Training Set'

    insects_red, insects_green, insects_blue = [], [], []
##    im_org = Image.open(img_list[z])                            # original image
    im_org = cv2.imread(img_list[z], cv2.CV_LOAD_IMAGE_COLOR)
    
    im_red_eq = lcn_2d(im_org[:,:,0],[10, 10])
    im_green_eq = lcn_2d(im_org[:,:,1],[10, 10])        # 1.591
    im_blue_eq = lcn_2d(im_org[:,:,2],[10, 10])

    def med_filter(channel):                                # median filter
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
##        img_min = np.min(image)
##        img_max = np.max(image)
##        for y in range(shape(image)[0]):
##           for x in range(shape(image)[1]):
##              image[y,x] = (((image[y,x] - img_min)*(newMax - newMin))/(img_max - img_min)) + newMin
##        return image
##
##    im_red = normalize(im_red, 255, 0)
##    im_green = normalize(im_green, 255, 0)
##    im_blue = normalize(im_blue, 255, 0)
    
    def save_insct(array, channel, insects):
        for i in range(len(array)):
            cropped = channel[y[i]:y1[i], x[i]:x1[i]]
            insects.append(cropped)
        insects = np.array(insects)
        return insects        

    save_insct(x, im_red, insects_red)    
    save_insct(x, im_green, insects_green)    
    save_insct(x, im_blue, insects_blue)

            
##    for i in range(2,3):             # len(x)
####       cropped = im_org.crop((int(x[i]),int(y[i]),int(x1[i]),int(y1[i])))         # PIL
##       cropped = im_org[y[i]:y1[i], x[i]:x1[i]]                                     # OpenCV
####       cropped = np.asarray(cropped)                            # weirdly auto-flipped
##       insects.append(cropped)
##    insects = np.array(insects)

    for ins in range(len(insects_red)):   # len(insects_red)
##        print ins
        im_red = insects_red[ins].copy()                        # PROBLEM: ALL THE IMAGES DO NOT CONFORM TO THIS CODE (POSSIBLE BUG - TRY ALL LCN IMAGES TO FIND OUT)
        im_green = insects_green[ins].copy()
        im_blue = insects_blue[ins].copy()

        ## Calculate the centre of each BB
        x_im = shape(im_red)[1]
        y_im = shape(im_red)[0]
        x_centre = x_im/2.0
        y_centre = y_im/2.0

        ## Calculate n by n neighbouring pixels from the centre (include centre) and store all the pixels in their respectice arrays

        nbr_list_red, nbr_list_green, nbr_list_blue = [], [], []

        def get_centre(nbr_list, centre_pointx, centre_pointy, start, end,  colour_channel):
            x_cent_nbr = np.arange(centre_pointx - start, centre_pointx + end)      
            y_cent_nbr = np.arange(centre_pointy - start, centre_pointy + end)
            for j in y_cent_nbr:
                for i in x_cent_nbr:
                    nbr_list.append(colour_channel[j,i])

        get_centre(nbr_list_red, x_centre, y_centre, 1, 2, im_red)          # optimum value (tunable)
        get_centre(nbr_list_green, x_centre, y_centre, 1, 2, im_green)
        get_centre(nbr_list_blue, x_centre, y_centre, 1, 2, im_blue)

##        nbr_list_red = np.array(nbr_list_red)
##        nbr_list_green = np.array(nbr_list_green)
##        nbr_list_blue = np.array(nbr_list_blue)

        ############################################################################################################################################################

        ## Perform background extraction (US Patent)

        red_avg = np.mean(nbr_list_red[:])
        green_avg = np.mean(nbr_list_green[:])
        blue_avg = np.mean(nbr_list_blue[:])

        im_red_back = im_red.copy()
        im_green_back = im_green.copy()
        im_blue_back = im_blue.copy()

        # tunable parameters
        def sub_backgnd(red, green, blue, tuner):       
           red[:][red[:] <= red_avg*tuner] = 255                
           green[:][green[:] <= green_avg*tuner] = 255
           blue[:][blue[:] <= blue_avg*tuner] = 255

        sub_backgnd(im_red_back, im_green_back, im_blue_back,  1.2)       # *1.15 or 1.2 - best results with LCN
##        im_back = dstack([im_red_back, im_green_back, im_blue_back])

        ############################################################################################################################################################

        ## Convert image (current and background) from RGB to YCbCr and create HSI model

        # RGB to YCbCr
        def rgb2ycbcr(red, green, blue):   # source - wiki
           y = 0.299*red + 0.587*green + 0.114*blue
           cb = 128 - 0.169*red - 0.331*green + 0.5*blue
           cr = 128 + 0.5*red - 0.419*green - 0.081*blue
           return y, cb, cr

        im_y, im_cb, im_cr = rgb2ycbcr(im_red, im_green, im_blue)
        im_y_back, im_cb_back, im_cr_back = rgb2ycbcr(im_red_back, im_green_back, im_blue_back)

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
           peak_bin = np.array(np.where(np.amax(hist[0:30])))[0][0]         # PROBLEM: ALL THE IMAGES DO NOT CONFORM TO THIS CODE (POSSIBLE BUG - TRY ALL LCN IMAGES TO FIND OUT) - set exception

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


        ## The last words (binary thresholding, morphological operations and connected-components labelling)

        inty, hue, sat = im_int_diff.copy(), im_hue_diff.copy(), im_sat_diff.copy()

        inty[:,:][inty[:,:] <= search_thresh_int] = False #True
        inty[:,:][inty[:,:] > search_thresh_int] = True #False
        hue[:,:][hue[:,:] <= search_thresh_hue] = False #True
        hue[:,:][hue[:,:] > search_thresh_hue] = True #False
        sat[:,:][sat[:,:] <= search_thresh_sat] = False #True
        sat[:,:][sat[:,:] > search_thresh_sat] = True #False

        im_combinedOR = np.logical_or(inty, hue, sat)

        open_img = ndimage.binary_opening(im_combinedOR, structure = np.ones((2,2)).astype(np.int)) # works best
        close_img = ndimage.binary_closing(open_img) # open_img

        mask = close_img > close_img.mean()
        label_im, nb_labels = ndimage.label(mask)
        sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
        mask_size = sizes < 10
        remove_pixel = mask_size[label_im]
        label_im[remove_pixel] = False
        im_label = np.array(label_im > 0, dtype = uint8)            # plotting is weird, probably due to dtype conversion
        cnt, hierarchy = cv2.findContours(im_label,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if shape(cnt)[0] > 1:
            print 'too many contours in ', img_list[z], 'id# ', ins
            continue
        elif shape(cnt)[1] < 5:
            print 'rectangular contour;  possible reason: region too noisy; image# ', img_list[z], 'id# ', ins
            continue
        else:
            #### FEATURE EXTRACTION ####

            ### Colour-based features


##            im_gray = cv2.cvtColor(im_org, cv2.COLOR_BGR2GRAY)
            for h, cntr in enumerate(cnt):
                mask = np.zeros(im_int.shape, np.uint8)
                cv2.drawContours(mask,[cntr],0,255,-1)
                mean = cv2.mean(im_int, mask = mask)
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
    ##        plt.bar(centre, hist, align = 'center', width = width)
    ##        ax = plt.gca()
    ##        ax.set_xlim((0,255))
    ##        plt.show()
            
                
                
            ### Conotur-based features
            
            area = cv2.contourArea(cnt[0])                          # Area
            perimeter = cv2.arcLength(cnt[0], True)                 # Perimeter
            ellipse = cv2.fitEllipse(cnt[0])
            (centre, axes, orientation) = ellipse
            length = max(axes)                                      # Length
            width = min(axes)                                       # Width
            circular_fitness = (4*pi*area)/np.square(perimeter)     # Circular fitness
            elongation = length/width                               # Elongation
            

    ##        print 'area = '                 , area
    ##        print 'perimeter = '            , perimeter
    ##        print 'length = '               , length
    ##        print 'width = '                , width
    ##        print 'circular_fitness = '     , circular_fitness
    ##        print 'elongation = '           , elongation
    ##        print 'average intensity = '    , avg_intensity
    ##        print 'intensity histogram = '  , hist            
            
            feature_dict = {'area': area, 'perimeter': perimeter, 'length': length, 'width': width, 'circular_fitness': circular_fitness, 'elongation': elongation, 'average intensity': avg_intensity, 'intensity histogram': hist}
            feature_database.append(feature_dict)

feature_database_TrainingSet = np.array(feature_database)

print time.time() - start_time, "seconds --> Execution time"
