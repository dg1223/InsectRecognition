start_time = time.time()

## Make an array of the 64 grids for a 640x480 image (i.e. divide the image into 64 grids)

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

# Make a dictionary of the grids (assigning a serial number to each grid)
grid_ref = defaultdict(list)
for i in range(shape(grid)[0]):
   grid_ref[i].append(grid[i])

### Training Set
##cd '/export/mlrg/salavi/shamir/Annotation data set/Original Images/Good Images/Positive Counts/Training Set'

# Test Set
cd '/mnt/ssd/shamir/Original Images/Good Images/Positive Counts/Test Set'

# create file list
img_list = glob.glob('*.jpg')
img_list = sort(np.array(img_list))

thresh = np.array([0.1])#np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])      # overlap thresholds for testing

# Initialize all; mr = miss rate, FP = False Positives, FN = False Negatives, FFPI = False Positives Per Image, FNPI = False Negatives Per Image
mr_array, all_FPPI, all_FNPI, all_match, total_matches, total_FPs, total_FNs = [], [], [], [], [], [], []
total_FP, total_FN, total_match, grand_total_match, grand_total_FP, grand_total_FN = 0, 0, 0, 0, 0, 0

for t in thresh:
##    print t
    mr_array.append([])
    for z in range(0,1):      # shape(img_list)[0]
        ## Decode JSON file and store all the corner coordinates of ground truth in an array
        
##        # Training Set
##        cd '/export/mlrg/salavi/shamir/Annotation data set/Original Images/Good Images/Positive Counts/Training Set'

        # Test Set
        cd '/mnt/ssd/shamir/Original Images/Good Images/Positive Counts/Test Set'

        json_data =  open(img_list[z][:-4])           # img_list[z][:-4]
        print img_list[z]
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
        x,y,x1,y1 = tlx+3, tly+3, brx, bry

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
        rectify(x, 640)
        rectify(x1, 640)
        rectify(y, 480)
        rectify(y1, 480)


##        # Training Set LCN Labelled
##        cd '/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/TrainingSet_LCN_Labelled_1.2'

        # Test Set LCN Labelled
        cd '/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Positive Counts/5_TestSet_LCN_Labelled_1.2'

        # Get corner points of the detected bouning boxes (BBdt)
        im = cv2.imread(img_list[z], cv2.CV_LOAD_IMAGE_COLOR)   # img_list[z]
        im_can = cv2.Canny(im, 100, 200)                                                  # Canny edge detection
        cnt, hierarchy = cv2.findContours(im_can,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   # find contours
        i = 0
        a, b, a1, b1 = [], [], [], []

        while i < shape(cnt)[0]:
           m,n,w,h = cv2.boundingRect(cnt[i])                                             # bounding rectangle
           # stretching the BB to reduce the size discrepancy between BBgt and BBdt 
           m = m+2
           n = n+2
           w = w+2
           h = h+2
           a.append(m)
           b.append(n)
           a1.append(m+w)
           b1.append(n+h)
           i += 2                # OpenCV's 'findContours' function duplicates contour arrays for some reasons. So we take every alternate array to override
                                 # this problem and get the actual number of bounding boxes
        a = np.array(a)
        b = np.array(b)
        a1 = np.array(a1)
        b1 = np.array(b1)


        ## Get all the coordinates of both ground truth (gt) and  detected (dt) boxes
    
        bb_gt_pixels, bb_dt_pixels = [], []                       # bb or BB = bounding box
               
        def get_coordinates(c1, c11, c2, c21, bb_pixels):         # c = corner
           x_val, y_val = [], []
           for k in range(shape(c1)[0]):
              bb_pixels.append([])
              for i in range(c1[k], c11[k]+1):
                 for j in range(c2[k], c21[k]+1):
                    x_val.append(i)
                    y_val.append(j)
              x_val = np.array(x_val)
              y_val = np.array(y_val)
              bb_pixels[k].append(dstack([x_val, y_val]))
              x_val, y_val = [], []

        # BBgt
        get_coordinates(x, x1, y, y1, bb_gt_pixels)
        bb_gt_pixels = np.array(bb_gt_pixels)
        # BBdt
        get_coordinates(a, a1, b, b1, bb_dt_pixels)
        bb_dt_pixels = np.array(bb_dt_pixels)
        

        ## Assign numbers to the corresponding Bounding Boxes

        # Make a dictionary for each type of BB which will contain the relevant pixel arrays and their corresponding grid numbers
        bb_gt_ref = defaultdict(list)
        bb_dt_ref = defaultdict(list)

        def assign_numbers(bb_pixels, bb_ref):
           for i in range(shape(bb_pixels)[0]):
              for j in range(shape(grid)[0]):
                 if (bb_pixels[i,0,0][0,0] <= grid_ref.items()[j][1][0][0,0,4799,0]) & (bb_pixels[i,0,0][0,1] <= grid_ref.items()[j][1][0][0,0,4799,1]):
                    bb_ref[j].append(bb_pixels[i][0][0])
                    break
        assign_numbers(bb_gt_pixels, bb_gt_ref)       # for BBgt
        assign_numbers(bb_dt_pixels, bb_dt_ref)       # for BBdt


        ## The actual evaluation algo #Idea: divide the image into smaller grids and compare bounding boxes in a smaller neighbourhood by grid matching

        common_pixels, match, false_pos, false_neg = 0, 0, 0, 0

        breaker = 0
        for k, v in bb_gt_ref.items():
           for k1, v1 in bb_dt_ref.items():
               w = np.array([k1, k1-1, k1+1, k1-8, k1+8, k1-7, k1+7, k1-9, k1+9])                     # taking a 1 by 1 grid neighbourhood
               if (k == w).any() == True:                   # if the grid numbers are a match to any of these, then both types of BBs are in the same neighbourhood
                   if (shape(bb_gt_ref.get(k))[0] == 1) & (shape(bb_dt_ref.get(k1))[0] == 1):   # if there is only one BBgt and BBdt in this grid
                       for m in range(shape(bb_gt_ref.get(k))[1]):
                           for n in range(shape(bb_dt_ref.get(k1))[1]):
                               if (bb_gt_ref.get(k)[0][m] == bb_dt_ref.get(k1)[0][n]).all() == True:  # if any of the coordinates are common, it's an overlap                             
                                   common_pixels += 1               
                       if common_pixels > 0:
                           all_pixels = shape(bb_gt_ref.get(k))[1] + shape(bb_dt_ref.get(k1))[1] - common_pixels  # union
                           match_value = common_pixels / float(all_pixels)          # match_value = intersection/union (of the BB areas)
                           common_pixels = 0
                           if match_value > t:
                               match += 1
                               break
                       
                   elif (shape(bb_gt_ref.get(k))[0] == 1) & (shape(bb_dt_ref.get(k1))[0] > 1):  # if there is only one BBgt but multiple BBdt in this grid
                       for a in range(shape(bb_dt_ref.get(k1))[0]):                             # for every dt in this grid
                           for m in range(shape(bb_gt_ref.get(k))[1]):
                               for n in range(shape(bb_dt_ref.get(k1)[a])[0]):
                                   if (bb_gt_ref.get(k)[0][m] == bb_dt_ref.get(k1)[a][n]).all() == True:
                                       common_pixels += 1                   
                           if common_pixels > 0:
                               all_pixels = shape(bb_gt_ref.get(k))[1] + shape(bb_dt_ref.get(k1)[a])[0] - common_pixels
                               match_value = common_pixels / float(all_pixels)
                               common_pixels = 0
                               if match_value > t:
                                   match += 1
                                   breaker += 1             # if there is a match, then there's no point in traversing the rest of the neighbouring grids
                                   break                               
                       if breaker > 0:
                           breaker = 0
                           break
                   elif (shape(bb_gt_ref.get(k))[0] > 1) & (shape(bb_dt_ref.get(k1))[0] == 1):  # if there are multiple BBgt but only BBdt in this grid
                       for a in range(shape(bb_gt_ref.get(k))[0]):                              # for every gt in this grid
                           for m in range(shape(bb_gt_ref.get(k)[a])[0]):
                               for n in range(shape(bb_dt_ref.get(k1))[1]):
                                   if (bb_gt_ref.get(k)[a][m] == bb_dt_ref.get(k1)[0][n]).all() == True:
                                       common_pixels += 1                   
                           if common_pixels > 0:
                               all_pixels = shape(bb_gt_ref.get(k)[a])[0] + shape(bb_dt_ref.get(k1))[1] - common_pixels
                               match_value = common_pixels / float(all_pixels)
                               common_pixels = 0
                               if match_value > t:
                                   match += 1
                                   breaker += 1             # if there is a match, then there's no point in traversing the rest of the neighbouring grids
                                   break                           
                       if breaker > 0:
                           breaker = 0
                           break
                   else:
                       # This section CAN BE OPTIMIZED --> (for each gt, if a matching dt is found, remove it from the array before next iteration)
                       for b in range(shape(bb_dt_ref.get(k1))[0]):                              # for every gt in this grid
                           for a in range(shape(bb_gt_ref.get(k))[0]):                         # for every dt in this grid
                               for m in range(shape(bb_gt_ref.get(k)[a])[0]):
                                   for n in range(shape(bb_dt_ref.get(k1)[b])[0]):
                                       if (bb_gt_ref.get(k)[a][m] == bb_dt_ref.get(k1)[b][n]).all() == True:
                                           common_pixels += 1                   
                               if common_pixels > 0:
                                   all_pixels = shape(bb_gt_ref.get(k)[a])[0] + shape(bb_dt_ref.get(k1)[b])[0] - common_pixels
                                   match_value = common_pixels / float(all_pixels)
                                   common_pixels = 0
                                   if match_value > t:
                                       match += 1
                                       breaker += 1
                                       break                               
                       if breaker ==  shape(bb_dt_ref.get(k1))[0]:    # if all the BBgt found their matches, do not traverse rest of the neighbouring grids
                           breaker = 0
                           break
        # for a single image
        total_match += match
        
        false_pos = shape(bb_dt_pixels)[0] - match  # number of FPs for the current image
        total_FP += false_pos        

        false_neg = shape(bb_gt_pixels)[0] - match  # number of FNs for the current image
        total_FN += false_neg
        

        miss_rate = false_neg/float(shape(bb_gt_pixels)[0])         # miss rate for the current image
        mr_array[np.where(thresh == t)[0][0]].append(miss_rate)     # all the miss rates for all the images for all the threshold values

        match, false_pos, false_neg = 0, 0, 0

    # for a single threshold value
    total_FPs.append(total_FP)      # total_FP for every threshold value
    grand_total_FP += total_FP      # overall FP
    FPPI = total_FP/len(img_list)   # False Positives Per Image
    all_FPPI.append(FPPI)

    total_FNs.append(total_FN)      # total_FP for every threshold value
    grand_total_FN += total_FN      # overall FN
    FNPI = total_FN/len(img_list)   # False Negatives Per Image
    all_FNPI.append(FNPI)

    total_matches.append(total_match)     # total_matches for every threshold value
    grand_total_match += total_match      # overall matches
    MPI = total_match/len(img_list)       # Matches Per Image
    all_match.append(MPI)
    
    total_FP, total_FN, total_match = 0, 0, 0

# overall stats
total_matches = np.array(total_matches)
total_FPs = np.array(total_FPs)
total_FNs = np.array(total_FNs)
mr_array = np.array(mr_array)
all_FPPI = np.array(all_FPPI)
all_FNPI = np.array(all_FNPI)
all_match = np.array(all_match)

print 'all_matches = ', total_matches
print 'all_FP = ', total_FPs
print 'all_FN = ', total_FNs
print time.time() - start_time, "seconds --> Execution time"
