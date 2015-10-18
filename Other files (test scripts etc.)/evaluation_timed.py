import timeit
t = timeit.timeit('evaluate()', number=1)
def evaluate():
    # Training Set

    cd '/mnt/data/shamir/Annotation data set/Original Images/Good Images/Positive Counts/Training Set'


    ## Decode JSON file and store all the corner coordinates of ground truth in an array

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


    # Training Set LCN Labelled

    cd '/mnt/data/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/TrainingSet_LCN_Labelled_1.2'


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


    ## The ultimate evaluation algo (as fast as a slug right now) #Idea: divide the image into smaller grids and compare bounding boxes in a close neighbourhood

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
elapsed = timeit.timeit() - t
print elapsed, "seconds"
