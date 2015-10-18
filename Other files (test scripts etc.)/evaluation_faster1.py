start_time = time.time()

# Training Set

cd '/mnt/data/shamir/Annotation data set/Original Images/Good Images/Positive Counts/Training Set'


## Decode JSON file and store all the corner coordinates of ground truth in an array

json_data =  open('T4_r3p6')           # list[z][:-4]
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


# Training Set LCN Labelled

cd '/mnt/data/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/TrainingSet_LCN_Labelled_1.2'


# Get corner coordinates of the detected bouning boxes (single image)

im = cv2.imread('T4_r3p6.jpg', cv2.CV_LOAD_IMAGE_COLOR)
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
bb_gt_pixels = np.array(bb_gt_pixels)

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
bb_dt_pixels = np.array(bb_dt_pixels)

## Assign numbers to the corresponding Bounding Boxes

#gt
bb_gt_ref = defaultdict(list)
for i in range(shape(bb_gt_pixels)[0]):
   for j in range(shape(grid)[0]):
      if (bb_gt_pixels[i,0,0][0,0] <= grid_ref.items()[j][1][0][0,0,4799,0]) & (bb_gt_pixels[i,0,0][0,1] <= grid_ref.items()[j][1][0][0,0,4799,1]):
         bb_gt_ref[j].append(bb_gt_pixels[i][0][0])
         break

#dt
bb_dt_ref = defaultdict(list)
for i in range(shape(bb_dt_pixels)[0]):
   for j in range(shape(grid)[0]):
      if (bb_dt_pixels[i,0,0][0,0] <= grid_ref.items()[j][1][0][0,0,4799,0]) & (bb_dt_pixels[i,0,0][0,1] <= grid_ref.items()[j][1][0][0,0,4799,1]):
         bb_dt_ref[j].append(bb_dt_pixels[i][0][0])
         break

## The ultimate evaluation algo #Idea: divide the image into smaller grids and compare bounding boxes in a smaller neighbourhood

ref = []
common_pixels, match, false_pos, false_neg = 0, 0, 0, 0

for k,v in bb_dt_ref.items():
    ref.append(k)
ref = np.array(ref)

for k, v in bb_gt_ref.items():
    if (ref == k).any() == True:
        if (shape(bb_gt_ref.get(k))[0] == 1) & (shape(bb_dt_ref.get(k))[0] == 1):
            for m in range(shape(bb_gt_ref.get(k))[1]):
                for n in range(shape(bb_dt_ref.get(k))[1]):
                    if (bb_gt_ref.get(k)[0][m] == bb_dt_ref.get(k)[0][n]).all() == True:
                        common_pixels += 1               
            if common_pixels > 0:
                all_pixels = shape(bb_gt_ref.get(k))[1] + shape(bb_dt_ref.get(k))[1] - common_pixels  # union
                match_value = common_pixels / float(all_pixels)
                if match_value > 0.5:
                    match += 1
            common_pixels = 0
        elif (shape(bb_gt_ref.get(k))[0] == 1) & (shape(bb_dt_ref.get(k))[0] > 1):
            for a in range(shape(bb_dt_ref.get(k))[0]):
                for m in range(shape(bb_gt_ref.get(k))[1]):
                    for n in range(shape(bb_dt_ref.get(k)[a])[0]):
                        if (bb_gt_ref.get(k)[0][m] == bb_dt_ref.get(k)[a][n]).all() == True:
                            common_pixels += 1                   
                if common_pixels > 0:
                    all_pixels = shape(bb_gt_ref.get(k))[1] + shape(bb_dt_ref.get(k)[a])[0] - common_pixels
                    match_value = common_pixels / float(all_pixels)
                    if match_value > 0.5:
                        match += 1
                common_pixels = 0
        elif (shape(bb_gt_ref.get(k))[0] > 1) & (shape(bb_dt_ref.get(k))[0] == 1):
            for a in range(shape(bb_gt_ref.get(k))[0]):
                for m in range(shape(bb_gt_ref.get(k)[a])[0]):
                    for n in range(shape(bb_dt_ref.get(k))[1]):
                        if (bb_gt_ref.get(k)[a][m] == bb_dt_ref.get(k)[0][n]).all() == True:
                            common_pixels += 1                   
                if common_pixels > 0:
                    all_pixels = shape(bb_gt_ref.get(k)[a])[0] + shape(bb_dt_ref.get(k))[1] - common_pixels
                    match_value = common_pixels / float(all_pixels)
                    if match_value > 0.5:
                        match += 1
                common_pixels = 0
        else:
            for a in range(shape(bb_gt_ref.get(k))[0]):
                for b in range(shape(bb_dt_ref.get(k))[0]):
                    for m in range(shape(bb_gt_ref.get(k)[a])[0]):
                        for n in range(shape(bb_dt_ref.get(k)[b])[0]):
                            if (bb_gt_ref.get(k)[a][m] == bb_dt_ref.get(k)[b][n]).all() == True:
                                common_pixels += 1                   
                    if common_pixels > 0:
                        all_pixels = shape(bb_gt_ref.get(k)[a])[0] + shape(bb_dt_ref.get(k)[b])[0] - common_pixels
                        match_value = common_pixels / float(all_pixels)
                        if match_value > 0.5:
                            match += 1
                    common_pixels = 0

false_pos = shape(bb_dt_pixels)[0] - match
false_neg = shape(bb_gt_pixels)[0] - match
print 'match = ', match
print 'false_pos = ', false_pos
print 'false_neg = ', false_neg
print time.time() - start_time, "seconds"
