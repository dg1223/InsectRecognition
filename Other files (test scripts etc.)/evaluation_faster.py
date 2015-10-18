start_time = time.time()

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

# Training Set

cd '/export/mlrg/salavi/shamir/Annotation data set/Original Images/Good Images/Positive Counts/Training Set'


## Decode JSON file and store all the corner coordinates of ground truth in an array

json_data =  open('120_5103')           # img_list[z][:-4]
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

# Rectify errors in JSON Data

def rectify_x(array_x):
    if (array_x >= 640).any() == True:
        find_unwanted_val_x = np.where(array_x >= 640)
        for i in find_unwanted_val_x[0]:
            array_x[i] = 639

def rectify_y(array_y):
    if (array_y >= 480).any() == True:
        find_unwanted_val_y = np.where(array_y >= 480)
        for i in find_unwanted_val_y[0]:
            array_y[i] = 479
rectify_x(x)
rectify_x(x1)
rectify_y(y)
rectify_y(y1)

# Training Set LCN Labelled

cd '/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Postive Counts/TrainingSet_LCN_Labelled_1.2'


# Get corner coordinates of the detected bouning boxes (single image)

im = cv2.imread('120_5103.jpg', cv2.CV_LOAD_IMAGE_COLOR)
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
    
bb_gt_pixels, bb_dt_pixels = [], [], 

# gt & dt
def get_coordinates(c1, c11, c2, c21, bb_pixels):
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

# gt & dt
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

## The ultimate evaluation algo #Idea: divide the image into smaller grids and compare bounding boxes in a smaller neighbourhood

common_pixels, match, false_pos, false_neg = 0, 0, 0, 0

for k, v in bb_gt_ref.items():
   for k1, v1 in bb_dt_ref.items():
       if k == k1:
           if (shape(bb_gt_ref.get(k))[0] == 1) & (shape(bb_dt_ref.get(k1))[0] == 1):
               for m in range(shape(bb_gt_ref.get(k))[1]):
                   for n in range(shape(bb_dt_ref.get(k1))[1]):
                       if (bb_gt_ref.get(k)[0][m] == bb_dt_ref.get(k1)[0][n]).all() == True:
                           common_pixels += 1               
               if common_pixels > 0:
                   all_pixels = shape(bb_gt_ref.get(k))[1] + shape(bb_dt_ref.get(k1))[1] - common_pixels  # union
                   match_value = common_pixels / float(all_pixels)
                   if match_value > 0.5:
                       match += 1
               common_pixels = 0
           elif (shape(bb_gt_ref.get(k))[0] == 1) & (shape(bb_dt_ref.get(k1))[0] > 1):
               for a in range(shape(bb_dt_ref.get(k1))[0]):
                   for m in range(shape(bb_gt_ref.get(k))[1]):
                       for n in range(shape(bb_dt_ref.get(k1)[a])[0]):
                           if (bb_gt_ref.get(k)[0][m] == bb_dt_ref.get(k1)[a][n]).all() == True:
                               common_pixels += 1                   
                   if common_pixels > 0:
                       all_pixels = shape(bb_gt_ref.get(k))[1] + shape(bb_dt_ref.get(k1)[a])[0] - common_pixels
                       match_value = common_pixels / float(all_pixels)
                       if match_value > 0.5:
                           match += 1
                   common_pixels = 0
           elif (shape(bb_gt_ref.get(k))[0] > 1) & (shape(bb_dt_ref.get(k1))[0] == 1):
               for a in range(shape(bb_gt_ref.get(k))[0]):
                   for m in range(shape(bb_gt_ref.get(k)[a])[0]):
                       for n in range(shape(bb_dt_ref.get(k1))[1]):
                           if (bb_gt_ref.get(k)[a][m] == bb_dt_ref.get(k1)[0][n]).all() == True:
                               common_pixels += 1                   
                   if common_pixels > 0:
                       all_pixels = shape(bb_gt_ref.get(k)[a])[0] + shape(bb_dt_ref.get(k1))[1] - common_pixels
                       match_value = common_pixels / float(all_pixels)
                       if match_value > 0.5:
                           match += 1
                   common_pixels = 0
           else:
               for a in range(shape(bb_gt_ref.get(k))[0]):
                   for b in range(shape(bb_dt_ref.get(k1))[0]):
                       for m in range(shape(bb_gt_ref.get(k)[a])[0]):
                           for n in range(shape(bb_dt_ref.get(k1)[b])[0]):
                               if (bb_gt_ref.get(k)[a][m] == bb_dt_ref.get(k1)[b][n]).all() == True:
                                   common_pixels += 1                   
                       if common_pixels > 0:
                           all_pixels = shape(bb_gt_ref.get(k)[a])[0] + shape(bb_dt_ref.get(k1)[b])[0] - common_pixels
                           match_value = common_pixels / float(all_pixels)
                           if match_value > 0.5:
                               match += 1
                       common_pixels = 0

false_pos = shape(bb_dt_pixels)[0] - match
false_neg = shape(bb_gt_pixels)[0] - match
miss_rate = false_neg/float(shape(bb_gt_pixels)[0])
print 'match = ', match
print 'false_pos = ', false_pos
print 'false_neg = ', false_neg
print time.time() - start_time, "seconds"
