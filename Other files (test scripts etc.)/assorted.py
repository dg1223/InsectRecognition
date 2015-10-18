grid = ImageGrid(fig, 111, # similar to subplot(111)
                nrows_ncols = (3, 3), # creates 2x2 grid of axes
                axes_pad=0.5, # pad between axes in inch.
                share_all=True,
                )
############################################################################################################################################################

### Get coordinates and extract BBgt

cd '/export/mlrg/salavi/shamir/Annotation data set/Original Images/Good Images/Positive Counts/Training Set'

##im = cv2.imread('120_5174.jpg', cv2.CV_LOAD_IMAGE_COLOR)

# decode JSON
json_data =  open('120_5174')           # img_list[z][:-4]
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


## Extract BBgt

cd '/export/mlrg/salavi/shamir/Annotation data set/Original Images/Good Images/Positive Counts/Training Set'

insects = []
##no_of_insects = 0
im = Image.open('120_5174.jpg') # 132_4554 120_5123
for i in range(len(x)):
##   insects.append([])
   cropped = im.crop((int(x[i]),int(y[i]),int(x1[i]),int(y1[i]))) # 179,198,201,226     347,132,361,155
   cropped = cropped.resize((32,32), Image.ANTIALIAS)   
   cropped = np.asarray(cropped) # weirdly auto-flipped
##   cropped = cv2.copyMakeBorder(cropped,1,1,1,1, cv2.BORDER_CONSTANT,value=(0,0,0))
   insects.append(cropped)
##   no_of_insects += 1
insects = np.array(insects)a = 

############################################################################################################################################################

### Contour Features

# Get corner coordinates of the detected bouning boxes (single image)

cd '/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Positive Counts/TrainingSet_LCN_Labelled_1.2'
##os.chdir('/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Positive Counts/TrainingSet_LCN_Labelled_1.2')

img_list = glob.glob('*.jpg')                  # creates a list of all the files with the given format
img_list = np.sort(np.array(img_list))
no_of_insects = 0
insects = []
for z in range(9,10): # shape(img_list)[0] 1,101,10
    cd '/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Positive Counts/TrainingSet_LCN_Labelled_1.2'
##    os.chdir('/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Positive Counts/TrainingSet_LCN_Labelled_1.2')
    print img_list[z]
    a, b, a1, b1 = [], [], [], []
    im = cv2.imread(img_list[z], cv2.CV_LOAD_IMAGE_COLOR)   # img_list[z]
    im_can = cv2.Canny(im, 100, 200)
    cnt, hierarchy = cv2.findContours(im_can,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # use cv2.RETR_EXTERNAL

area, perimeter, length, width = [], [], [], []

i = 0
while i < np.shape(cnt)[0]:
    area.append(cv2.contourArea(cnt[i]))            # Area
    perimeter.append(cv2.arcLength(cnt[i], True))   # Perimeter
    ellipse = cv2.fitEllipse(cnt[i])
    (centre, axes, orientation) = ellipse
    length.append(max(axes))                        # Length
    width.append(min(axes))                         # Width
    i += 2

area = np.array(area)
perimeter = np.array(perimeter)
length = np.array(length)
width = np.array(width)

plt.figure(2)
grid = ImageGrid(fig, 236, # similar to subplot(111)
                nrows_ncols = (tile, tile), # creates 2x2 grid of axes
                axes_pad=0.0, # pad between axes in inch.
                share_all=True,
                )

############################################################################################################################################################


#### Mask out one insect, extract pixel values, and create histogram

i = 0
im = cv2.imread('120_5174.jpg', cv2.CV_LOAD_IMAGE_COLOR)
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
for h, cntr in enumerate(cnt):
    i += 1
    mask = np.zeros(imgray.shape,np.uint8)
    cv2.drawContours(mask,[cntr],0,255,-1)
    mean = cv2.mean(im,mask = mask)
    if i == 15:
        break

x = np.where(mask > 0)
y = x[1][:]
z = x[0][:]
##x_mask, y_mask, cntour = [], [], []
##cntour = dstack([x_mask, y_mask])


pixels = []
def create_hist(pixel, image, x_coord, y_coord):
   for i in range(len(x_coord)):
       pixel.append(image[y_coord[i],x_coord[i]])
   hist, bins = np.histogram(pixel, bins = 63, range = (0.0, 255.0))
   width = 0.7*(bins[1] - bins[0])                 # just for plotting the histogram (comment it out if unnecessary)
   centre = (bins[:-1] + bins[1:])/2               # just for plotting the histogram (comment it out if unnecessary)
   return hist, bins, width, centre 

hist, bins, width, centre = create_hist(pixels, im_y, y, z)
plt.bar(centre, hist, align = 'center', width = width)
ax = plt.gca()
ax.set_xlim((0,255))
plt.show()

############################################################################################################################################################








       
