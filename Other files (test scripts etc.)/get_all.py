##import os
##import numpy as np
##import matplotlib.pyplot as plt
##import PIL
##import _imaging
##import cv2
##import json
##import sys
##import glob
##import time
##import scipy
##
##from PIL import Image
##from pprint import pprint
##from scipy import ndimage
##from mpl_toolkits.axes_grid1 import ImageGrid

start_time = time.time()

# Get corner coordinates of the detected bouning boxes (single image)

cd '/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Positive Counts/TrainingSet_LCN_Labelled_1.2'
##os.chdir('/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Positive Counts/TrainingSet_LCN_Labelled_1.2')

##fig = plt.figure(1)
##plt.subplot(1,2,1)
img_list2 = glob.glob('*.jpg')                  # creates a list of all the files with the given format
img_list2 = np.sort(np.array(img_list2))
no_of_insects = 0
insects = []
for z in range(8,9): # shape(img_list2)[0] 1,101,10
    cd '/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Positive Counts/TrainingSet_LCN_Labelled_1.2'
##    os.chdir('/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Positive Counts/TrainingSet_LCN_Labelled_1.2')
##    print img_list2[z]
    a, b, a1, b1 = [], [], [], []
    im = cv2.imread(img_list2[z], cv2.CV_LOAD_IMAGE_COLOR)   # img_list2[z]
    im_can = cv2.Canny(im, 100, 200)
    cnt, hierarchy = cv2.findContours(im_can,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    
    while i < np.shape(cnt)[0]:
       m,n,w,h = cv2.boundingRect(cnt[i])
       m = m+2
       n = n+2
       w = w+2
       h = h+2
   
       a.append(m)
       b.append(n)
       a1.append(m+w)
       b1.append(n+h)
       i += 1           # i += 2, to bypass duplicate arrays

##    a = np.array(a)
##    b = np.array(b)
##    a1 = np.array(a1)
##    b1 = np.array(b1)
##    print len(a)

    ## Extract insects

    cd '/export/mlrg/salavi/shamir/Annotation data set/Original Images/Good Images/Positive Counts/Training Set'
##    os.chdir('/export/mlrg/salavi/shamir/Annotation data set/Normalized Images/Good Images/Positive Counts/TrainingSet_LCN_Labelled_1.2')
##    print len(a)
    im = Image.open(img_list2[z])    
    for j in range(len(a)):    
       cropped = im.crop((int(a[j]),int(b[j]),int(a1[j]),int(b1[j])))
       cropped = cropped.resize((30,30), Image.ANTIALIAS)   
       cropped = np.asarray(cropped) # weirdly auto-flipped
       cropped = cv2.copyMakeBorder(cropped,1,1,1,1, cv2.BORDER_CONSTANT,value=(0,0,0))       
       insects.append(cropped)
       no_of_insects += 1
print 'no_of_insects = ', no_of_insects
insects = np.array(insects)

### Subplot
    
tile = int(np.ceil(np.sqrt(no_of_insects)))  # grid size
##print tile
grid = ImageGrid(fig, 236, # similar to subplot(111)
                nrows_ncols = (tile, tile), # creates 2x2 grid of axes
                axes_pad=0.0, # pad between axes in inch.
                share_all=True,
                )

for i in range(len(insects)):
    grid[i].imshow(insects[i], cmap = 'gray') # The AxesGrid object work as a list of axes.

grid.axes_llc.set_xticks([])
grid.axes_llc.set_yticks([])
plt.show()

print time.time() - start_time, "seconds --> Execution time"
