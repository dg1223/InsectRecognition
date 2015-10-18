import cv2

img = cv2.imread('120_5096.jpg', cv2.CV_LOAD_IMAGE_COLOR)
cv2.namedWindow('Image')
cv2.imshow('Image',img)
cv2.waitKey(0)
##cv2.destroyAllWindows()

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('color_image',image)
cv2.imshow('gray_image',gray_image)
cv2.waitKey(0)


In [81]: centroids = np.array(centroids,dtype = np.float32)

In [82]: c = centroids.reshape((shape(centroids)[0],shape(centroids)[1]))

In [86]: a = shape(centroids)[0]

In [87]: b = np.vstack([c2[i*(a/10):(i+1)*(a/10)][np.argsort(c2[i*(a/10):(i+1)*(a/10),0])] for i in xrange(a/10)])

bm = b.reshape(((a/10),(a/10),2))



############################################################################################################################################################

## unwarp image (Sudoku Solver example)

# 1. Image PreProcessing ( closing operation )

img = cv2.imread('124_4038.jpg') # 120_5096
img = cv2.GaussianBlur(img,(5,5),0)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
mask = np.zeros((gray.shape),np.uint8)
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)) # ellipsiodal kernel (should test using other kernels to see the difference)

close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1) # advanced morphological transformation (closing operation - removing small black areas)
div = np.float32(gray)/(close) # cleans up the image (no idea why!)
res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX)) # linear normalization
res2 = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR) # gray to RGB

# 2. Finding Sudoku Square and Creating Mask Image

thresh = cv2.adaptiveThreshold(res,255,0,1,19,2)
contour,hier = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

max_area = 0
best_cnt = None
for cnt in contour:
    area = cv2.contourArea(cnt)
    if area > 1000: # need to test other values for project images
        if area > max_area:
            max_area = area
            best_cnt = cnt

cv2.drawContours(mask,[best_cnt],0,255,-1)
cv2.drawContours(mask,[best_cnt],0,0,2)

res = cv2.bitwise_and(res,mask)

# 3. Finding Vertical lines

kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(2,10)) # a 2 by 10 rectangle

dx = cv2.Sobel(res,cv2.CV_16S,1,0)
dx = cv2.convertScaleAbs(dx)
cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)

contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contour:
    x,y,w,h = cv2.boundingRect(cnt)
    if h/w > 5:
        cv2.drawContours(close,[cnt],0,255,-1)
    else:
        cv2.drawContours(close,[cnt],0,0,-1)
close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
closex = close.copy()

# 4. Finding Horizontal Lines

kernely = cv2.getStructuringElement(cv2.MORPH_RECT,(10,2))
dy = cv2.Sobel(res,cv2.CV_16S,0,2)
dy = cv2.convertScaleAbs(dy)
cv2.normalize(dy,dy,0,255,cv2.NORM_MINMAX)
ret,close = cv2.threshold(dy,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernely)

contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contour:
    x,y,w,h = cv2.boundingRect(cnt)
    if w/h > 5:
        cv2.drawContours(close,[cnt],0,255,-1)
    else:
        cv2.drawContours(close,[cnt],0,0,-1)

close = cv2.morphologyEx(close,cv2.MORPH_DILATE,None,iterations = 2)
closey = close.copy()

# 5. Finding Grid Points

res = cv2.bitwise_and(closex,closey)

############################################################################################################################################################

## get rid of the grids

res = cv2.bitwise_or(closex, closey)
x = np.where(res > 0)
x = np.array(x)

for i in range(shape(x)[1]):
    gray[x[0,i], x[1,i]] = 255

############################################################################################################################################################

# 6. Correcting the defects

contour, hier = cv2.findContours(res,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
centroids = []
for cnt in contour:
    mom = cv2.moments(cnt)
    (x,y) = int(mom['m10']/mom['m00']), int(mom['m01']/mom['m00'])
    cv2.circle(img,(x,y),4,(0,255,0),-1)
    centroids.append((x,y))

centroids = np.array(centroids,dtype = np.float32)
c = centroids.reshape((shape(centroids)[0],2)) # check whether this line is required or not

### just to properly sort the centroid indices

c2 = c[np.argsort(c[:,1])] # sort along y-axis (sort the values along y-axis only leaving x-axis as it is to get the coordinates with y-axis sorted)
                           # (100,2) array

b = np.vstack([c2[i*10:(i+1)*10][np.argsort(c2[i*10:(i+1)*10,0])] for i in xrange(10)])
bm = b.reshape((10,10,2))
####

# unwarp the image

output = np.zeros((450,450,3),np.uint8)
for i,j in enumerate(b): # almost like a percentage basis
    ri = i/10
    ci = i%10
    if ci != 9 and ri!=9:
        src = bm[ri:ri+2, ci:ci+2 , :].reshape((4,2))
        dst = np.array( [ [ci*50,ri*50],[(ci+1)*50-1,ri*50],[ci*50,(ri+1)*50-1],[(ci+1)*50-1,(ri+1)*50-1] ], np.float32) # see tutorial for details (logic)
        retval = cv2.getPerspectiveTransform(src,dst)
        warp = cv2.warpPerspective(res2,retval,(450,450))
        output[ri*50:(ri+1)*50-1 , ci*50:(ci+1)*50-1] = warp[ri*50:(ri+1)*50-1 , ci*50:(ci+1)*50-1].copy()

############################################################################################################################################################

## edit for project

a = round(shape(c2)[0]/10)
b = np.vstack([c2[i*a:(i+1)*a][np.argsort(c2[i*a:(i+1)*a,0])] for i in xrange(10)])
a1 = round(shape(c2)[0]/a)
bm = b.reshape((a1,a,2))
output = np.zeros((432, 576, 3),np.uint8)

for i,j in enumerate(b):
    ri = i/6
    ci = i%10
    if ri != 9 and ci < 5:
        src = bm[ri:ri+2, ci:ci+2 , :].reshape((4,2))
        #print i, src
        dst = np.array( [ [ci*50,ri*50],[(ci+1)*50-1,ri*50],[ci*50,(ri+1)*50-1],[(ci+1)*50-1,(ri+1)*50-1] ], np.float32)
        retval = cv2.getPerspectiveTransform(src,dst)
        warp = cv2.warpPerspective(res2,retval,(432, 576))
        output[ri*50:(ri+1)*50-1 , ci*50:(ci+1)*50-1] = warp[ri*50:(ri+1)*50-1 , ci*50:(ci+1)*50-1].copy()


retval = cv2.getPerspectiveTransform(src,dst)
warp = cv2.warpPerspective(res2,retval,(450,450))
output[ri*50:(ri+1)*50-1 , ci*50:(ci+1)*50-1] = warp[ri*50:(ri+1)*50-1 , ci*50:(ci+1)*50-1].copy()
