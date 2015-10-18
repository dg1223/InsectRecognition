### Training Set LCN (No Counts)
##cd '/mnt/ssd/shamir/Original Images/Good Images/No Counts/LCN/Training_Set'
##
### Training Set LCN (+ve counts)
##cd '/mnt/ssd/shamir/Normalized Images/Good Images/Positive Counts/TrainingSet_LCN'
##
# Test Set LCN (no counts)
cd '/mnt/ssd/shamir/Original Images/Good Images/No Counts/LCN/Test_Set'

### Test Set LCN (+ve)
##cd '/mnt/ssd/shamir/Normalized Images/Good Images/Positive Counts/Test_Set_LCN'

## Open image

img_list2 = glob.glob('*.jpg')                                          # creates a list of all the files with the given format
img_list2 = sort(np.array(img_list2))
for z in range(shape(img_list2)[0]):
    ### detect corners

    # 1. Image PreProcessing ( closing operation )

    img = cv2.imread(img_list2[z], cv2.CV_LOAD_IMAGE_COLOR) # 120_5096
    img = cv2.GaussianBlur(img,(5,5),0)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mask = np.zeros((gray.shape),np.uint8)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)) # ellipsiodal kernel (should test using other kernels to see the difference)

    close = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel1) # advanced morphological transformation (closing operation - removing small black areas)
    div = np.float32(gray)/(close) # cleans up the image (no idea why!)
    res = np.uint8(cv2.normalize(div,div,0,255,cv2.NORM_MINMAX)) # linear normalization
    res2 = cv2.cvtColor(res,cv2.COLOR_GRAY2BGR) # gray to RGB

    # 2. Finding Squares and Creating Mask Image

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

    a = np.where(gray == 255)
    x_min = np.min(a[1])
    y_min = np.min(a[0])
    x_max = np.max(a[1])
    y_max = np.max(a[0])

    im_org = Image.open(img_list2[z])
    cropped = im_org.crop((x_min, y_min, x_max, y_max))
    cropped = np.asarray(cropped)
    im = cropped.copy()

    # save image
##    scipy.misc.imsave('/mnt/ssd/shamir/Normalized Images/Good Images/Positive Counts/TrainingSet_LCN_cropped/'+ img_list2[z], cropped)
##    scipy.misc.imsave('/mnt/ssd/shamir/Normalized Images/Good Images/Positive Counts/TestSet_LCN_cropped/'+ img_list2[z], cropped)
##    scipy.misc.imsave('/mnt/ssd/shamir/Normalized Images/Good Images/Positive Counts/TrainingSet_LCN_nc_cropped/'+ img_list2[z], cropped)
    scipy.misc.imsave('/mnt/ssd/shamir/Normalized Images/Good Images/Positive Counts/TestSet_LCN_nc_cropped/'+ img_list2[z], cropped)

