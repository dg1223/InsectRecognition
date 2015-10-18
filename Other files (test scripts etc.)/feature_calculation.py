# find contours

cd '/mnt/ssd/shamir/insects_gt'

img_list = glob.glob('*.jpg')                   # creates a list of all the files with the given format
img_list = np.sort(np.array(img_list))        

##s = np.array([[1,1,1],
##             [1,1,1],
##             [1,1,1]])
for z in range(661,662):                           # shape(img_list)[0]
##    cd '/mnt/ssd/shamir/insects_gt'
##    print img_list[z]
##    im = plt.imread()
    im = cv2.imread(img_list[z], cv2.CV_LOAD_IMAGE_COLOR)   # img_list[z]        
    im_fil1 = scipy.ndimage.filters.median_filter(im, size = (4,3,4))
    im_gray = cv2.cvtColor(im_fil1,cv2.COLOR_RGB2GRAY)
##    thresh = cv2.adaptiveThreshold(imgray,1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 1)
    cnt, hierarchy = cv2.findContours(im_gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
##    im_can = cv2.Canny(im_fil1, 100, 200)
##    im_can = filter.canny(im_fil1, sigma = 1)
##    contours = measure.find_contours(im_can, 0.8, 'high')
##    im_close = ndimage.binary_closing(im_can)

    mask = im_can > im_can.mean()
    label_im, nb_labels = ndimage.label(mask, structure=s)
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    mask_size = sizes < np.max(sizes)
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = False
    label_im = np.array(label_im > 0, dtype = int)
##    im_final_morph_label, num_components = ndimage.label(label_im)
    labelled = label(label_im, neighbors=8)

    cd '/export/mlrg/salavi/Desktop/Codes'
    
    contours = measure.find_contours(label_im, 0.8, 'high')
    for n, contour in enumerate(contours):
        plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
##    plt.imshow(label_im, cmap = 'gray')
    scipy.misc.imsave('/mnt/ssd/shamir/' + 'test1.jpg', label_im)
    
    

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

print 'area = ', area
print 'perimeter = ', perimeter
print 'length = ', length
print 'width = ', width
