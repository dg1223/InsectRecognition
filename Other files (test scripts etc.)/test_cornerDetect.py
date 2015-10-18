# Training Set LCN
cd '/mnt/ssd/shamir/Original Images/Good Images/No Counts/LCN/Training_Set'

img =  cv2.imread('120_3725.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

biggest = None
max_area = 0
for i in contours:
        area = cv2.contourArea(i)
        if area > 100:
                peri = cv2.arcLength(i,True)
                approx = cv2.approxPolyDP(i,0.02*peri,True)
                if area > max_area and len(approx)==4:
                        biggest = approx
                        max_area = area

def rectify(h):
        h = h.reshape((4,2))
        hnew = np.zeros((4,2),dtype = np.float32)
 
        add = h.sum(1)
        hnew[0] = h[np.argmin(add)]
        hnew[2] = h[np.argmax(add)]
         
        diff = np.diff(h,axis = 1)
        hnew[1] = h[np.argmin(diff)]
        hnew[3] = h[np.argmax(diff)]
  
        return hnew

approx=rectify(approx)
h = np.array([ [0,0],[449,0],[449,449],[0,449] ],np.float32)

retval = cv2.getPerspectiveTransform(approx,h)
warp = cv2.warpPerspective(gray,retval,(450,450))
