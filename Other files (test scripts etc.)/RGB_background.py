for y in range(shape(red_copy)[0]):
    for x in range(shape(red_copy)[1]):
        if im_fil_copy[:,:,0][y,x] <= red_avg and im_fil_copy[:,:,1][y,x] <= green_avg and im_fil_copy[:,:,2][y,x] <= blue_avg:
            im_fil_copy[:,:,0][y,x] = 255
            im_fil_copy[:,:,1][y,x] = 255
            im_fil_copy[:,:,2][y,x] = 255

        elif im_fil_copy[:,:0][y,x] <= red_avg and im_fil_copy[:,:,1][y,x] <= green_avg:
            im_fil_copy[:,:,0][y,x] = 255
            im_fil_copy[:,:,1][y,x] = 255
            im_fil_copy[:,:,2][y,x] = im_fil_copy[:,:,2][y,x]*1

        elif im_fil_copy[:,:,1][y,x] <= green_avg and im_fil_copy[:,:,2][y,x] <= blue_avg:
            im_fil_copy[:,:,0][y,x] = im_fil_copy[:,:,0][y,x]*1
            im_fil_copy[:,:,1][y,x] = 255
            im_fil_copy[:,:,2][y,x] = 255

        elif im_fil_copy[:,:,0][y,x] <= red_avg and im_fil_copy[:,:,2][y,x] <= blue_avg:
            im_fil_copy[:,:,0][y,x] = 255
            im_fil_copy[:,:,1][y,x] = im_fil_copy[:,:,1][y,x]*1
            im_fil_copy[:,:,2][y,x] = 255

        elif im_fil_copy[:,:,0][y,x] <= red_avg:
            im_fil_copy[:,:,0][y,x] = 255
            im_fil_copy[:,:,1][y,x] = im_fil_copy[:,:,1][y,x]*1
            im_fil_copy[:,:,2][y,x] = im_fil_copy[:,:,2][y,x]*1

        elif im_fil_copy[:,:,1][y,x] <= green_avg:
            im_fil_copy[:,:,0][y,x] = im_fil_copy[:,:,0][y,x]*1
            im_fil_copy[:,:,1][y,x] = 255
            im_fil_copy[:,:,2][y,x] = im_fil_copy[:,:,2][y,x]*1

        else:
            im_fil_copy[:,:,0][y,x] = im_fil_copy[:,:,0][y,x]*1
            im_fil_copy[:,:,1][y,x] = im_fil_copy[:,:,1][y,x]*1
            im_fil_copy[:,:,2][y,x] = im_fil_copy[:,:,2][y,x]*1   
