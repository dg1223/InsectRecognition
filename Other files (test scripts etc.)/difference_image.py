In [224]: im_fil_c = im_fil.copy()
In [225]: im_fil_c[:,:,0][im_fil_c[:,:,0] <= red_avg*0.65] = 255
In [226]: im_fil_c[:,:,1][im_fil_c[:,:,1] <= green_avg*0.65] = 255
In [227]: im_fil_c[:,:,2][im_fil_c[:,:,2] <= blue_avg*0.65] = 255
In [228]: back_im = im_fil_c.copy()
In [229]: Y_back = back_im[:,:,0]*0.299 + back_im[:,:,1]*0.587 + back_im[:,:,2]*0.114

In [230]: Cb_back = 128 - back_im[:,:,0]*0.169 - back_im[:,:,1]*0.331 + back_im[:,:,2]*0.5

In [231]: Cr_back = 128 + back_im[:,:,0]*0.5 - back_im[:,:,1]*0.419 - back_im[:,:,2]*0.081

In [232]: hue_back = np.arctan(np.divide(Cr_back, Cb_back))
In [233]: saturation_back = np.sqrt(np.square(Cr_back) + np.square(Cb_back))

In [234]: diff_int = Y_curr - Y_back
In [235]: diff_hue = hue_curr - hue_back
In [236]: diff_sat = saturation_curr - saturation_back
