#### FEATURE EXTRACTION ####

            ### Colour-based features

            for h, cntr in enumerate(cnt):
                mask = np.zeros(im_int.shape, np.uint8)
                cv2.drawContours(mask,[cntr],0,255,-1)
                mean = cv2.mean(im, mask = mask)
            find = np.where(mask > 0)
            x_axis = find[1][:]
            y_axis = find[0][:]
            

            ## Average intensity
            
            intensity = []
            def avg_int(ipixel, image, x_coord, y_coord):
               for i in range(len(x_coord)):
                   ipixel.append(image[y_coord[i],x_coord[i]])
               mean = round(np.mean(np.array(ipixel)))
               return mean

            avg_intensity = avg_int(intensity, im_int, x_axis, y_axis)


            ## Intensity histogram

            pixels = []
            def create_hist(hpixel, image, x_coord, y_coord):
               for i in range(len(x_coord)):
                   hpixel.append(image[y_coord[i],x_coord[i]])
               hist, bins = np.histogram(hpixel, bins = 64, range = (0.0, 255.0))
               width = 0.7*(bins[1] - bins[0])                 # just for plotting the histogram (comment it out if unnecessary)
               centre = (bins[:-1] + bins[1:])/2               # just for plotting the histogram (comment it out if unnecessary)
               return hist, bins, width, centre 

            hist, bins, width, centre = create_hist(pixels, im_int, x_axis, y_axis)
    ##        plt.bar(centre, hist, align = 'center', width = width)
    ##        ax = plt.gca()
    ##        ax.set_xlim((0,255))
    ##        plt.show()
            
                
                
            ### Conotur-based features

            if np.shape(cnt)[0] == 1:
                area = cv2.contourArea(cnt[0])                          # Area
                perimeter = cv2.arcLength(cnt[0], True)                 # Perimeter
                ellipse = cv2.fitEllipse(cnt[0])
                (centre, axes, orientation) = ellipse
                length = max(axes)                                      # Length
                width = min(axes)                                       # Width
                circular_fitness = (4*pi*area)/np.square(perimeter)     # Circular fitness
                elongation = length/width                               # Elongation
            else:
                print 'possibly a broken contour in ', img_list[z], 'id# ', ins

    ##        print 'area = '                 , area
    ##        print 'perimeter = '            , perimeter
    ##        print 'length = '               , length
    ##        print 'width = '                , width
    ##        print 'circular_fitness = '     , circular_fitness
    ##        print 'elongation = '           , elongation
    ##        print 'average intensity = '    , avg_intensity
    ##        print 'intensity histogram = '  , hist            
            
            feature_dict = {'area': area, 'perimeter': perimeter, 'length': length, 'width': width, 'circular_fitness': circular_fitness, 'elongation': elongation, 'average intensity': avg_intensity, 'intensity histogram': hist}
            feature_database.append(feature_dict)
feature_database = np.array(feature_database)
