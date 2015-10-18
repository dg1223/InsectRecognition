random.seed()
probabilities = clf.predict_proba(X_test)
for i in range(len(probabilities)):
    probabilities[i] = probabilities[i] + (np.random.rand()*0.00001)
    
thresh = np.array([0.5])#, 0.7, 0.8, 0.9, 1.0])
for j in thresh:
    if j == 1.0:
        matches = [i for i in probabilities if i == j]
        print len(matches), 'matches @ threshold', j
    else:
        matches = [i for i in probabilities if i > j]
        print len(matches), 'matches @ threshold', j

matches = sorted(matches, reverse = True)
common_pixels, match, false_pos, false_neg, breaker, match_val = 0, 0, 0, 0, 0, []
matched_gt = np.array([])
for i in range(len(matches)): #len(matches)
##    print i
    index_p = np.where(probabilities == matches[i])[0][0]
    print 'index = ', index_p
    for k1, v1 in bb_dt_ref.items():
##        print 'dt ',k1
        if shape(bb_dt_ref.get(k1))[0] == 1:                                                    # if one BBdt in grid
##            print k1, ' has shape 1'
            if (bb_dt_ref.get(k1)[0][0] == bb_dt_pixels[index_p][0][0][0]).all() == True:       # match 1st array
                for k, v in bb_gt_ref.items():
                    if (k == matched_gt).any() == True:     # remember to make it empty before each image
                        continue
                    else:
                        w = np.array([k, k-1, k+1, k-8, k+8, k-7, k+7, k-9, k+9])
                        if ((k1 == w).any() == True) & (shape(bb_gt_ref.get(k))[0] == 1):           # one BBgt
                            for m in range(shape(bb_gt_ref.get(k))[1]):
                                for n in range(shape(bb_dt_ref.get(k1))[1]):
                                    if (bb_gt_ref.get(k)[0][m] == bb_dt_ref.get(k1)[0][n]).all() == True:  # if any of the coordinates are common, it's an overlap
                                        common_pixels += 1
                            if common_pixels > 0:
                                all_pixels = shape(bb_gt_ref.get(k))[1] + shape(bb_dt_ref.get(k1))[1] - common_pixels  # union
                                match_value = common_pixels / float(all_pixels)                     # match_value = intersection/union (of the BB areas)
                                if match_value > t:
                                    match += 1
                                    matched_gt = matched_gt.tolist()
                                    matched_gt.append(k)
                                    matched_gt = np.array(matched_gt)
                                    print 'match in loop 1!!'
    ##                                print 'found a match @ (1 1) '
                            common_pixels = 0
                        elif ((k1 == w).any() == True) & (shape(bb_gt_ref.get(k))[0] > 1):          # multiple BBgt
                            for a in range(shape(bb_gt_ref.get(k))[0]):
                                for m in range(shape(bb_gt_ref.get(k)[a])[0]):
                                    for n in range(shape(bb_dt_ref.get(k1))[1]):
                                        if (bb_gt_ref.get(k)[a][m] == bb_dt_ref.get(k1)[0][n]).all() == True:
                                            common_pixels += 1
                                if common_pixels > 0:
                                    all_pixels = shape(bb_gt_ref.get(k)[a])[0] + shape(bb_dt_ref.get(k1))[1] - common_pixels
                                    match_value = common_pixels / float(all_pixels)
                                    if match_value > t:
                                        match += 1
                                        matched_gt = matched_gt.tolist()
                                        matched_gt.append(k)
                                        matched_gt = np.array(matched_gt)
                                        print 'match in loop 2!!'
                                        breaker += 1             # if there is a match, then there's no point in traversing the rest of the neighbouring grids
                                        break
                                common_pixels = 0
                            if breaker > 0:
                                breaker = 0
                                break 
        elif shape(bb_dt_ref.get(k1))[0] > 1:                                                                                   # if multiple BBdt in grid
            for a in range(shape(bb_dt_ref.get(k1))[0]):                
                if (bb_dt_ref.get(k1)[a][0] == bb_dt_pixels[index_p][0][0][0]).all() == True:   # match 1st array
##                    print k1, a
                    for k, v in bb_gt_ref.items():                        
                        if (k == matched_gt).any() == True:     # remember to make it empty before each image
                            continue
                        else:                            
                            w = np.array([k, k-1, k+1, k-8, k+8, k-7, k+7, k-9, k+9])
                            if ((k1 == w).any() == True) & (shape(bb_gt_ref.get(k))[0] == 1):           # one BBgt
                                print k
                                for m in range(shape(bb_gt_ref.get(k))[1]):
                                    for n in range(shape(bb_dt_ref.get(k1)[a])[0]):
                                        if (bb_gt_ref.get(k)[0][m] == bb_dt_ref.get(k1)[a][n]).all() == True:
                                            common_pixels += 1
                                if common_pixels > 0:
                                    all_pixels = shape(bb_gt_ref.get(k))[1] + shape(bb_dt_ref.get(k1)[a])[0] - common_pixels
                                    match_value = common_pixels / float(all_pixels)
                                    if match_value > t:
                                        match += 1
                                        matched_gt = matched_gt.tolist()
                                        matched_gt.append(k)
                                        matched_gt = np.array(matched_gt)
                                        print 'match in loop 3!!'
                                        breaker += 1             # if there is a match, then there's no point in traversing the rest of the neighbouring grids
                                        break
                                    common_pixels = 0
                                if breaker > 0:
                                    breaker = 0
                                    break                    
                            elif ((k1 == w).any() == True) & (shape(bb_gt_ref.get(k))[0] > 1):           # multiple BBgt
                                # This section CAN BE OPTIMIZED --> (for each gt, if a matching dt is found, remove it from the array before next iteration)
                                for a in range(shape(bb_gt_ref.get(k))[0]):                              # for every gt in this grid
                                    for b in range(shape(bb_dt_ref.get(k1))[0]):                         # for every dt in this grid
                                        for m in range(shape(bb_gt_ref.get(k)[a])[0]):
                                            for n in range(shape(bb_dt_ref.get(k1)[b])[0]):
                                                if (bb_gt_ref.get(k)[a][m] == bb_dt_ref.get(k1)[b][n]).all() == True:
                                                    common_pixels += 1
                                        if common_pixels > 0:
                                            all_pixels = shape(bb_gt_ref.get(k)[a])[0] + shape(bb_dt_ref.get(k1)[b])[0] - common_pixels
                                            match_value = common_pixels / float(all_pixels)
                                            if match_value > t:
                                                match += 1
                                                matched_gt = matched_gt.tolist()
                                                matched_gt.append(k)
                                                matched_gt = np.array(matched_gt)
                                                print 'match in loop 4!!'
                                                breaker += 1
                                                break
                                        common_pixels = 0
                                if breaker ==  shape(bb_gt_ref.get(k))[0]:    # if all the BBgt found their matches, do not traverse rest of the neighbouring grids
                                    breaker = 0
                                    break 

print 'match = ', match
print 'FP = ', len(matches) - match
print 'FN = ', shape(bb_gt_pixels)[0] - match
    
