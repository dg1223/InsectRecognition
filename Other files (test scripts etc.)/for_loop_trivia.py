for i in range(b):
    print 'i = ', i
    for j in range(a):
        if j ==3:
            if i == 1:
                break
            else:
                print j
        else:
            print j
    if i == 1:
        break
