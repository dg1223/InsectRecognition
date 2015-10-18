miss_rate = np.array([0.92, 0.83, 0.68, 0.58, 0.50, 0.56, 0.64, 0.78, 0.87, 0.90])          # sample: 0.5 ov
FPPI = np.array([23.41, 34.19, 36.10, 36.57, 37.45, 43.75, 48.03, 40.74, 33.32, 22.36])
a = np.logspace(-2, 1, num = 10, base = 10)

points = []

##for i in range(len(FPPI)):
##        points.append((miss_rate[i] * 9)/float(FPPI[i])) # a[i]), now 10 FPPI
##
##points = np.array(points)

for i in range(len(FPPI)):
    points.append([])
##    points = np.array(points)
    for j in range(len(a)):
        points[i].append((miss_rate[i] * a[j])/float(FPPI[i]))

points = np.array(points)

miss_vs_FPPI = []

for i in range(len(a)):
    miss_vs_FPPI.append([])
    for j in range(len(a)):
        miss_vs_FPPI[i].append(points[j,i])

miss_vs_FPPI = np.array(miss_vs_FPPI)

avg_miss = []

for i in range(len(a)):
    avg_miss.append(np.mean(miss_vs_FPPI[i]))

avg_miss = np.array(avg_miss)


#############################################################################################################################

MRPI1  = np.array([.331, .368, .471, .574, .688, .808, .839, .892, .951, 1])
FPPI1  = np.array([46.71, 34.76, 27.87, 21.81, 17.08, 13.75, 10.76, 7.56, 4.76, 0.02])

MRPI2  = np.array([.335, .377, .474, .577, .691, .810, .842, .893, .952, 1])
FPPI2  = np.array([46.79, 36.24, 27.90, 21.84, 17.11, 13.78, 10.79, 7.57, 4.78, 0.02])

MRPI3  = np.array([.282, .383, .503, .606, .693, .812, .843, .893, .952, 1])
FPPI3  = np.array([46.89, 36.33, 27.98, 21.92, 17.16, 13.83, 10.83, 7.57, 4.78, 0.02])

MRPI4  = np.array([.339, .416, .550, .649, .733, .851, .880, .929, .954, 1])
FPPI4  = np.array([47.43, 36.78, 27.41, 22.22, 17.41, 14.05, 10.87, 7.75, 4.79, 0.02])

MRPI5  = np.array([.378, .447, .575, .670, .746, .859, .885, .933, .955, 1])
FPPI5  = np.array([48.27, 37.29, 28.90, 22.70, 17.73, 14.25, 11.17, 7.86, 4.84, 0.02])

MRPI6  = np.array([.589, .617, .680, .769, .820, .909, .931, .972, .988, 1])
FPPI6  = np.array([49.86, 38.76, 29.97, 23.60, 18.49, 14.86, 11.71, 8.27, 5.13, 0.02])

MRPI7  = np.array([.798, .817, .875, .900, .923, .971, .983, .987, .995, 1])
FPPI7  = np.array([51.63, 40.41, 31.40, 24.43, 19.44, 15.63, 12.27, 8.60, 5.25, 0.02])

MRPI8  = np.array([.971, .982, .986, .989, .992, .993, .997, .998, .999, 1])
FPPI8  = np.array([52.97, 41.60, 32.43, 25.54, 20.10, 16.10, 12.60, 8.86, 5.33, 0.02])

MRPI9  = np.array([.999, .999, 1, 1 ,1, 1, 1, 1, 1, 1])
FPPI9  = np.array([53.27, 41.89, 32.65, 25.71, 20.52, 16.29, 12.65, 8.89, 5.35, 0.02])

MRPI10 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
FPPI10  = np.array([53.29, 41.90, 32.65, 25.71, 20.52, 16.29, 12.65, 8.89, 5.35, 0.02])

line2d = plt.loglog(FPPI1, MRPI1, label = 'ov_thresh = 0.1')
xvalues = line2d[0].get_xdata()
yvalues = line2d[0].get_ydata()
low_lim = 9
high_lim = 11
idx = np.where((xvalues > low_lim) & (xvalues < high_lim))
while np.size(idx[0][0]) == 0:
	idx = np.where((xvalues > low_lim) & (xvalues < (high_lim+1)))
	high_lim += 1


plt.ylim(1e-1, 1.1)
plt.xlim(0.02, 100)
plt.legend(loc=3)
leg = plt.gca().get_legend()
ltext  = leg.get_texts()
plt.setp(ltext, fontsize='small')

xvalues = line2d[0].get_xdata()
yvalues = line2d[0].get_ydata()



avg_miss = np.array([.59, .7, .7, .73, .74, .83, .92, .99, .99, 1])
ov_thresh = np.array([.1, .2, .3, .4, .5, .6, .7 ,.8, .9, 1.0])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
line = ax.plot(ov_thresh, avg_miss)
ax.set_yscale('log')
show()
plt.ylim(0.00, 1.00)





