all_FP = []
totalFP = 0
for j in range(len(total_FPs[1])):
    for i in range(1,len(total_FPs)-1):
        totalFP +=  total_FPs[i][j]
    all_FP.append(totalFP)
    totalFP = 0
all_FP = np.array(all_FP)

all_FN = []
totalFN = 0
for j in range(len(total_FNs[1])):
    for i in range(1,len(total_FNs)-1):
        totalFN +=  total_FNs[i][j]
    all_FN.append(totalFN)
    totalFN = 0
all_FN = np.array(all_FN)


all_the_miss = []
totalmiss = 0
for j in range(len(miss_rates[1])):
    for i in range(1,len(miss_rates)-1):
        totalmiss +=  miss_rates[i][j]
    all_the_miss.append(totalmiss)
    totalmiss = 0
all_the_miss = np.array(all_the_miss)


