##from sklearn import preprocessing
##from sklearn.linear_model import SGDClassifier


#### Training Set


## Preprocessing

X_train = []

for i in range(len(feature_database_TrainingSet)):                   # len(feature_database_TrainingSet)
    X_train.append([])
    for keys, val in enumerate(feature_database_TrainingSet[i]):
        if val == 'intensity histogram':
            inty_list = feature_database_TrainingSet[i]['intensity histogram'].tolist()
            for j in range(len(inty_list)):
                X_train[i].append(inty_list[j])
##            continue
        X_train[i].append(feature_database_TrainingSet[i][val])
    X_train[i] = np.array(X_train[i])

count = 0
for i in range(len(feature_database_TrainingSet), (len(feature_database_TrainingSet) + len(feature_database_TrainingSet_nc))):                   # len(feature_database_TrainingSet)
    X_train.append([])
    count += 1
    if count > len(feature_database_TrainingSet_nc)+1:
        break
    else:
        for keys, val in enumerate(feature_database_TrainingSet_nc[count-1]):
            if val == 'intensity histogram':
                inty_list = feature_database_TrainingSet_nc[count-1]['intensity histogram'].tolist()
                for j in range(len(inty_list)):
                    X_train[i].append(inty_list[j])
##                continue
            X_train[i].append(feature_database_TrainingSet_nc[count-1][val])
        X_train[i] = np.array(X_train[i])
        
# convert all feature vectors to floats

for j in range(len(X_train)):
        X_train[j] = X_train[j].astype(float)

scaler = preprocessing.Scaler().fit(X_train)
X_train = scaler.transform(X_train)


## SGD Classifier

y_train = np.zeros(len(feature_database_TrainingSet) + len(feature_database_TrainingSet_nc))
y_train[0:len(feature_database_TrainingSet)] = 1

### convert all labels to ints
##for k in range(len(y_train)):
##    y_train[k] = np.int(y_train[k])

y_train = y_train.astype(int)

clf = SGDClassifier(loss = 'log', shuffle = True, learning_rate = 'optimal', alpha = 0.01, n_iter = 35, penalty = 'l1')
clf.fit(X_train, y_train)
