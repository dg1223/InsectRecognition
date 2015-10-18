##from sklearn import preprocessing
##from sklearn.linear_model import SGDClassifier

## Preprocessing

X_test = []

for i in range(len(feature_database_TestSet)):                   # len(feature_database_TestSet)
    X_test.append([])
    for keys, val in enumerate(feature_database_TestSet[i]):
        if val == 'intensity histogram':
            inty_list = feature_database_TestSet[i]['intensity histogram'].tolist()
            for j in range(len(inty_list)):
                X_test[i].append(inty_list[j])
            continue
        X_test[i].append(feature_database_TestSet[i][val])

# convert all feature vectors to floats

for j in range(len(X_test)):
    for i in range(len(X_test[j])):
        X_test[j][i] = float(X_test[j][i])

X_test = scaler.transform(X_test)


X_test_nc = []

for i in range(len(feature_database_TestSet_nc)):                   # len(feature_database_TestSet)
    X_test_nc.append([])
    for keys, val in enumerate(feature_database_TestSet_nc[i]):
        if val == 'intensity histogram':
            inty_list = feature_database_TestSet_nc[i]['intensity histogram'].tolist()
            for j in range(len(inty_list)):
                X_test_nc[i].append(inty_list[j])
            continue
        X_test_nc[i].append(feature_database_TestSet_nc[i][val])

# convert all feature vectors to floats

for j in range(len(X_test_nc)):
    for i in range(len(X_test_nc[j])):
        X_test_nc[j][i] = float(X_test_nc[j][i])

X_test_nc = scaler.transform(X_test_nc)

a = clf.predict(X_test)
b = [i for i in a if i > 0]
match = len(b)

a1 = clf.predict(X_test_nc)
b1 = [i for i in a1 if i > 0]
match1 = len(b1)

print 'match = ', match, '        match1 = ', match1
