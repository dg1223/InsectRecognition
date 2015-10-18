##n_samples, n_features = X_train.shape
##half = int(n_samples / 2)

# Run classifier
classifier = SGDClassifier(loss = 'log', shuffle = True, learning_rate = 'optimal', alpha = 0.01, n_iter = 35, penalty = 'l1')
probas_ = classifier.fit(X_train, y_train).predict_proba(X_train)

# Compute Precision-Recall and plot curve
precision, recall, thresholds = precision_recall_curve(y_train, probas_)
area = auc(recall, precision)
print "Area Under Curve: %0.2f" % area

plt.clf()
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall example: AUC=%0.2f' % area)
plt.legend(loc="lower left")
plt.show()
