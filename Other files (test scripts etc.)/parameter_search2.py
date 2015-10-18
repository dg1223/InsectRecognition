pipeline = Pipeline([('clf', SGDClassifier())])

parameters = {
# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way

    'clf__alpha': (0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001),
    'clf__penalty': ('l1', 'l2', 'elasticnet'),
    'clf__n_iter': (5, 10, 15, 20, 35, 40, 45, 50, 55, 60, 80),
##    'clf_learning_rate': ('optimal', 'constant', 'invscaling'),
##    'clf_eta0': (0.1, 0.01, 0.001),
##    'clf_power_t': (0.1, 0.5, 0.9)
    
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, cv=KFold(len(y_train), 6), n_jobs=-1, verbose=1)

    print "Performing grid search..."
    print "pipeline:", [name for name, _ in pipeline.steps]
    print "parameters:"
    pprint(parameters)
##    t0 = time()
    grid_search.fit(X_train, y_train)
##    print "done in %0.3fs" % (time() - t0)
    print

    print "Best score: %0.3f" % grid_search.best_score_
    print "Best parameters set:"
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print "\t%s: %r" % (param_name, best_parameters[param_name])
