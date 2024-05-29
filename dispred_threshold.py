# Threshold Optimzaiton optimization on Validation Set
def threshold_optimization(run_name,clf,X_valid,X_train,seed_value, y_proba, validation, y_train, y_test,  y_valid,data_read_time,training_time):
 
    print("\n","*"*20,"Threshold Optimzaiton optimization on Validation Set ","*"*20, "\n")
    # predict probabilities
    yhat = clf.predict(X_valid)
    # yhat = clf.predict_proba(X_valid)  # this does not work for LGBMClassifier
    # keep probabilities for the positive outcome only
    # yhat = yhat[:, 1]
    
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(y_valid, yhat)
    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    # print('ROC GMeans, Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

    # get the best threshold
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    # print('ROC, Best Threshold=%f' % (best_thresh))

    # # Threshold Optimzaiton based on PRAUC curve
    # # calculate roc curves
    # precision, recall, thresholds = precision_recall_curve(y_valid, yhat)
    # # convert to f score
    # fscore = (2 * precision * recall) / (precision + recall)
    # # locate the index of the largest f score
    # ix = np.argmax(fscore)
    # print('PRAUC, Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
    # # y_pred = (y_proba1 >= threshold).astype(int)

    # # define thresholds
    # thresholds = np.arange(0, 1, 0.001)
    # # evaluate each threshold
    # scores = [f1_score(y_valid, (yhat >= t).astype(int)) for t in thresholds]
    # # get best threshold
    # ix = np.argmax(scores)
    # print('f1_score, Best Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))

    # # define thresholds
    # thresholds = np.arange(0, 1, 0.001)
    # # evaluate each threshold
    # scores = [matthews_corrcoef(y_valid, (yhat >= t).astype(int)) for t in thresholds]
    # # get best threshold
    # ix = np.argmax(scores)
    # print('MCC, Best Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))

    # # define thresholds
    # thresholds = np.arange(0, 1, 0.001)
    # # evaluate each threshold
    # scores = [cohen_kappa_score(y_valid, (yhat >= t).astype(int)) for t in thresholds]
    # # get best threshold
    # ix = np.argmax(scores)
    # print('kappa, Best Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
    # print("\n","*"*20,"Threshold Optimzaiton optimization on Validation ","*"*20, "\n")
    best_thresh_valid=best_thresh
    print('ROC, Best Threshold for validation=%f' % (best_thresh_valid))
    run_name3=run_name+"OptThreshValidation_"+str(best_thresh_valid)
    mlflow.start_run(run_name=run_name3)
    classificationScore(run_name3,seed_value, y_test,y_proba,validation, y_train, y_test,  y_valid,data_read_time,training_time,threshold=best_thresh_valid)
    mlflow.end_run()

    # Threshold Optimzaiton optimization on Training Set


    print("\n","*"*20,"Threshold Optimzaiton optimization on Training Set","*"*20, "\n")
    # Threshold Optimzaiton based on ROC curve    
    # predict probabilities
    yhat = clf.predict(X_train)
    # yhat = clf.predict_proba(X_train)  # this does not work for LGBMClassifier
    # keep probabilities for the positive outcome only
    # yhat = yhat[:, 1]
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(y_train, yhat)
    # calculate the g-mean for each threshold
    gmeans = np.sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    # print('ROC gmeans, Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

    # get the best threshold
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    print('ROC, Best  Threshold=%f' % (best_thresh))

    # # Threshold Optimzaiton based on PRAUC curve
    # # calculate roc curves
    # precision, recall, thresholds = precision_recall_curve(y_train, yhat)
    # # convert to f score
    # fscore = (2 * precision * recall) / (precision + recall)
    # # locate the index of the largest f score
    # ix = np.argmax(fscore)
    # print('PRAUC, Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))


    # # define thresholds
    # thresholds = np.arange(0, 1, 0.001)
    # # evaluate each threshold
    # scores = [f1_score(y_train, (yhat >= t).astype(int)) for t in thresholds]
    # # get best threshold
    # ix = np.argmax(scores)
    # print('f1_score, Best Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))

    # # define thresholds
    # thresholds = np.arange(0, 1, 0.001)
    # # evaluate each threshold
    # scores = [matthews_corrcoef(y_train, (yhat >= t).astype(int)) for t in thresholds]
    # # get best threshold
    # ix = np.argmax(scores)
    # print('MCC, Best Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))

    # # define thresholds
    # thresholds = np.arange(0, 1, 0.001)
    # # evaluate each threshold
    # scores = [cohen_kappa_score(y_train, (yhat >= t).astype(int)) for t in thresholds]
    # # get best threshold
    # ix = np.argmax(scores)
    # print('kappa, Best Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))



    best_thresh_train=best_thresh
    print('ROC - Best Threshold training=%f' % (best_thresh_train))
    run_name3=run_name+"OptThreshTrain_"+str(best_thresh_train)
    
    mlflow.start_run(run_name=run_name3)
    classificationScore(run_name3,seed_value,y_test,y_proba,validation, y_train, y_test,  y_valid,data_read_time,training_time,threshold=best_thresh_train)
    mlflow.end_run()


    # Threshold Optimzaiton optimization on Validation and Training Set
    # print("########################################")
    print("\n","*"*20,"Threshold Optimzaiton optimization on Validation and Training Set","*"*20, "\n")
    best_thresh_avg=(best_thresh_train+best_thresh_valid)/2
    print('Best Threshold average=%f' % (best_thresh_avg))
    run_name3=run_name+"OptThreshtrain+validation_"+str(best_thresh_avg)
    # print(run_name3)
    mlflow.start_run(run_name=run_name3)
    classificationScore(run_name3,seed_value, y_test,y_proba,validation, y_train, y_test,  y_valid,data_read_time,training_time,threshold=best_thresh_avg)
    mlflow.end_run()

