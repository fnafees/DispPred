from lightgbm import early_stopping, log_evaluation, record_evaluation
from sklearn.metrics import *
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

def trainModel(X_train, y_train, X_nox_test,  X_pdb_test,y_nox_test,y_pdb_test, seed_value, n_estimators, eval_metric, Validation,production):

    if production:
        
        # ESMDispred and ESM2Dispred
        # X_train=np.concatenate((X_train,X_nox_test),axis=0)
        # y_train=np.concatenate((y_train,y_nox_test),axis=0)
        
        # ESM2PDBDispred
        X_train=np.concatenate((X_train,X_pdb_test),axis=0)
        print(y_train.shape)
        print(y_pdb_test.shape)
        
        y_train=np.array(y_train)
        y_train=y_train.reshape(-1,1)
        y_train=np.concatenate((y_train,y_pdb_test),axis=0)
  
    
    StandardSc=False
    
    if StandardSc:
        clf =LGBMClassifier(random_state =seed_value, n_estimators= n_estimators, num_threads=64, verbose=0 ) 
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    else:
        clf =LGBMClassifier(random_state =seed_value, n_estimators= n_estimators, num_threads=64, verbose=0 ) 
        pipe = Pipeline([('clf', clf)])
    if Validation:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=seed_value)
    
        if eval_metric=="f1":
            clf.fit(X_train,y_train,eval_set=[(X_valid, y_valid)],eval_metric=lgb_f1_score )
        else:
            clf.fit(X_train,y_train,eval_set=[(X_valid, y_valid)],eval_metric=eval_metric )
    else:
        X_valid=None
        y_valid=None
        if eval_metric=="f1":
            pipe.fit(X_train,y_train,clf__eval_metric=lgb_f1_score )
        else:
            pipe.fit(X_train,y_train,clf__eval_metric=eval_metric )
            
    y_proba_NOX=pipe.predict_proba(X_nox_test)
    y_proba_NOX=y_proba_NOX[:,1]
    y_proba_PDB=pipe.predict_proba(X_pdb_test)
    y_proba_PDB=y_proba_PDB[:,1]
     
        
    return pipe, y_proba_NOX,y_proba_PDB, X_valid, y_valid
    


# def CVResults(X_train, y_train, X_nox_test,  X_pdb_test,y_pdb_test, seed_value, n_estimators,eval_metric,Validation):

#     StandardSc=False
    
#     if StandardSc:
#         clf =LGBMClassifier(random_state =seed_value, n_estimators= n_estimators, num_threads=64, verbose=0 ) 
#         pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
#     else:
#         clf =LGBMClassifier(random_state =seed_value, n_estimators= n_estimators, num_threads=64, verbose=0 ) 
#         pipe = Pipeline([('clf', clf)])
#     if Validation:
#         X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=seed_value)
    
#         if eval_metric=="f1":
#             # crossvaliation
            
#             clf.fit(X_train,y_train,eval_set=[(X_valid, y_valid)],eval_metric=lgb_f1_score )
#         else:
#             clf.fit(X_train,y_train,eval_set=[(X_valid, y_valid)],eval_metric=eval_metric )
#     else:
#         X_valid=None
#         y_valid=None
#         if eval_metric=="f1":
#             pipe.fit(X_train,y_train,clf__eval_metric=lgb_f1_score )
#         else:
#             pipe.fit(X_train,y_train,clf__eval_metric=eval_metric )          
 
     
        














# train ML model
def lgb_modelfit_nocv(params, dtrain, dvalid, objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):


    # lgb_params.update(params)

    evals_results = {}

    bst1 = lgb.train(params, 
                     dtrain, 
                     valid_sets=[dtrain, dvalid], 
                     valid_names=['train','valid'], 
                    #  evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                    #  early_stopping_rounds=early_stopping_rounds,
                     callbacks=[early_stopping(early_stopping_rounds),log_evaluation(verbose_eval),record_evaluation(evals_results)],
                     feval=feval)

    n_estimators = bst1.best_iteration
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])

    return bst1
def lgb_f1_score(preds, labels):
    labels = (labels >= 0.5).astype(int)
    return 'f1', f1_score(labels, preds), True





def trainModelwithparameters(X_train, y_train, X_test, y_test, X_valid, y_valid, seed_value, n_estimators,eval_metric):


    d_train = lgb.Dataset(X_train, label=y_train)
    d_valid = lgb.Dataset(X_valid, label=y_valid)

    # params0 = {
    #     'boosting_type': 'gbdt',
    #     'objective': objective,
    #     'metric':metrics,
    #     'learning_rate': 0.01,
    #     #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
    #     'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
    #     'max_depth': -1,  # -1 means no limit
    #     'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
    #     'max_bin': 255,  # Number of bucketed bin for feature values
    #     'subsample': 0.6,  # Subsample ratio of the training instance.
    #     'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
    #     'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
    #     'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    #     'subsample_for_bin': 200000,  # Number of samples for constructing bin
    #     'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    #     'reg_alpha': 0,  # L1 regularization term on weights
    #     'reg_lambda': 0,  # L2 regularization term on weights
    #     'nthread': 64,
    #     'verbose': 0,
    #     'metric':metrics
    # }

    params1 = {
        'learning_rate': 0.15,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':99 # because training data is extremely unbalanced 
    }

    params2 = {
 
	# 'n_estimators': 1000,
	'boosting_type': 'gbdt',
	'objective': 'binary',
	'learning_rate': 0.15,
	'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
	'num_leaves': 7,  # 2^max_depth - 1
	'max_depth': 3,  # -1 means no limit
	'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
	'max_bin': 100,  # Number of bucketed bin for feature values
	'subsample': 0.7,  # Subsample ratio of the training instance.
	'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
	'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
	# 'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
	
	# 'subsample_for_bin': 200000,  # Number of samples for constructing bin
	# 'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
	# 'reg_alpha': 0,  # L1 regularization term on weights
	# 'reg_lambda': 0,  # L2 regularization term on weights
	# 'nthread': 4,
	'verbose': 0,
	'metric':'auc',
	# 'scale_pos_weight':99 # because training data is extremely unbalanced 
    }

    
    lgb_params = params2
    clf = lgb_modelfit_nocv(lgb_params, 
                        d_train,
                        d_valid ,
                        objective='binary', 
                        metrics='auc',
                        early_stopping_rounds=1000, 
                        verbose_eval=False, 
                        num_boost_round=n_estimators)  
    y_proba=clf.predict(X_test)   
    return clf, y_proba