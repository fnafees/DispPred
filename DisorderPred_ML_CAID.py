# %%
# Author : Md Wasi Ul Kabir  

# Importing Libraries
# 
# from lightgbm import LGBMClassifier
# import lightgbm as lgb
# from sklearn.metrics import *
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# import matplotlib.pyplot as pyplot
# from mlflow import log_metric, log_param, log_artifacts
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
# 
# import seaborn as sns
# 

import joblib
import numpy as np
import random 
import os
import time
import pathlib
from optparse import OptionParser
import sys
import mlflow
import pandas as pd

from dispred_results import classificationScore
from dispred_ML import trainModel
# from dispred_plot import *
# from dispred_threshold import *
from dispred_Data import readDataset
from  dispred_featImportance import shapImportance

import warnings
warnings.filterwarnings("ignore")
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
np.set_printoptions(precision=3)


# Set a seed value to reporduce the results
seed_value= 2515  
os.environ['PYTHONHASHSEED']=str(seed_value) 
random.seed(seed_value) 
np.random.seed(seed_value) 

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

# %%
def checkifpathexists(path):
    path = pathlib.Path(path)
    if path.exists():
        # print("Path exist")
        pass
    else:
        print("The following Path does not exist. Please check the path and try again.")
        print(path)
        sys.exit()
        
# read dataset from file



# %%
# def main():
if __name__ == "__main__": 

    parent_path = str(pathlib.Path(__file__).resolve().parents[1])
    print("Parent Dir",parent_path)

    parser = OptionParser()
    parser.add_option("-r", "--run_name", dest="run_name", help="run_name", default="FldpnnFeatures_LightGBM_estimator_")
    parser.add_option("-p", "--test", dest="test", help="test", default="True")
    parser.add_option("-e", "--n_estimators", dest="n_estimators", help="n_estimators", default=1)
    parser.add_option("-n", "--savedFeatures", dest="savedFeatures", help="savedFeatures", default="True")

    parser.add_option("-l", "--train_Dir_path", dest="train_Dir_path", help="Path to shared Directory." , default="")
    parser.add_option("-t", "--nox_test_path", dest="nox_test_path", help="Path to test dataset.", default="" )
    parser.add_option("-s", "--pdb_test_path", dest="pdb_test_path", help="Path to test dataset", default="" )
    
    # parser.add_option("-m", "--eval_metric", dest="eval_metric", help="eval_metric", default="auc")
    # parser.add_option("-v", "--dataset3_features_path", dest="dataset3_features_path", help="Path to dataset3 language features." , default="")
    
    # parser.add_option("-s", "--test_features_path", dest="test_features_path", help="Path to test language features." )
    # parser.add_option("-v", "--validation_features_path", dest="validation_features_path", help="Path to validation language features.")
    

    (options, args) = parser.parse_args()
    print("\n","#"*40,"Starting the ML Pipeline","#"*40, "\n")
    

 
    
    
    # #set number of estimators
    n_estimators=int(options.n_estimators)
    print("Number of Estimators:",options.n_estimators)
    
    train_path=options.train_Dir_path
    checkifpathexists(train_path)
    print("Train Path:",options.train_Dir_path)
    
    nox_test_path=options.nox_test_path
    checkifpathexists(nox_test_path)
    print("Nox Test Path:",options.nox_test_path)
    
    pdb_test_path=options.pdb_test_path
    checkifpathexists(pdb_test_path)
    print("PDB Test Path:",options.pdb_test_path)
    

    # # Set the run name
    run_name=options.run_name
    if run_name=="":
        run_name="TestRun"
    print("Run Name:",options.run_name)

   # Check if this is test run
    test=(options.test == 'True')
    print("Test Run Parameter:",options.test)
    if test:
        print("\n","#"*40,"This is a Test Run","#"*40, "\n")        
        run_name="_Test"+run_name
        n_estimators=1
      
    # check if the saved features are to be used
    savedFeatures=(options.savedFeatures == 'True');
    print("Saved Features Parameter:",options.savedFeatures)
      
    # Make validation true if you want to validate the model
    validation= False

    # set the run name for mlflow
    Experiment_name="Evaluate_CAID3"
    experiment = mlflow.set_experiment(Experiment_name)
    print("Experiment_id: {}".format(experiment.experiment_id))
    mlflow.start_run(run_name=run_name)
    
    # read dataset
    print("\n","#"*40,"Reading Dataset","#"*40, "\n")
  
    # print("Data Shapes:")
    start = time.time()
    X_train, y_train, X_nox_test, y_nox_test, X_pdb_test, y_pdb_test = readDataset(train_path,nox_test_path,pdb_test_path,validation,test,savedFeatures)
    end = time.time()
    data_read_time=(end - start)/60
    eval_metric = "auc"  # Define the variable eval_metric

    print("Data Read Time in minutes",data_read_time)
           
           
    # train the model with all features
    print("\n","#"*40,"Training Model with all features","#"*40, "\n")

    start = time.time()
    production=True
    clf, y_proba_NOX,y_proba_PDB, X_valid, y_valid= trainModel(X_train, y_train, X_nox_test,  X_pdb_test,y_nox_test,y_pdb_test, seed_value, n_estimators, "auc", validation,production)
    mlflow.log_param("production", production)   
    end = time.time()
    training_time=(end - start)/60    
    print("Model Training Time in minutes",data_read_time)
    
    
    #create output directory
    pathlib.Path("./output_allfeatures").mkdir(parents=True, exist_ok=True) 
    pathlib.Path("./output_allfeatures/"+run_name+"/modelsandproba").mkdir(parents=True, exist_ok=True) 
    pathlib.Path("./output_allfeatures/"+run_name+"/images").mkdir(parents=True, exist_ok=True) 
    # save the model and y_proba in csv file
    joblib.dump(clf,"./output_allfeatures/"+run_name+"/modelsandproba/model.pkl")
    pd.DataFrame(y_proba_NOX).to_csv("./output_allfeatures/"+run_name+"/modelsandproba/y_proba_NOX.csv")
    pd.DataFrame(y_proba_PDB).to_csv("./output_allfeatures/"+run_name+"/modelsandproba/y_proba_PDB.csv")
    feature_list= X_train.columns.tolist()
    pd.DataFrame(feature_list).to_csv("./output_allfeatures/"+run_name+"/modelsandproba/feature_list.csv")
    
    print("\n","#"*40,"Classification Score for Dispredict with all features","#"*40, "\n")

    
    classificationScore(run_name,seed_value, y_nox_test,y_pdb_test,y_proba_NOX,y_proba_PDB,validation, data_read_time,training_time,y_train,y_valid,0.5)
    mlflow.sklearn.log_model(clf, "model_allfeatures")
    mlflow.log_params(clf.get_params())

    # Train model with parameters
    # mlflow.log_params(clf.params)
    mlflow.end_run()

    
    
    # # # Feature Importance
    # print("\n","#"*40,"Feature Importance","#"*40, "\n")
    # feature_list = shapImportance(clf, X_train, y_train,run_name)
    
    # X_train = X_train[feature_list]
    # if validation:
    #     X_valid = X_valid[feature_list]
    # X_nox_test = X_nox_test[feature_list]
    # X_pdb_test = X_pdb_test[feature_list]
    
    
    # # train the model
    # print("\n","#"*40,"Training Model after feature Selection","#"*40, "\n")

    # start = time.time()
    # clf, y_proba_NOX,y_proba_PDB, X_valid, y_valid= trainModel(X_train, y_train, X_nox_test,  X_pdb_test,y_pdb_test, seed_value, n_estimators, "auc", validation)
        
    # end = time.time()
    # training_time=(end - start)/60    
    # print("Model Training Time in minutes",data_read_time)

    # #create output directory
    # pathlib.Path("./output_selectedfeat").mkdir(parents=True, exist_ok=True) 
    # pathlib.Path("./output_selectedfeat/"+run_name+"/modelsandproba").mkdir(parents=True, exist_ok=True) 
    # pathlib.Path("./output_selectedfeat/"+run_name+"/images").mkdir(parents=True, exist_ok=True) 
    # # save the model and y_proba in csv file
    # joblib.dump(clf,"./output_selectedfeat/"+run_name+"/modelsandproba/model.pkl")
    # pd.DataFrame(y_proba_NOX).to_csv("./output_selectedfeat/"+run_name+"/modelsandproba/y_proba_NOX.csv")
    # pd.DataFrame(y_proba_PDB).to_csv("./output_selectedfeat/"+run_name+"/modelsandproba/y_proba_PDB.csv")
    # pd.DataFrame(feature_list).to_csv("./output_selectedfeat/"+run_name+"/modelsandproba/feature_list.csv")
    
    # print("\n","#"*40,"Classification Score for Dispredict after feature Selection","#"*40, "\n")

    
    # classificationScore(run_name,seed_value, y_nox_test,y_pdb_test,y_proba_NOX,y_proba_PDB,validation, data_read_time,training_time,y_train,y_valid,0.5)
    # mlflow.sklearn.log_model(clf, "model_selectedfeatures")
    # mlflow.log_params(clf.get_params())

    # # Train model with parameters
    # # mlflow.log_params(clf.params)
    # mlflow.end_run()

    # # #  threshold optimization
    # print("\n","#"*40,"Threshold optimization","#"*40, "\n")
 
    # start = time.time()
    # threshold_optimization(run_name,clf,X_valid,X_train,seed_value, y_proba, validation, y_train, y_test,  y_valid,data_read_time,training_time)
    # end = time.time()
    # optimization_time=(end - start)/60
    # print("Threshold optimization Time in minutes",optimization_time)
    
    # print("\n","#"*40,"Plotting the results","#"*40, "\n")
    # # plot the roc curve
    # plot_roc_auc(y_test,y_proba,y_probaFl,run_name,sharedDir)

    # # plot the precision recall curve
    # plot_precisionrecall(y_test,y_proba,y_probaFl,run_name,sharedDir)

    # plot the Comparisons
    # plot_comparisons(y_test,run_name)





