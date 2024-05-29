
import pathlib
import pandas as pd
import numpy as np
import mlflow

def terminal_gen(data):
   flag=0
   for pid in data.ProteinID.unique():
      # print(pid)
      len=data[data.ProteinID==pid].AminoAcid_1.count()
      evenly_spaced_numbers = np.linspace(-1, 1, len)
      if flag==0:
         evenly_spaced_numbers_merged = evenly_spaced_numbers
         flag=1
      else:
         evenly_spaced_numbers_merged = np.concatenate((evenly_spaced_numbers, evenly_spaced_numbers_merged))
   return evenly_spaced_numbers_merged
 
def readDataset(train_path,nox_test_path,pdb_test_path,validation,test,savedFeatures):     
   
   
   print("Selecting features")      
   columnNames=["DispredPred","DispredProba"]
   window_size=7
   Windowed_columnList=[]
   for columnName in columnNames:
      for t in range(window_size, 0,-1):               
               Windowed_columnList.append(f'{columnName}-{t}')
      for t in range(1, window_size+1):               
               Windowed_columnList.append(f'{columnName}+{t}')
                     
   ESM2StatsColumns=["ESM2650_mean", "ESM2650_std", "ESM2650_max", "ESM2650_min", "ESM2650_median", "ESM2650_skew", "ESM2650_kurtosis", "ESM2650_kurt", "ESM2650_var", "ESM2650_sem", "ESM2650_quantile"]
            
   window_size=7   
   ESM2StatsWindowed_columnList=[]
   for columnName in ESM2StatsColumns:
      for t in range(window_size, 0,-1):               
               ESM2StatsWindowed_columnList.append(f'{columnName}-{t}')
      for t in range(1, window_size+1):               
               ESM2StatsWindowed_columnList.append(f'{columnName}+{t}')                    
   ESM2ColumnName=[f'ESM2650_{i}' for i in range(1, 1281)]                  
   ESM1_StatsColumns=["ESM1_650_mean", "ESM1_650_std", "ESM1_650_max", "ESM1_650_min", "ESM1_650_median", "ESM1_650_skew", "ESM1_650_kurtosis", "ESM1_650_kurt", "ESM1_650_var", "ESM1_650_sem", "ESM1_650_quantile"]
   # ColumnName=["ProteinID",  "No_1", "AminoAcid_1",  "Target" ]+[f'fldpnn_{i}' for i in range(1, 318)]+[f'FLmodel_{i}' for i in range(1, 5125)]+[ "No_2", "AminoAcid_2", "DispredProba","DispredPred", "AminoAcid_proba" ]
   # SelectedColumns=[f'fldpnn_{i}' for i in range(1, 318)]+[f'FLmodel_{i}' for i in range(1, 5125)]+[  "DispredProba","DispredPred" ,"Terminal_posneg"]+Windowed_columnList+ESM2ColumnName+ESM2StatsColumns+ESM2StatsWindowed_columnList
   
   SelectedColumns_WithESM2=[f'fldpnn_{i}' for i in range(1, 318)]+[f'FLmodel_{i}' for i in range(1, 5125)]+[  "DispredProba","DispredPred" ,"Terminal_posneg"]+Windowed_columnList+ESM2ColumnName+ESM2StatsColumns+ESM2StatsWindowed_columnList
   SelectedColumns_WithoutESM2=[f'fldpnn_{i}' for i in range(1, 318)]+[f'FLmodel_{i}' for i in range(1, 5125)]+[  "DispredProba","DispredPred" ,"Terminal_posneg"]+Windowed_columnList
   SelectedColumns_ESM1EMS2=[f'FLmodel_{i}' for i in range(1, 5125)]+[  "DispredProba","DispredPred" ,"Terminal_posneg"]+Windowed_columnList+ESM2ColumnName+ESM2StatsColumns+ESM2StatsWindowed_columnList
   SelectedColumns_EMS2=[  "DispredProba","DispredPred" ,"Terminal_posneg"]+Windowed_columnList+ESM2ColumnName+ESM2StatsColumns+ESM2StatsWindowed_columnList
   SelectedColumns_EMS2WihoutDispredict3=[  "Terminal_posneg"]+ESM2ColumnName+ESM2StatsColumns+ESM2StatsWindowed_columnList
   
   # in production we mege the nox dataset
   ESMDispred=SelectedColumns_WithoutESM2
   ESM2Dispred=SelectedColumns_WithESM2
   
   # in production we mege the pdb dataset
   ESM2PDBDispred=SelectedColumns_EMS2
   
   SelectedColumns=ESM2PDBDispred
   print("Selected Columns length: ",len(SelectedColumns))
   

   if savedFeatures:
      print("Reading Modified Saved Data")
      X_train=pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/SavedModifiedData/X_train.feather")
      y_train=pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/SavedModifiedData/y_train.feather")
      X_nox_test= pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/SavedModifiedData/X_nox_test.feather")
      y_nox_test= pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/SavedModifiedData/y_nox_test.feather")
      X_pdb_test= pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/SavedModifiedData/X_pdb_test.feather")
      y_pdb_test=pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/SavedModifiedData/y_pdb_test.feather")
      
      print("Selecting Columns on saved data")
      X_train = X_train.loc[:, SelectedColumns]  
      X_nox_test = X_nox_test.loc[:, SelectedColumns]
      X_pdb_test = X_pdb_test.loc[:, SelectedColumns]
      
      #For PDB model
         # REmove mobidb missing values
      # print("Adding Mobidb Missing Values")
      mobidb=pd.read_csv("/home/SharedFiles/Wasi/MergedFeatures/DisPred/MobidbTrainingMissing.csv")   
      X_train["MobiDBmissing"]=mobidb["MobiDBmissing"] 
      X_train["Target"]=y_train
      print("Removing Mobidb Missing Values")  
      X_train=X_train[X_train["MobiDBmissing"]!=1]
      y_train= X_train.loc[:,"Target"  ]
      X_train.pop("MobiDBmissing")
      X_train.pop("Target")   
      if test:
         print("Test Mode. Using a fraction of the data.")
         X_train=X_train[:10000]
         y_train=y_train[:10000]
         X_nox_test=X_nox_test[:10000]
         y_nox_test=y_nox_test[:10000]
         X_pdb_test=X_pdb_test[:10000]
         y_pdb_test=y_pdb_test[:10000]
 
   else:
      
      print("Reading and Preprocessing Data")    
      trainData =pd.read_feather(train_path)
      nox_testData=pd.read_feather(nox_test_path)
      pdb_testData=pd.read_feather(pdb_test_path)
      
      if test:
         print("Test Mode. Using a fraction of the data.")
         trainData=trainData[:10000]
         nox_testData=nox_testData[:10000]
         pdb_testData=pdb_testData[:10000]        
         # SelectedColumns=[f'fldpnn_{i}' for i in range(1, 318)]+[f'FLmodel_{i}' for i in range(1, 5125)]

      else:   
         # Add windowed files
         print("Adding Windowed Files")
         window_size=7
         mlflow.log_param("window_size", window_size)
         train_windowed=pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/Windowing/TrainSet_Dispred_window_"+str(window_size)+".feather")
         pdb_testData_windowed=pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/Windowing/PDBSet_Dispred_window_"+str(window_size)+".feather")
         nox_testData_windowed=pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/Windowing/NoxSet_Dispred_window_"+str(window_size)+".feather")
         
         print("Train Data: ",trainData.shape)
         print("Nox Test Data: ",nox_testData.shape)
         print("PDB Test Data: ",pdb_testData.shape)
      
         #merge windowed files
         trainData=pd.concat([trainData,train_windowed], axis=1)
         pdb_testData=pd.concat([pdb_testData,pdb_testData_windowed], axis=1)
         nox_testData=pd.concat([nox_testData,nox_testData_windowed], axis=1)
         
         print("after window Train Data: ",trainData.shape)
         print("after window Nox Test Data: ",nox_testData.shape)
         print("after window PDB Test Data: ",pdb_testData.shape)
      
         # Relase memory
         del train_windowed
         del pdb_testData_windowed
         del nox_testData_windowed
         
         # Add ESM2 features
         print("Adding ESM2 Features")
         train_esm2 = pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/ESM2650M-TrainSET_complete.feather")
         pdb_testData_esm2= pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/ESM2650M-disorder_pdb_complete.feather")
         nox_testData_esm2 = pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/ESM2650M-disorder_NOX_complete.feather")

         train_esm2 = train_esm2.loc[:, ESM2ColumnName]
         pdb_testData_esm2 = pdb_testData_esm2.loc[:, ESM2ColumnName] 
         nox_testData_esm2 = nox_testData_esm2.loc[:, ESM2ColumnName]         

         trainData=pd.concat([trainData,train_esm2], axis=1)
         pdb_testData=pd.concat([pdb_testData,pdb_testData_esm2], axis=1)
         nox_testData=pd.concat([nox_testData,nox_testData_esm2], axis=1)
         # Relase memory
         del train_esm2
         del pdb_testData_esm2
         del nox_testData_esm2
         
         #Add ESM2 Stats
         print("Adding ESM2 Stats")
         trainstats_esm2 = pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/ESM2_650M_Train_stats.feather")
         pdb_stats_esm2= pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/ESM2_650M_pdb_stats.feather")
         nox_stats_esm2 = pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/ESM2_650M_nox_stats.feather")
      
         trainData=pd.concat([trainData,trainstats_esm2], axis=1)
         pdb_testData=pd.concat([pdb_testData,pdb_stats_esm2], axis=1)
         nox_testData=pd.concat([nox_testData,nox_stats_esm2], axis=1)
         
         # Relase memory
         del trainstats_esm2
         del pdb_stats_esm2
         del nox_stats_esm2
         
         print("Adding ESM2 stats Windowed Files")
         window_size=11
         mlflow.log_param("window_size_esm2Stats", window_size)
         train_esm2Statswindowed=pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/Windowing/TrainSet_esm2Stats_window_"+str(window_size)+".feather")
         pdb_testDataesm2Stats_windowed=pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/Windowing/PDBSet_esm2Stats_window_"+str(window_size)+".feather")
         nox_testDataesm2Stats_windowed=pd.read_feather("/home/SharedFiles/Wasi/MergedFeatures/DisPred/Windowing/NoxSet_esm2Stats_window_"+str(window_size)+".feather")

         # print("Train Data: ",trainData.shape)
         # print("Nox Test Data: ",nox_testData.shape)
         # print("PDB Test Data: ",pdb_testData.shape)
      
         #merge windowed files
         trainData=pd.concat([trainData,train_esm2Statswindowed], axis=1)
         pdb_testData=pd.concat([pdb_testData,pdb_testDataesm2Stats_windowed], axis=1)
         nox_testData=pd.concat([nox_testData,nox_testDataesm2Stats_windowed], axis=1)
         
         # print("after window Train Data: ",trainData.shape)
         # print("after window Nox Test Data: ",nox_testData.shape)
         # print("after window PDB Test Data: ",pdb_testData.shape)
         
         # Relase memory
         del train_esm2Statswindowed
         del pdb_testDataesm2Stats_windowed
         del nox_testDataesm2Stats_windowed
         
         #Add ESM1 Stats
         # print("Adding ESM1 Stats")
         # trainstats_esm1 = pd.read_feather("/home/mkabir3/Research/40_CAID3/6_MergeData/ESM1_650M_Train_stats.feather")
         # pdb_stats_esm1= pd.read_feather("/home/mkabir3/Research/40_CAID3/6_MergeData/ESM1_650M_pdb_stats.feather")
         # nox_stats_esm1 = pd.read_feather("/home/mkabir3/Research/40_CAID3/6_MergeData/ESM1_650M_nox_stats.feather")
      


         # trainData=pd.concat([trainData,trainstats_esm1], axis=1)
         # pdb_testData=pd.concat([pdb_testData,pdb_stats_esm1], axis=1)
         # nox_testData=pd.concat([nox_testData,nox_stats_esm1], axis=1)
         

         # Add Mobidb missing values
         # print("Adding Mobidb Missing Values")
         # mobidb=pd.read_csv("/home/mkabir3/Research/40_CAID3/9_MobiDB/MobidbTrainingMissing.csv")   
         # trainData["MobiDBmissing"]=mobidb["MobiDBmissing"]     
            
         # Add Amino acid probabilities ## No Improvements
         # # print("Adding Amino Acid Probabilities")
         # amino_acid_prob=pd.read_csv("/home/mkabir3/Research/40_CAID3/7_ExploratoryDataAnalysis/AminoAcid_disordered_Proba.csv")
         # amino_acid_prob["proportion"]=amino_acid_prob["proportion"]*10
         # trainData["AminoAcid_proba"]= trainData["AminoAcid_2"].replace(amino_acid_prob.set_index('AminoAcid')['proportion'].to_dict())
         # nox_testData["AminoAcid_proba"]= nox_testData["AminoAcid_2"].replace(amino_acid_prob.set_index('AminoAcid')['proportion'].to_dict())
         # pdb_testData["AminoAcid_proba"]= pdb_testData["AminoAcid_2"].replace(amino_acid_prob.set_index('AminoAcid')['proportion'].to_dict())

         # Remove large proteins
         print("Removing Large Proteins")
         listofLargeProteins = trainData.groupby('ProteinID').AminoAcid_1.count()[trainData.groupby('ProteinID').AminoAcid_1.count()>2000].index
         # Remove the proteins with more than 3000 amino acids
         trainData = trainData[~trainData.ProteinID.isin(listofLargeProteins)]
         print("Number of Unique Id Removed: ",len(listofLargeProteins))
         mlflow.log_metric("LargeProteinsRemoved", len(listofLargeProteins))
         
         # add Terminal values
         print("Adding Terminal Values")
         evenly_spaced_numbers_merged=terminal_gen(trainData)
         trainData["Terminal_posneg"]=evenly_spaced_numbers_merged
         trainData["Terminal_pos"]=np.abs(evenly_spaced_numbers_merged)

         evenly_spaced_numbers_merged=terminal_gen(nox_testData)
         nox_testData["Terminal_posneg"]=evenly_spaced_numbers_merged
         nox_testData["Terminal_pos"]=np.abs(evenly_spaced_numbers_merged)

         evenly_spaced_numbers_merged=terminal_gen(pdb_testData)
         pdb_testData["Terminal_posneg"]=evenly_spaced_numbers_merged
         pdb_testData["Terminal_pos"]=np.abs(evenly_spaced_numbers_merged)
         
         # REmove mobidb missing values
         # print("Removing Mobidb Missing Values")  
         # trainData=trainData[trainData["MobiDBmissing"]!=1]
         
         
         # Selecting targets
         print("Selecting Targets")
         # trainData["Target"]=trainData["Disorder_consensus"]
         trainData["Target"]=trainData["Disorder_consensus"]+trainData["Transition_consensus"]
      

         print("Selecting features") 
         X_train = trainData.loc[:, SelectedColumns]    
         y_train = trainData.loc[:,"Target"  ]
         print("X_train: ",X_train.shape)
         print("y_train: ",y_train.shape)

         X_nox_test = nox_testData.loc[:, SelectedColumns]
         y_nox_test = nox_testData.loc[:,"Target"  ]
         X_pdb_test = pdb_testData.loc[:, SelectedColumns]
         y_pdb_test = pdb_testData.loc[:,"Target"  ]
         print("X_nox_test: ",X_nox_test.shape)
         print("y_nox_test: ",y_nox_test.shape)

         print("X_pdb_test: ",X_pdb_test.shape)
         print("y_pdb_test: ",y_pdb_test.shape)
         X_pdb_test=X_pdb_test.iloc[np.where(y_pdb_test!="-")]
         y_pdb_test=y_pdb_test.iloc[np.where(y_pdb_test!="-")]
         y_pdb_test=y_pdb_test.astype(int)

         print("After Removing - X_pdb_test: ",X_pdb_test.shape)
         print("After Removing - y_pdb_test: ",y_pdb_test.shape)

         mlflow.log_metric("traindata_UniqueProteins", trainData.ProteinID.nunique())
         mlflow.log_metric("nox_testData_UniqueProteins", nox_testData.ProteinID.nunique())
         mlflow.log_metric("pdb_testData_UniqueProteins", pdb_testData.ProteinID.nunique())

         # Relase memory
         del trainData
         del nox_testData
         del pdb_testData
   
         # if validation:
         #     # Create validation set
         #     validationData = trainData.sample(frac=0.2, random_state=1)
         #     trainData = trainData.drop(validationData.index)
         #     trainData.reset_index(drop=True, inplace=True)
         #     validationData.reset_index(drop=True, inplace=True)
         #     print("Validation Data: ",validationData.shape)
         #     print("Train Data: ",trainData.shape)
         #     y_valid = validationData.loc[:,"Target"  ]
         #     X_valid = validationData.loc[:, SelectedColumns]
         #     print("X_valid: ",X_valid.shape)
         #     print("y_valid: ",y_valid.shape)
         #     return X_train, y_train, X_nox_test, y_nox_test, X_pdb_test, y_pdb_test, X_valid, y_valid

         # # convert pandas dataframe to numpy array
         # train = trainData.to_numpy()
         # np.random.shuffle(train)
         # y_train = train[:,-1]
         # X_train = train[:,:-1]

         # test = testData.to_numpy()
         # y_test = test[:,-1]
         # X_test = test[:,:-1]

         # if validation: 
         #     valid = validationData.to_numpy()
         #     y_valid = valid[:,-1]
         #     X_valid = valid[:,:-1]

         # Save all into a file     

         if test==False:
            print("Saving Modified Data")
            pathlib.Path("./SavedModifiedData").mkdir(parents=True, exist_ok=True) 
            X_train.to_feather("./SavedModifiedData/X_train.feather")
            pd.DataFrame(y_train).to_feather("./SavedModifiedData/y_train.feather")
            X_nox_test.to_feather("./SavedModifiedData/X_nox_test.feather")
            pd.DataFrame(y_nox_test).to_feather("./SavedModifiedData/y_nox_test.feather")
            X_pdb_test.to_feather("./SavedModifiedData/X_pdb_test.feather")
            pd.DataFrame(y_pdb_test).to_feather("./SavedModifiedData/y_pdb_test.feather")      

  
   print("Logging Parameters")
   mlflow.log_param("TestRun", test)
   mlflow.log_param("UsingsavedFeatures", savedFeatures)
   mlflow.log_param("train_path", train_path)
   mlflow.log_param("nox_test_path", nox_test_path)
   mlflow.log_param("pdb_test_path", pdb_test_path)
   mlflow.log_metric("traindata_NumberofFeatures", X_train.shape[1])
   mlflow.log_metric("nox_testData_NumberofFeatures", X_nox_test.shape[1])
   mlflow.log_metric("pdb_testData_NumberofFeatures", X_pdb_test.shape[1])
      
   return X_train, y_train, X_nox_test, y_nox_test, X_pdb_test, y_pdb_test