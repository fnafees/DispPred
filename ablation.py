import subprocess

n_estimators = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100,1500,2000]

# n_estimators = [1, 2]

print("\n","#"*40,"Starting Experiment","#"*40, "\n")
print("\n","#"*40,"Ablation Analysis","#"*40, "\n")

# ../../.venv/bin/python DisorderPred_ML_CAID.py \
#                                     --run_name $exp_name  \
#                                     --test "False" \
#                                     --n_estimators  100 \
#                                     --savedFeatures "True" \
#                                     --train_Dir_path /home/mkabir3/Research/40_CAID3/6_MergeData/disorder_disprot23-CAID2-TrainSET_complete.feather \
#                                     --nox_test_path /home/mkabir3/Research/40_CAID3/6_MergeData/disorder_nox_complete.feather \
#                                     --pdb_test_path /home/mkabir3/Research/40_CAID3/6_MergeData/disorder_pdb_complete.feather

# ablation analysis
for n_estimator in n_estimators:
    run_name = "Run_estimator_"+str(n_estimator)
    print("Starting run: "+run_name)              
    p = subprocess.Popen(["../../.venv/bin/python", "DisorderPred_ML_CAID.py", \
                    "--run_name", run_name,  \
                    "--test", "False" ,\
                    "--n_estimators", str(n_estimator), \
                    "--savedFeatures" ,"True", \
                    "--train_Dir_path", "/home/mkabir3/Research/40_CAID3/6_MergeData/disorder_disprot23-CAID2-TrainSET_complete.feather", \
                    '--nox_test_path' , "/home/mkabir3/Research/40_CAID3/6_MergeData/disorder_nox_complete.feather", \
                    "--pdb_test_path" , "/home/mkabir3/Research/40_CAID3/6_MergeData/disorder_pdb_complete.feather"])

    
    #This makes the wait possible
    p_status = p.wait()
            

