rm -rf __pycache__
# ../../.venv/bin/pyflakes  DisorderPred_ML_CAID.py 

# print the last experiment name
echo "Last Experiment Name:"
cat exp_name.txt

# Ask if the user wants to use the last experiment name
echo "Use the last experiment name? (y/n)"
read use_last_exp_name
if [ $use_last_exp_name == "y" ]
then
        exp_name=$(cat exp_name.txt | cut -d ":" -f 2)
else
    echo "Enter Experiment  Name:"
    read exp_name
    if [ -z $exp_name ]
    then
            exp_name="TestExp"
    fi
fi

# log the exp_name
echo "$exp_name" > exp_name.txt


# True for Test
.venv/bin/python DisorderPred_ML_CAID.py \
                                    --run_name $exp_name  \
                                    --test "False" \
                                    --n_estimators  100 \
                                    --savedFeatures "True" \
                                    --train_Dir_path /home/SharedFiles/Wasi/MergedFeatures/DisPred/disorder_disprot23-CAID2-TrainSET_complete.feather \
                                    --nox_test_path /home/SharedFiles/Wasi/MergedFeatures/DisPred/disorder_nox_complete.feather \
                                    --pdb_test_path /home/SharedFiles/Wasi/MergedFeatures/DisPred/disorder_pdb_complete.feather

                                    