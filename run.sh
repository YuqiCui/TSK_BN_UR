# run TSK-BN-UR on the Yeast dataset with numer of rules set to 20, the weight for UR will automatically be tuned during training with --weight_frs > 0
python3 main_diff_data_split.py --data Yeast --bn --n_rules 20 --init kmean --cpu --repeats 5 --weight_frs 1
