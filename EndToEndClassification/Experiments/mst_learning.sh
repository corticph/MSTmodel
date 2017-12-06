#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"

# specify:
# 1. the base_data_path to the esc50 features and
# 2. the base_save_folder in which each MSTmodel (+results +predictions) will be saved in a fold-specific subfolder


base_data_path=""
base_save_folder=""


python mst_learning.py $base_data_path $base_save_folder 1 2
python mst_learning.py $base_data_path $base_save_folder 2 3
python mst_learning.py $base_data_path $base_save_folder 3 4
python mst_learning.py $base_data_path $base_save_folder 4 5
python mst_learning.py $base_data_path $base_save_folder 5 1
