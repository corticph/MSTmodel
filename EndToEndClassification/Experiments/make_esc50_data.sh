#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=""

# specify
# 1. the esc_50_path to the downloaded dataset folder
# 2. destination_folder where you want to store the parsed esc50 dataset (subfolder 1) and the features (subfolder 2)

esc_50_source_path=""
destination_folder_path=""

python make_esc50_data.py $esc_50_source_path $destination_folder_path
