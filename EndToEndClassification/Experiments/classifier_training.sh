#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"

# specify:
# 1. the base_data_path to the esc50 features and
# 2. the save folder (in it 3 classifier save subfolders will be made) and
# 3. in the case of initialization: the base_path_pretrained_MST to the folder with fold-specific trained MSTmodels


base_data_path=""
base_save_folder=""
base_path_pretrained_MST=""


# baseline spectrogram training
python classifier_training.py $base_data_path $base_save_folder 1 2 'spects'
python classifier_training.py $base_data_path $base_save_folder 2 3 'spects'
python classifier_training.py $base_data_path $base_save_folder 3 4 'spects'
python classifier_training.py $base_data_path $base_save_folder 4 5 'spects'
python classifier_training.py $base_data_path $base_save_folder 5 1 'spects'

# baseline raw waveform training
python classifier_training.py $base_data_path $base_save_folder 1 2 'raw'
python classifier_training.py $base_data_path $base_save_folder 2 3 'raw'
python classifier_training.py $base_data_path $base_save_folder 3 4 'raw'
python classifier_training.py $base_data_path $base_save_folder 4 5 'raw'
python classifier_training.py $base_data_path $base_save_folder 5 1 'raw'

# MSTmodel-pretrained initialized training
python classifier_training.py $base_data_path $base_save_folder 1 2 'initialized' --base_path_pretrained_MST $base_path_pretrained_MST
python classifier_training.py $base_data_path $base_save_folder 2 3 'initialized' --base_path_pretrained_MST $base_path_pretrained_MST
python classifier_training.py $base_data_path $base_save_folder 3 4 'initialized' --base_path_pretrained_MST $base_path_pretrained_MST
python classifier_training.py $base_data_path $base_save_folder 4 5 'initialized' --base_path_pretrained_MST $base_path_pretrained_MST
python classifier_training.py $base_data_path $base_save_folder 5 1 'initialized' --base_path_pretrained_MST $base_path_pretrained_MST
