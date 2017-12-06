from EndToEndClassification.EnvClassification.Models import Piczak, RawPiczak
from EndToEndClassification.EnvClassification import ClassifierLoader, ClassifierTrainer
import argparse
import os

description = 'training one of the 3 different classifiers for a single fold'

parser = argparse.ArgumentParser(description=description)
parser.add_argument("base_data_path", type=str)
parser.add_argument("base_save_folder", type=str)
parser.add_argument("validation_fold", type=int)
parser.add_argument("test_fold", type=int)
parser.add_argument("type", type=str)
parser.add_argument("--base_path_pretrained_MST", type=str)

args = parser.parse_args()


if args.type == 'spects':
    path_spect = os.path.join(args.base_data_path, 'spect')
    base_save_folder_spects = os.path.join(args.base_save_folder, 'spects_model')
    if not os.path.isdir(base_save_folder_spects):
        os.mkdir(base_save_folder_spects)
    d_s = ClassifierLoader(path_spect, args.test_fold, args.validation_fold, raw=False)
    c_s = Piczak('Piczak_baseline')
    t = ClassifierTrainer(c_s, d_s, base_save_folder_spects, save_separate=False)
    t.train()
elif args.type == 'raw':
    path_raw = os.path.join(args.base_data_path, 'raw')
    base_save_folder_raw = os.path.join(args.base_save_folder, 'raw_model')
    if not os.path.isdir(base_save_folder_raw):
        os.mkdir(base_save_folder_raw)
    d_r = ClassifierLoader(path_raw, args.test_fold, args.validation_fold, raw=True)
    c_r = RawPiczak('RawPiczak_baseline', dropout=True)
    t_r = ClassifierTrainer(c_r, d_r, base_save_folder_raw, MSTfrozen=False)
    t_r.train()
elif args.type == 'initialized':
    path_raw = os.path.join(args.base_data_path, 'raw')
    base_save_folder_raw_initialized = os.path.join(args.base_save_folder, 'mst_initialized_model')
    if not os.path.isdir(base_save_folder_raw_initialized):
        os.mkdir(base_save_folder_raw_initialized)
    d_r = ClassifierLoader(path_raw, args.test_fold, args.validation_fold, raw=True)
    c_r_i = RawPiczak('RawPiczak_initialized', dropout=False)
    m_r_p = os.path.join(args.base_path_pretrained_MST, 'MSTmodel_' + str(args.validation_fold) + str(args.test_fold) +
                         '/MSTmodel')
    t_r_i = ClassifierTrainer(c_r_i, d_r, base_save_folder_raw_initialized,
                              MSTmodel_initialized=m_r_p)
    t_r_i.train()
else:
    raise ValueError('Please provide either spects, raw, or initialized as type argument')


