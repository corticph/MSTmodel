# Copyright 2018 Corti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from EndToEndClassification.MSTmodel import MSTmodel, MSTLoader, MSTtrainer
import os
import argparse

description = 'log-scaled mel-spectrogram learning for a single fold'

parser = argparse.ArgumentParser(description=description)
parser.add_argument("base_data_path", type=str)
parser.add_argument("base_save_folder", type=str)
parser.add_argument("validation_fold", type=int)
parser.add_argument("test_fold", type=int)

args = parser.parse_args()

path_raw = os.path.join(args.base_data_path, 'raw')
path_spect = os.path.join(args.base_data_path, 'spect')

# train the model for each of the fold combinations
d = MSTLoader(path_raw, path_spect, args.test_fold, args.validation_fold)
m = MSTmodel('MSTmodel')
t = MSTtrainer(m, d, args.base_save_folder, restore_path=None)
t.train(batch_size=100, no_epochs=800, lr=3e-4, overfit_window=50)
