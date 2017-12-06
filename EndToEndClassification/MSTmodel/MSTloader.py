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

import numpy as np
import random

from EndToEndClassification.Utilities import load_features_esc50


class MSTLoader():
    """
    Loads the raw esc50 features as input, and the melspectrogram features as labels.
    """

    def __init__(self, features_raw_path, features_spect_path, test_fold, validation_fold):
        """
        Initializes the loader.

        Args:
            features_raw (str): path to raw esc50 features.
            features_spect (str): path to spect esc50 features.
            test_fold (int): test fold.
            validation_fold (int): validation fold.
        """

        # load features from paths
        features_raw = load_features_esc50(features_raw_path, raw=True)
        features_spect = load_features_esc50(features_spect_path, raw=False)

        self.raw_shape = (-1, 51200, 1)
        self.spect_shape = (-1, 60, 101, 1)

        self.validation_fold = validation_fold
        self.test_fold = test_fold

        train_raw = features_raw[(features_raw['fold'] != test_fold) & (features_raw['fold'] != validation_fold)]
        train_spect = features_spect[
            (features_spect['fold'] != test_fold) & (features_spect['fold'] != validation_fold)]

        validation_pd_raw = features_raw[(features_raw['fold'] == validation_fold) & (features_raw['augmented'] == 0)]
        validation_pd_spect = features_spect[(features_spect['fold'] == validation_fold) &
                                             (features_spect['augmented'] == 0)]

        test_pd_raw = features_raw[(features_raw['fold'] == test_fold) & (features_raw['augmented'] == 0)]
        test_pd_spect = features_spect[(features_spect['fold'] == test_fold) & (features_spect['augmented'] == 0)]

        self.start_raw = 'raw_0'
        self.start_spect = 'logspec_b0_f0.0'

        self.end_raw = features_raw.columns[-1]
        self.end_spect = features_spect.columns[-1]

        X = train_raw.loc[:, self.start_raw:self.end_raw].as_matrix()
        y = train_spect.loc[:, self.start_spect:self.end_spect].as_matrix()

        X_validation = validation_pd_raw.loc[:, self.start_raw:self.end_raw].as_matrix()
        y_validation = validation_pd_spect.loc[:, self.start_spect:self.end_spect].as_matrix()

        X_test = test_pd_raw.loc[:, self.start_raw:self.end_raw].as_matrix()
        y_test = test_pd_spect.loc[:, self.start_spect:self.end_spect].as_matrix()

        X = np.reshape(X, self.raw_shape, order='F')
        X_validation = np.reshape(X_validation, self.raw_shape, order='F')
        X_test = np.reshape(X_test, self.raw_shape, order='F')

        # normalize and rescale the melspectrogram labels

        y_mean, y_std = np.mean(y), np.std(y)
        y = (y - y_mean) / y_std
        y_validation = (y_validation - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std
        y_min = np.min(y)
        y_max = np.max(y)
        y = 2 * (y - y_min) / (y_max - y_min) - 1
        y_validation = 2 * (y_validation - y_min) / (y_max - y_min) - 1
        y_test = 2 * (y_test - y_min) / (y_max - y_min) - 1

        y = np.reshape(y, self.spect_shape, order='F')
        y_validation = np.reshape(y_validation, self.spect_shape, order='F')
        y_test = np.reshape(y_test, self.spect_shape, order='F')

        if len(y) != len(X):
            raise ValueError('train data does not line up')
        if len(y_validation) != len(X_validation):
            raise ValueError('val data does not line up')
        if len(y_test) != len(X_test):
            raise ValueError('test data does not line up')

        self.train = (X, y)
        self.validation = (X_validation, y_validation)
        self.test = (X_test, y_test)

    def make_batch_indices(self, batch_size):
        """
        Generates randomized batch indices for a single train epoch.

        Args:
            batch_size (int): size of the batch.

        Returns:
            (list): containing the indices for each of the batches.
        """
        indices = list(range(self.train[0].shape[0]))
        random.shuffle(indices)

        nr_batches = int(np.floor(self.train[0].shape[0] / batch_size))
        if nr_batches == 0:
            raise ValueError('batch size is too big')
        else:
            return [indices[b * batch_size:(b + 1) * batch_size] for b in range(nr_batches)]

    def load_batch(self, batch_indices):
        """
        Loads a single batch from the trainset.

        Args:
            batch_indices (list): indices for the batch.

        Returns:
            (np.array): batch examples.
            (np.array): batch labels.
        """
        return self.train[0][batch_indices], self.train[1][batch_indices]
