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

from EndToEndClassification.Utilities import generate_delta_031, load_features_esc50


class ClassifierLoader():
    """
    """

    def __init__(self, features_path, test_fold, validation_fold, raw=False, deltas=False):
        """
        Initializes the classfifier loader.

        Args:
            features_path (str): path to the processed esc50 features.
            test_fold (int): fold used for testing (not used).
            validation_fold (int): fold used for validation.
            raw (bool): whether or not raw features are used for training the classifier.
            deltas (bool): whether to compute and use delta features (can only be used together with mel-spectrogram
                            features).
        """

        if raw:
            if deltas:
                raise ValueError('delta computations are not compatible with raw waveform input')
            features = load_features_esc50(features_path, raw=True)
            self.shape = (-1, 51200, 1)
        else:
            features = load_features_esc50(features_path, raw=False)
            self.shape = (-1, 60, 101, 1)

        train = features[(features['fold'] != test_fold) & (features['fold'] != validation_fold)]

        self.validation_pd = features[(features['fold'] == validation_fold) & (features['augmented'] == 0)]
        self.test_pd = features[(features['fold'] == test_fold) & (features['augmented'] == 0)]

        self.deltas = deltas

        self.validation_fold = validation_fold
        self.test_fold = test_fold

        if raw:
            self.start = 'raw_0'
        else:
            self.start = 'logspec_b0_f0.0'

        self.end = features.columns[-1]

        X = train.loc[:, self.start:self.end].as_matrix()
        y = train['category'].as_matrix()

        X_validation = self.validation_pd.loc[:, self.start:self.end].as_matrix()
        y_validation = self.validation_pd['category'].as_matrix()

        X_test = self.test_pd.loc[:, self.start:self.end].as_matrix()
        y_test = self.test_pd['category'].as_matrix()

        # normalize the features
        X_mean, X_std = np.mean(X), np.std(X)
        X = (X - X_mean) / X_std
        X_validation = (X_validation - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std
        X_min = np.min(X)
        X_max = np.max(X)
        X = 2 * (X - X_min) / (X_max - X_min) - 1
        X_validation = 2 * (X_validation - X_min) / (X_max - X_min) - 1
        X_test = 2 * (X_test - X_min) / (X_max - X_min) - 1

        # perform the appropriate reshaping
        X = np.reshape(X, self.shape, order='F')
        X_validation = np.reshape(X_validation, self.shape, order='F')
        X_test = np.reshape(X_test, self.shape, order='F')

        if self.deltas:
            X = self.generate_deltas(X)
            X_validation = self.generate_deltas(X_validation)
            X_test = self.generate_deltas(X_test)

        # do some testing shape wise
        if not X.shape[0] == len(y):
            raise ValueError('nr train samples should match nr train labels')
        if not X_validation.shape[0] == len(y_validation):
            raise ValueError('nr val samples should match nr val labels')
        if not X_test.shape[0] == len(y_test):
            raise ValueError('nr test samples should match nr test labels')

        self.train = (X, y)
        self.validation = (X_validation, y_validation)
        self.test = (X_test, y_test)

    def generate_deltas(self, X):
        """
        Generates deltas for each of the mel-spectrograms.

        Args:
            X (np.array): the mel-spectrograms.

        Returns:
            (np.array): the mel-spectrograms with their deltas (added in the channel dimension).
        """
        new_dim = np.zeros(np.shape(X))
        X = np.concatenate((X, new_dim), axis=3)
        del new_dim
        for i in range(len(X)):
            X[i, :, :, 1] = generate_delta_031(X[i, :, :, 0])

        return X

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
