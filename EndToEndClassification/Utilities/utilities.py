import os
import pydub
import pickle
import numpy as np
import pandas as pd
import scipy.signal


def load_audio(path, duration=5000, framerate=22050, channel_nr=1):
    """
    Loads .ogg files and converts to np.array with specified nr of channels, and sample rate.
    
    Args:
        path (str): to the audio file.
        duration (int): duration in ms. 
        framerate (int): sample rate. 
        channel_nr (int): numer of channels.

    Returns:
        (np.array): raw waveform.

    """
    audio = pydub.AudioSegment.silent(duration=duration)
    audio = audio.overlay(pydub.AudioSegment.from_file(path).set_frame_rate(framerate).set_channels(channel_nr))[
            0:duration]
    raw = (np.fromstring(audio._data, dtype="int16") + 0.5) / (0x7FFF + 0.5)  # convert to float

    return raw


def load_processed_esc50(load_folder_path):
    """
    Loads previously processed esc50 dataset. 

    Args:
        load_folder_path (str): folder should contain the esc50_meta.pkl and esc50_audio.dat files.

    Returns:
        (pd.DataFrame): meta data with columns 'category', 'category_name', 'filename', 'fold'.
        (np.memmap): raw waveform data with hard-coded shape (2000, 110250) and sample rate 22050.
    """

    if os.path.isfile(os.path.join(load_folder_path, 'esc50_meta.pkl')) and os.path.isfile(
            os.path.join(load_folder_path, 'esc50_audio.dat')):
        rows_meta = pd.read_pickle(os.path.join(load_folder_path, 'esc50_meta.pkl'))
        rows_audio = np.memmap(os.path.join(load_folder_path, 'esc50_audio.dat'), dtype='float32', mode='r',
                               shape=(2000, 110250))
        return rows_meta, rows_audio
    else:
        return ValueError('seems like you forgot to do esc50 file parsing')


def load_features_esc50(load_esc50_feature_path, raw=False):
    """
    Loads segmented/augmented esc50 features, either mel spectrograms or raw waveform.

    Args:
        load_esc50_feature_path (str): path to folder with 'esc50_features_long_raw.pkl' or 'esc50_features_long.pkl'.
        raw (bool): raw waveform (True) or logscaled melspecs.
    """

    if raw:
        esc50_features = pd.concat(
            (pd.read_pickle(os.path.join(load_esc50_feature_path, 'esc50_features_long_raw0.pkl')),
             pd.read_pickle(os.path.join(load_esc50_feature_path, 'esc50_features_long_raw1.pkl')),
             pd.read_pickle(os.path.join(load_esc50_feature_path, 'esc50_features_long_raw2.pkl'))))

        return esc50_features
    else:
        esc50_features = pd.concat(
            (pd.read_pickle(os.path.join(load_esc50_feature_path, 'esc50_features_long_logspec0.pkl')),
             pd.read_pickle(os.path.join(load_esc50_feature_path, 'esc50_features_long_logspec1.pkl')),
             pd.read_pickle(os.path.join(load_esc50_feature_path, 'esc50_features_long_logspec2.pkl'))))

        return esc50_features


def dump_pickle(save_path, dict_to_save):
    """
    Dumps a dictionary as a .pickle file.

    Args:
        save_path (str): path specifying location where you want to save your pickle file.
        dict_to_save (dict): dictionary that will be saved to the .pickle file.
    """

    with open(save_path, 'wb') as f:
        pickle.dump(dict_to_save, f)


def load_pickle(load_path):
    """
    Loads a .pickle file as dictionary.
    
    Args:
        load_path (str): path to .pkl file. 

    Returns:
        (dict): dictionary.
    """

    with open(load_path, 'rb') as f:
        pickle_to_load = pickle.load(f)
    return pickle_to_load


def generate_delta_031(spec):
    """
    Ported librosa v0.3.1. delta generation, taken from https://github.com/karoldvl/echonet.

    Args:
        spec (np.array): spectrogram.

    Returns:
        (np.array): corresponding delta.
    """

    window = np.arange(4, -5, -1)
    padding = [(0, 0), (5, 5)]
    delta = np.pad(spec, padding, mode='edge')
    delta = scipy.signal.lfilter(window, 1, delta, axis=-1)
    idx = [Ellipsis, slice(5, -5, None)]
    return delta[idx]


def classification_accuracy(test_data, predicted_labels):
    """
    Majority-voting based classification accuracy computation.
    
    Args:
        test_data (pd.DataFrame): data subset for which the accuracy needs to be calculated.
        predicted_labels (np.array): labels predicted by the model.

    Returns:
        (float): the accuracy based on majority-voting for each of the segments corresponding to a sound file.
    """

    test_data_with_preds = test_data.copy()
    test_data_with_preds.loc[:, 'prediction'] = predicted_labels

    group = test_data_with_preds.groupby('filename', sort=False)
    group = group[['category', 'prediction']].agg(lambda x: x.value_counts().index[0])

    accuracy = np.sum(group['category'] == group['prediction']) / float(len(group['category']))

    return accuracy


def retrieve_fold_results(experiment_folder_path):
    """
    Retrieves the validation and test results.

    Args:
        experiment_folder_path (str): path to the results folder.

    Returns:
        (list): validation results averaged over the folds.
        (list): test results averaged over the folds.
    """

    directories = os.listdir(experiment_folder_path)
    if not len(directories) == 5:
        raise ValueError('seems like there where more/less than the 5 folds in the folder')

    print('Printing results of the {} experiment'.format(os.path.split(experiment_folder_path)[1]))

    # val_accuracies = []
    test_accuracies = []

    for fold_exp in directories:
        pickle_path = os.path.join(experiment_folder_path, fold_exp)

        results = load_pickle(pickle_path)

        # val_accuracies.append(results['val_accuracy'])
        test_accuracies.append(results['test_accuracy'])

    # val_accuracies = np.mean(np.array(val_accuracies), axis=0)
    test_accuracies = np.mean(np.array(test_accuracies), axis=0)

    return test_accuracies
