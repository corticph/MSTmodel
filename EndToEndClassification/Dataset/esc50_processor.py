import os
import progressbar
import multiprocessing
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import librosa

from EndToEndClassification.Utilities import load_audio, load_processed_esc50

# note: the original code for processing the ESC-50 dataset by Piczak uses libc. This is not strictly necessary
# but potentially speeds-up processing. See commented-out code.

# from ctypes import cdll, CDLL
# cdll.LoadLibrary("libc.so.6")
# libc = CDLL("libc.so.6")

CPU_COUNT = multiprocessing.cpu_count()


# Note: most of the processing code below code adapted with permission from
# 'https://github.com/karoldvl/paper-2015-esc-convnet'.

# Note #2: the ported librosa v0.3.1. delta generation is added below for those interested in reproducing
# Piczak's original results with delta features added when using a recent librosa version (0.5.1).


def ESC50Processor(esc_50_path, destination_folder):
    """
    Wrapper function for convenient processing. First subfolders are made in the destination folder for the processed 
    dataset (1) and the logmel and raw waveform features (i.e. the segmented/augmented features). 
    One can use the internal functions if more flexibility is required.
    
    Args:
        esc_50_path (str): path to the esc50 data.
        destination_folder (str): path to a destination folder.
    """

    if not (os.path.isdir(esc_50_path) and os.path.isdir(destination_folder)):
        raise ValueError('please provide valid paths to a source and a destination folder')

    # make the necessary subfolders for processing
    processed_esc50_path = os.path.join(destination_folder, 'processed_esc50')
    os.mkdir(processed_esc50_path)
    features_path = os.path.join(destination_folder, 'features')
    os.mkdir(features_path)
    features_raw_path = os.path.join(features_path, 'raw')
    os.mkdir(features_raw_path)
    features_logmel_path = os.path.join(features_path, 'spect')
    os.mkdir(features_logmel_path)

    _process_esc50(esc_50_path, processed_esc50_path)
    _dump_features_processed_esc50_combined(processed_esc50_path, features_logmel_path, features_raw_path,
                                            augmentations=4, frames=101, seed=41, batch_size=50)

    print('done')


def _process_esc50(esc_50_path, save_path):
    """
    Processes the 2000 5-sec clips of the esc50 dataset and dumps a pickle with the metadata for each audio file. 
    The sample rate is hard-coded to 22050.
    
    Taken with permission from 'https://github.com/karoldvl/paper-2015-esc-convnet' with minor adaptions.
    
    Args:
        esc_50_path (str): path to the base folder containing the class-specific subfolders.
        save_path (str): folder in which the esc50_audio.dat and the esc50_meta.pkl files will be saved.
    """

    rows_meta = []
    rows_audio = []
    category_counter = 0

    for directory in sorted(os.listdir(esc_50_path)):

        directory = os.path.join(esc_50_path, directory)

        if not (os.path.isdir(directory) and os.path.basename(directory)[0:3].isdigit()):
            continue
        print('Parsing ' + directory)

        bar = progressbar.DataTransferBar(max_value=len(os.listdir(directory)))
        for i, clip in enumerate(sorted(os.listdir(directory))):
            if clip[-3:] != 'ogg':
                continue
            filepath = '{0}/{1}'.format(directory, clip)
            filename = os.path.basename(filepath)
            fold = filename[0]
            category = category_counter
            category_name = os.path.dirname(filepath).split('/')[-1]
            rows_meta.append(
                pd.DataFrame({'filename': filename, 'fold': fold, 'category': category, 'category_name': category_name},
                             index=[0]))
            rows_audio.append(load_audio(filepath, 5000, framerate=22050, channel_nr=1))
            bar.update(i)
        bar.finish()
        # libc.malloc_trim(0)
        rows_meta = [pd.concat(rows_meta, ignore_index=True)]
        rows_audio = [np.vstack(rows_audio)]
        category_counter = category_counter + 1

    rows_meta = rows_meta[0]
    rows_meta[['category', 'fold']] = rows_meta[['category', 'fold']].astype(int)

    rows_meta.to_pickle(os.path.join(save_path, 'esc50_meta.pkl'))
    mm = np.memmap(os.path.join(save_path, 'esc50_audio.dat'), dtype='float32', mode='w+', shape=(2000, 110250))

    mm[:] = rows_audio[0][:]
    mm.flush()
    del rows_audio

    print('processed and saved')


def _dump_features_processed_esc50_combined(load_parsed_esc50, save_folder_path, save_folder_path_raw, augmentations=4,
                                            frames=101, seed=41, batch_size=50):
    """
    Generates ESC50 features from the 'processed' dataset. It does so according to the specifications in the paper. 
    Each of the 2000 5sec clips is cut into 50% overlapping segments. 4 augmentations are made of each. 
    Largely the same in implementation as the original Piczak code.
    
    Args:
        load_parsed_esc50 (str): folder containing the esc50_meta.pkl and esc50_audio.dat files.
        save_folder_path (str): folder for saving logscaled mel features.
        save_folder_path_raw (str): folder for saving raw waveform features.
        augmentations (int): number of augmentations of each segment.
        frames (int): nr of frames of the mel features.
        seed (int): seed for pseudo RNG.  
        batch_size (int): batch size for multiprocessing (note, this has nothing to do with the minibatch size).
    """

    np.random.seed(seed)

    # checks
    if isinstance(load_parsed_esc50, str):
        meta, audio = load_processed_esc50(load_parsed_esc50)
    else:
        raise ValueError('load_parsed_esc50 should be a to folder')
    if not (os.path.isdir(save_folder_path) and os.path.isdir(save_folder_path_raw)):
        raise ValueError('please provide valid folders for saving the features')

    segments = []
    segments_raw = []

    for b in range(len(audio) // batch_size + 1):
        print('b:{}'.format(b))
        start = b * batch_size
        end = (b + 1) * batch_size
        if end > len(audio):
            end = len(audio)

        seg_combined = Parallel(n_jobs=CPU_COUNT)(delayed(_extract_segments_combined)((
            audio[i, :],
            meta.loc[i, 'filename'],
            meta.loc[i, 'fold'],
            meta.loc[i, 'category'],
            meta.loc[i, 'category_name'],
            0,
            frames
        )) for i in range(start, end))

        segments_batch = [seg[0] for seg in seg_combined]
        segments_raw_batch = [seg[1] for seg in seg_combined]

        segments.extend(segments_batch)
        segments_raw.extend(segments_raw_batch)

        for _ in range(augmentations):
            seg_combined = Parallel(n_jobs=CPU_COUNT)(delayed(_extract_segments_combined)((
                _augment_esc50(audio[i, :]),
                meta.loc[i, 'filename'],
                meta.loc[i, 'fold'],
                meta.loc[i, 'category'],
                meta.loc[i, 'category_name'],
                1,
                frames
            )) for i in range(start, end))

            segments_batch = [seg[0] for seg in seg_combined]
            segments_raw_batch = [seg[1] for seg in seg_combined]

            segments.extend(segments_batch)
            segments_raw.extend(segments_raw_batch)

        segments = [pd.concat(segments, ignore_index=True)]
        segments_raw = [pd.concat(segments_raw, ignore_index=True)]

        print('{} / {}'.format(end, len(audio)))

    # # split among 3 files because of size
    segments[0][0:30000].to_pickle(os.path.join(save_folder_path, 'esc50_features_long_logspec0.pkl'))
    segments[0][30000:60000].to_pickle(os.path.join(save_folder_path, 'esc50_features_long_logspec1.pkl'))
    segments[0][60000:].to_pickle(os.path.join(save_folder_path, 'esc50_features_long_logspec2.pkl'))

    segments_raw[0][0:30000].to_pickle(os.path.join(save_folder_path_raw, 'esc50_features_long_raw0.pkl'))
    segments_raw[0][30000:60000].to_pickle(os.path.join(save_folder_path_raw, 'esc50_features_long_raw1.pkl'))
    segments_raw[0][60000:].to_pickle(os.path.join(save_folder_path_raw, 'esc50_features_long_raw2.pkl'))


def _extract_segments_combined(args):
    """
    Segments the audio clip and adds the raw waveform as well as the logscaled mel-spec segments to pd.DataFrame.

    Args:
        args (tuple): (clip, filename, fold, category, category_name, augmented, frames).
    
    Returns:
        (pd.DataFrame): segmented/augmented logscaled mel-spec features for single clip.
        (pd.DataFrame): segmented/augmented raw waveform features for single clip.
    """

    clip, filename, fold, category, category_name, augmented, frames = args

    FRAMES_PER_SEGMENT = frames - 1
    WINDOW_SIZE = 512 * FRAMES_PER_SEGMENT
    STEP_SIZE = 512 * FRAMES_PER_SEGMENT // 2
    BANDS = 60

    s = 0
    segments = []
    segments_raw = []

    normalization_factor = 1 / np.max(np.abs(clip))
    clip = clip * normalization_factor

    while len(clip[s * STEP_SIZE:s * STEP_SIZE + WINDOW_SIZE]) == WINDOW_SIZE:
        signal = clip[s * STEP_SIZE:s * STEP_SIZE + WINDOW_SIZE]

        melspec = librosa.feature.melspectrogram(signal, sr=22050, n_fft=1024, hop_length=512, n_mels=BANDS)
        logspec = librosa.logamplitude(melspec)
        logspec = logspec.T.flatten()[:, np.newaxis].T
        logspec = pd.DataFrame(data=logspec, dtype='float32', index=[0], columns=list(
            'logspec_b{}_f{}'.format(i % BANDS, i / BANDS) for i in range(np.shape(logspec)[1])))

        signal = signal.reshape([signal.shape[0], 1])
        signal_pd = pd.DataFrame(data=signal.T, dtype='float32', index=[0], columns=list(
            'raw_{}'.format(i) for i in range(len(signal))))

        if np.mean(logspec.as_matrix()) > -72.0:  # drop silent frames
            segment_meta = pd.DataFrame(
                {'filename': filename, 'fold': fold, 'category': category, 'category_name': category_name,
                 's_begin': s * STEP_SIZE, 's_end': s * STEP_SIZE + WINDOW_SIZE, 'augmented': augmented}, index=[0])
            segments.append(pd.concat((segment_meta, logspec), axis=1))
            segments_raw.append(pd.concat((segment_meta, signal_pd), axis=1))
            # print('processed segment')
        else:
            # print('discarded silent segment:')
            # print(pd.DataFrame(
            #    {'filename': filename, 'fold': fold, 'category': category, 'category_name': category_name,
            #     's_begin': s * STEP_SIZE, 's_end': s * STEP_SIZE + WINDOW_SIZE, 'augmented': augmented}, index=[0]))
            pass
        s = s + 1

    segments = pd.concat(segments, ignore_index=True)
    segments_raw = pd.concat(segments_raw, ignore_index=True)
    # libc.malloc_trim(0)

    return segments, segments_raw


def _augment_esc50(audio, sample_rate=22050):
    """
    Applies random pitch/time shifting and time stretching to a segment of an audio clip.
     
    Args:
        audio (np.array): audio segment.
        sample_rate (int): sample rate.
    
    Returns:
        (np.array): 'augmented' audio segment.
    """

    limits = ((0, 0), (1.0, 1.0))  # pitch shift in half-steps, time stretch

    pitch_shift = np.random.randint(limits[0][1], limits[0][1] + 1)
    time_stretch = np.random.random() * (limits[1][1] - limits[1][0]) + limits[1][0]
    time_shift = np.random.randint(sample_rate)

    augmented_audio = np.hstack((np.zeros((time_shift)),
                                 librosa.effects.time_stretch(
                                     librosa.effects.pitch_shift(audio, sample_rate, pitch_shift),
                                     time_stretch)))

    return augmented_audio
