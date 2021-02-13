from pathlib import Path
import os
import tensorflow as tf
from librosa import filters, core
from scipy import signal
import numpy as np
import json
import glob
import random


class PreprocessWrapper:
    """ Wrapper object for creating, reading and preprocessing datasets.
    """
    def __init__(self, dlnet_config: dict, ds_config: str, binary=True):
        """
        Init wrapper object. Reads DL Network config and dataset config.

        Parameters
        ----------
        dlnet_config : dict
            Config for DL Network and preprocessing
        ds_config : str
            Path to Dataset config to extract classes
        binary : bool, optional
            Set to True if for binary problem (compress/uncompressed), by
            default True
        """
        self._config = dlnet_config
        # Set random seed:
        random.seed = self._config['random_seed']
        # Classes
        if binary:
            self._config['classes'] = ['compressed_wav', 'uncompr_wav']
        else:
            self._config['classes'] = self.get_classes_from_dataset(ds_config)
        # Mel filter init:
        if self._config['calculate_mel']:
            self._mel_filter = filters.mel(self._config['sr'],
                                           self._config['n_fft'],
                                           n_mels=dlnet_config['n_mels'],
                                           norm='slaney')

    # Groundtruth extraction from folder name
    def folder_name_to_one_hot(self, file_path: str):
        """
        Extracts label from path information by searching matches between
        classes and path parts.

        Parameters
        ----------
        file_path : str
            Path to wav file.

        Returns
        -------
        tensorflow.one_hot
            Tensor with label information.

        Raises
        ------
        ValueError
            [description]
        """
        for label_idx, label in enumerate(self._config['classes']):
            if label in Path(file_path).parts:
                # get one hot encoded array
                return tf.one_hot(label_idx, len(self._config['classes']),
                                  dtype=tf.uint8)
        raise ValueError("Data cannot be labeled.")

    def load_and_preprocess_data(self, file_path: str):
        """
        Loads wav data and computes mel spectrum.

        Parameters
        ----------
        file_path : str
            Path to wav file.

        Returns
        -------
        Tuple(numpy.ndarray, tensorflow.one_hot)
            Mel spectrum and Tensorflow one_hot object.
        """
        # path string is saved as byte array in tf.data.dataset
        # -> convert back to str
        if type(file_path) is not str:
            file_path = file_path.numpy()
            file_path = file_path.decode('utf-8')

        # load audio data
        y, _ = core.load(file_path, sr=self._config['sr'],
                         mono=self._config['mono'], dtype=np.float32,
                         res_type='kaiser_best')

        if self._config['filter_signal']:
            sos = signal.butter(10,
                                self._config['filter_config'][1],
                                self._config['filter_config'][0],
                                fs=self._config['sr'],
                                output='sos')
            y = signal.sosfilt(sos, y)

        # calculate stft from audio data
        spectrogram = np.abs(core.stft(y, n_fft=self._config['n_fft'],
                                       hop_length=self._config['hop_length'],
                                       win_length=self._config['win_length'],
                                       window=self._config['window'],
                                       center=self._config['center'],
                                       dtype=np.complex64,
                                       pad_mode=self._config['pad_mode'])
                             ).astype(np.float32)

        # get ground truth from file_path string
        one_hot = self.folder_name_to_one_hot(file_path)

        if self._config['calculate_mel']:
            # filter stft with mel-filter
            spectrogram = self._mel_filter.dot(spectrogram **
                                               self._config['power'])

        # add channel dimension for conv layer compatibility
        spectrogram = np.expand_dims(spectrogram, axis=-1)

        return spectrogram, one_hot

    def preprocessing_wrapper(self, file_path: str):
        """
        Wrapper for preprocessing function to use in tensorflow API. 

        Parameters
        ----------
        file_path : str
            Path to wav file.

        Returns
        -------
        Tuple(numpy.ndarray, tensorflow.one_hot)
            Mel spectrum and Tensorflow one_hot object.
        """
        spec, one_hot = tf.py_function(func=self.load_and_preprocess_data,
                                       inp=[file_path],
                                       Tout=[tf.float32, tf.uint8])

        spec = tf.ensure_shape(spec, self._config['input_shape'])
        one_hot = tf.ensure_shape(one_hot, len(self._config['classes']))
        return spec, one_hot

    def tf_dataset_from_codec(self, codec_dir, train_test_ratio=.8,
                              save=False):
        """
        Generate a tensorflow dataset from the codec directory.

        Parameters
        ----------
        codec_dir : str
            Path to codec directory in database directory
        train_test_ratio : float, optional
            Ratio between test and train set, by default .8
        save : bool, optional
            Save database to disk, by default False

        Returns
        -------
        tuple, tensorflow.data.Dataset
            Two tensowrflow datasets: Train, Test
        """
        # autotune computation
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        train_list, test_list = self.get_train_test_lists(codec_dir,
                                                          train_test_ratio)
        # Train set
        train_set = tf.data.Dataset.list_files(train_list[0])
        for train in train_list[1:]:
            # define a dataset of file paths
            train_set = train_set.concatenate(tf.data.Dataset.list_files(
                                            train))
        # Test set
        test_set = tf.data.Dataset.list_files(test_list[0])
        for test in test_list[1:]:
            # define a dataset of file paths
            test_set = test_set.concatenate(tf.data.Dataset.list_files(test))
        # Preprocessing via map
        train_set = train_set.map(self.preprocessing_wrapper,
                                  num_parallel_calls=AUTOTUNE)
        test_set = test_set.map(self.preprocessing_wrapper,
                                num_parallel_calls=AUTOTUNE)
        if save:
            # save dataset to disk
            train_name = 'train_set_'+os.path.split(codec_dir)[1]
            train_path = os.path.join('_data', train_name)
            test_name = 'test_set_'+os.path.split(codec_dir)[1]
            test_path = os.path.join('_data', test_name)
            tf.data.experimental.save(dataset=train_set,
                                      path=train_path,
                                      compression='GZIP')
            tf.data.experimental.save(dataset=test_set,
                                      path=test_path,
                                      compression='GZIP')
        return train_set, test_set

    def tf_dataset_from_database(self, db_path: str, train_test_ratio=.8,
                                 save=False):
        """
        Generate a tensorflow dataset from the database directory with all
        subfolders(codecs) included.

        Parameters
        ----------
        db_path : str
            Path to database directory.
        train_test_ratio : float, optional
            Ratio between test and train set, by default .8
        save : bool, optional
            Save database to disk, by default False

        Returns
        -------
        tuple, tensorflow.data.Dataset
            Two tensowrflow datasets: Train, Test
        """
        # Get codec dirs:
        codecs = glob.glob(os.path.join(db_path, 'compressed_wav', '**'))
        codecs.append(os.path.join(db_path, 'uncompr_wav'))
        # Loop over codecs:
        train_set, test_set = self.tf_dataset_from_codec(codecs[0],
                                                         train_test_ratio)
        for codec in codecs:
            train, test = self.tf_dataset_from_codec(codec, train_test_ratio)
            train_set.concatenate(train)
            test_set.concatenate(test)
        return train_set, test_set

    def get_train_test_lists(self, codec_dir: str, train_test_ratio=.8):
        """
        Returns two lists of subfolders in codec_dir for train and test
        directories.

        Parameters
        ----------
        codec_dir : str
            Path to codec directory in database directory
        train_test_ratio : float, optional
            Ratio between test and train set, by default .8

        Returns
        -------
        tuple, list
            Two list for train and test file directories.
        """
        # Number of subfolders:
        n_folders = len(glob.glob(os.path.join(codec_dir, '**')))
        # Seed list
        seeds = list(range(1, n_folders))
        # Train and test indices:
        train_idx = random.sample(seeds, int(train_test_ratio*n_folders))
        test_idx = list(set(seeds)-set(train_idx))
        # Train list
        train = [os.path.join(codec_dir, str(idx), '*.wav')
                 for idx in train_idx]
        # Test list
        test = [os.path.join(codec_dir, str(idx), '*.wav') for idx in test_idx]
        return train, test

    def load_tf_dataset(self, directory: str):
        """
        Wrapper to load dataset from directory.

        Parameters
        ----------
        directory : str
            Path to directory.

        Returns
        -------
        tensorflow.data.Dataset
            Dataset
        """
        return tf.data.experimental.load(directory,
                        (tf.TensorSpec(self._config['input_shape'],
                                       dtype=tf.float32, name=None),
                         tf.TensorSpec(len(self._config['classes']),
                                       dtype=tf.uint8, name=None)),
                        compression='GZIP')

    def get_classes_from_dataset(self, json_file: str):
        """
        Get all available classes from the dataset config.

        Parameters
        ----------
        json_file : str
            Path to dataset config file

        Returns
        -------
        list
            List of all possible classes.
        """
        # Read json Dataset_config
        with open(json_file, "r") as read_file:
            config = json.load(read_file)
            # Get list of codec settings:
            codec_list = list(config.keys())[5:]
            # Replace 'db_format' by 'uncompr_wav' and return
            return ['uncompr_wav' if i == 'db_format' else i for i in
                    codec_list]

    @property
    def classes(self):
        return self._config['classes']
