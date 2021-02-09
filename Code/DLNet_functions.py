from pathlib import Path
import os
import tensorflow as tf
from librosa import filters
from librosa import core
import numpy as np
import json


class preprocess_wrapper:
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
        if binary:
            self._config['classes'] = ['compressed_wav', 'uncompr_wav']
        else:
            self._config['classes'] = self.get_classes_from_dataset(ds_config)
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

        # calculate stft from audio data
        stft = core.stft(y, n_fft=self._config['n_fft'],
                         hop_length=self._config['hop_length'],
                         win_length=self._config['win_length'],
                         window=self._config['window'],
                         center=self._config['center'],
                         dtype=np.complex64,
                         pad_mode=self._config['pad_mode'])

        # filter stft with mel-filter
        mel_spec = self._mel_filter.dot(
            np.abs(stft).astype(np.float32) ** self._config['power'])

        # add channel dimension for conv layer compatibility
        mel_spec = np.expand_dims(mel_spec, axis=-1)

        # get ground truth from file_path string
        one_hot = self.folder_name_to_one_hot(file_path)

        return mel_spec, one_hot

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
        mel_spec, one_hot = tf.py_function(func=self.load_and_preprocess_data,
                                           inp=[file_path],
                                           Tout=[tf.float32, tf.uint8])

        mel_spec = tf.ensure_shape(mel_spec, self._config['input_shape'])
        one_hot = tf.ensure_shape(one_hot, len(self._config['classes']))
        return mel_spec, one_hot

    def gen_tf_dataset(self, directory: str, save=False):
        """
        Takes a path to a directory with samples to create a tensorflow
        dataset.

        Parameters
        ----------
        directory : str
            Path to directory
        save : bool, optional
            Set to True if datset should be saved, by default False

        Returns
        -------
        tensorflow.data.Dataset
            Dataset
        """
        # autotune computation
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        # define a dataset of file paths
        dataset = tf.data.Dataset.list_files(os.path.join(directory, '*.wav'))
        # run the preprocessing via map
        dataset = dataset.map(self.preprocessing_wrapper,
                              num_parallel_calls=AUTOTUNE)
        if save:
            # save dataset to disk
            name = 'dataset_'+os.path.split(directory)[1]
            path = os.path.join('_data', name)
            tf.data.experimental.save(dataset=dataset,
                                      path=path,
                                      compression='GZIP')
        return dataset

    def gen_tf_dataset_from_list(self, dirs: list, save=False, ds_name=None):
        """
        Takes a list of directories with samples to create a tensorflow
        dataset.

        Parameters
        ----------
        directory : list
            List of directories
        save : boolean
        Set to True if datset should be saved, by default False.
        ds_name : str
            Name of dataset

        Returns
        -------
        tensorflow.data.Dataset
            Dataset
        """
        if save and not ds_name:
            raise ValueError("Name must be given, if dataset is saved.")
        dataset = self.gen_tf_dataset(dirs[0])
        # For list of dirs:
        for dir in dirs[1:]:
            dataset = dataset.concatenate(self.gen_tf_dataset(dir))
        # Save
        if save:
            name = 'dataset_'+ds_name
            path = os.path.join('_data', name)
            tf.data.experimental.save(dataset=dataset,
                                    path=path,
                                    compression='GZIP')
        return dataset

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
            codec_list = list(config.keys())[4:]
            # Replace 'db_format' by 'uncompr_wav' and return
            return ['uncompr_wav' if i == 'db_format' else i for i in
                    codec_list]