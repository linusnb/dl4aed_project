from pathlib import Path
import os
import tensorflow as tf
from librosa import filters
from librosa import core
import numpy as np


class preprocess_wrapper:

    def __init__(self, config: dict):
        self._config = config
        self._mel_filter = filters.mel(self._config['sr'],
                                       self._config['n_fft'],
                                       n_mels=config['n_mels'],
                                       norm='slaney')

    # Groundtruth extraction from folder name
    def folder_name_to_one_hot(self, file_path):
        for label_idx, label in enumerate(self._config['classes']):
            if label in Path(file_path).parts:
                # get one hot encoded array
                return tf.one_hot(label_idx, len(self._config['classes']),
                                dtype=tf.uint8)
        raise ValueError("Data cannot be labeled.")

    def load_and_preprocess_data(self, file_path: str):
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
        mel_spec, one_hot = tf.py_function(func=self.load_and_preprocess_data,
                                           inp=[file_path],
                                           Tout=[tf.float32, tf.uint8])

        # mel_spec.set_shape([self._config['n_mels'], self._config['n_frames'], 1])
        # one_hot.set_shape([len(self._config['classes'])])
        mel_spec = tf.ensure_shape(mel_spec, self._config['input_shape'])
        one_hot = tf.ensure_shape(one_hot, len(self._config['classes']))
        return mel_spec, one_hot

    def gen_tf_dataset(self, directory: str):
        # autotune computation
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        # define a dataset of file paths
        dataset = tf.data.Dataset.list_files(os.path.join(directory, '*.wav'))
        # run the preprocessing via map
        dataset = dataset.map(self.preprocessing_wrapper,
                              num_parallel_calls=AUTOTUNE)
        # save dataset to disk
        name = 'dataset_'+os.path.split(directory)[1]
        path = os.path.join('_data', name)
        tf.data.experimental.save(dataset=dataset,
                                  path=path,
                                  compression='GZIP')
        return dataset

    def load_tf_dataset(self, directory: str):
        return tf.data.experimental.load(directory,
                        (tf.TensorSpec(self._config['input_shape'], dtype=tf.float32, name=None),
                         tf.TensorSpec(len(self._config['classes']), dtype=tf.uint8, name=None)),
                        compression='GZIP')
