# %%
from pathlib import Path
import json
import glob
from re import VERBOSE
import numpy as np
import librosa
import os
from librosa import display
import matplotlib.pyplot as plt
from wrapper_functions import PreprocessWrapper
import tensorflow as tf
from time import strftime
assert tf.__version__ >= "2.0"
# autotune computation
AUTOTUNE = tf.data.experimental.AUTOTUNE
RANDOM_SEED = 10
DATA_PATH = '_data'

# %%
# Create Config for preprocessing and pipeline parameters

# if true analysis is conducted with mel-spectrograms, if false with "full"
# spectrograms
time_stamp = f'{strftime("%d_%m_%Y_%H_%M")}'

config: {} = {'time_stamp': time_stamp,
              'sr': 44100,
              'audio_length': 1,
              'mono': True,
              'n_mels': 64,
              'n_fft': 1024,
              'hop_length': 256,
              'win_length': 512,
              'window': 'hamm',
              'center': True,
              'pad_mode': 'reflect',
              'power': 2.0,
              'calculate_mel': False,
              'filter_signal': True,
              'filter_config': ['high', 4000],
              'random_seed': 10,
              'binary': False
              }


# save number of frames from length in samples divided by fft hop length
config['n_frames']: int = int(
    config['sr']*config['audio_length']/config['hop_length']) + 1

# Creater wrapper object:
ds_config: str = os.path.join(DATA_PATH, 'dataset_config.json')
wrapper: PreprocessWrapper = PreprocessWrapper(config, ds_config)

# %% Saving dataset
train_dataset, test_dataset = wrapper.tf_dataset_from_database(
                                    os.path.join(DATA_PATH, 'VinylDB'),
                                    save=True)

# %%
