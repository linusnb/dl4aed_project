# %%
from wrapper_functions import PreprocessWrapper
import os
import tensorflow as tf
from time import strftime
assert tf.__version__ >= "2.0"

# set path to raw dataset folder
DATA_PATH = '_data'
# Name of raw dataset folder
DATASET_NAME = 'VinylDB'
# Path to raw dataset config
ds_config: str = os.path.join(DATA_PATH, 'dataset_config.json')

# %%
# Create Config for preprocessing and pipeline parameters
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
wrapper: PreprocessWrapper = PreprocessWrapper(config, ds_config)

# Create new dataset from raw database
train_dataset, test_dataset = wrapper.tf_dataset_from_database(
                                    os.path.join(DATA_PATH, DATASET_NAME),
                                    save=True)
