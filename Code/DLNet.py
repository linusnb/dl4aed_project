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
from DLNet_functions import PreprocessWrapper
from models import ModelType, ModelBuilder
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

# save config
# with open('DLNet_config.json', 'a') as fp:
#     json.dump(wrapper.config, fp, sort_keys=True, indent=4)

# %% Saving dataset
train_dataset, test_dataset = wrapper.tf_dataset_from_database(
                                    os.path.join(DATA_PATH, 'MedleyDB'),
                                    save=True)

# %%
# VISUALIZE WAVEFORMS
# # get all wav files
# fps = glob.glob('_data/MedleyDB/compressed_wav/**/*.wav', recursive=True)
# fps_random = []

# # setup subplot
# nrows, ncols = 2, 2
# fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 6))

# # plot some audio waveforms
# for r in range(nrows):
#     for c in range(ncols):
#         fp_random = fps[np.random.randint(0, len(fps))]
#         audio, sr = librosa.core.load(fp_random, sr=None)
#         ax[r][c].plot(audio, c='k')
#         # ax[r][c].axis('off')
#         ax[r][c].set_title(Path(fp_random).parts[-2:])
#         if r == 0:
#             ax[r][c].set_xticks([])
#         # save random audio filepaths
#         fps_random.append(fp_random)

# %%
# # VISUALIZE SPECTROGRAMS
# # setup subplot
# specs_c = [None]*4
# specs_uc = [None]*4
# uncompr_file_path = [None]*4Bug fix in dataset_from_database

# plt.figure(figsize=(15, 10))
# plt.subplot(4, 2, 1)
# librosa.display.specshow(specs_c[0], sr=config['sr'],
#                          hop_length=config['hop_length'],
#                          y_axis='log')
# plt.title(fps_random[0])
# plt.colorbar(format='%+2.0f dB')
# plt.subplot(4, 2, 2)
# librosa.display.specshow(specs_uc[0], sr=config['sr'],
#                          hop_length=config['hop_length'],
#                          y_axis='log')
# plt.title(uncompr_file_path[0])
# plt.colorbar(format='%+2.0f dB')
# plt.subplot(4, 2, 3)
# librosa.display.specshow(specs_c[1], sr=config['sr'],
#                          hop_length=config['hop_length'],
#                          y_axis='log')
# plt.title(fps_random[1])
# plt.colorbar(format='%+2.0f dB')
# plt.subplot(4, 2, 4)
# librosa.display.specshow(specs_uc[1], sr=config['sr'],
#                          hop_length=config['hop_length'],
#                          y_axis='log')
# plt.title(uncompr_file_path[1])
# plt.colorbar(format='%+2.0f dB')
# plt.subplot(4, 2, 5)
# librosa.display.specshow(specs_c[2], sr=config['sr'],
#                          hop_length=config['hop_length'],
#                          y_axis='log')
# plt.title(fps_random[2])
# plt.colorbar(format='%+2.0f dB')
# plt.subplot(4, 2, 6)
# librosa.display.specshow(specs_uc[2], sr=config['sr'],
#                          hop_length=config['hop_length'],
#                          y_axis='log')
# plt.title(uncompr_file_path[2])
# plt.colorbar(format='%+2.0f dB')
# plt.subplot(4, 2, 7)
# librosa.display.specshow(specs_c[3], sr=config['sr'],
#                          hop_length=config['hop_length'],
#                          x_axis='time',
#                          y_axis='log')
# plt.title(fps_random[3])
# plt.colorbar(format='%+2.0f dB')
# plt.subplot(4, 2, 8)
# librosa.display.specshow(specs_uc[3], sr=config['sr'],
#                          hop_length=config['hop_length'],
#                          x_axis='time',
#                          y_axis='log')
# plt.title(uncompr_file_path[3])
# plt.colorbar(format='%+2.0f dB')
# plt.show()
