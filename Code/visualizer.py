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

DATA_PATH = '_data'
DATASET_CONFIG: str = os.path.join(DATA_PATH, 'dataset_config.json')
DATASET_NAME = 'Examples_10s'
config: {} = {'sr': 44100,
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

wrapper: PreprocessWrapper = PreprocessWrapper(config, DATASET_CONFIG)

# %%
# Get all wav files and spectrograms
fps = glob.glob(os.path.join(DATA_PATH, DATASET_NAME, '**/52/16.wav'),
                recursive=True)
labels = [Path(path).parts[-3] for path in fps]
specs = []

for i, file in enumerate(fps):
    # Audio data
    audio, _ = librosa.core.load(file, sr=44100)
    audio = audio[:44100]
    # Calculate stft
    stft = librosa.stft(audio, n_fft=config['n_fft'],
                        hop_length=config['hop_length'],
                        win_length=config['win_length'],
                        window=config['window'],
                        center=config['center'],
                        dtype=np.complex64,
                        pad_mode=config['pad_mode'])
    stft = librosa.amplitude_to_db(np.abs(stft),
                                   ref=np.max)
    specs.append(stft)


# %% 
# VISUALIZE SPECTROGRAMS
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Spectrograms', fontsize=16)
plt.subplot(4, 2, 1)
librosa.display.specshow(specs[0], sr=config['sr'],
                         hop_length=config['hop_length'],
                         x_axis='time',
                         y_axis='linear')
plt.title(labels[0])
plt.xticks
plt.colorbar(format='%+2.0f dB')
plt.subplot(4, 2, 2)
librosa.display.specshow(specs[1], sr=config['sr'],
                         hop_length=config['hop_length'],
                         x_axis='time',
                         y_axis='linear')
plt.title(labels[1])
plt.colorbar(format='%+2.0f dB')
plt.subplot(4, 2, 3)
librosa.display.specshow(specs[2], sr=config['sr'],
                         hop_length=config['hop_length'],
                         x_axis='time',
                         y_axis='linear')
plt.title(labels[2])
plt.colorbar(format='%+2.0f dB')
plt.subplot(4, 2, 4)
librosa.display.specshow(specs[3], sr=config['sr'],
                         hop_length=config['hop_length'],
                         x_axis='time',
                         y_axis='linear')
plt.title(labels[3])
plt.colorbar(format='%+2.0f dB')
plt.subplot(4, 2, 5)
librosa.display.specshow(specs[4], sr=config['sr'],
                         hop_length=config['hop_length'],
                         x_axis='time',
                         y_axis='linear')
plt.title(labels[4])
plt.colorbar(format='%+2.0f dB')
plt.subplot(4, 2, 6)
librosa.display.specshow(specs[5], sr=config['sr'],
                         hop_length=config['hop_length'],
                         x_axis='time',
                         y_axis='linear')
plt.title(labels[5])
plt.colorbar(format='%+2.0f dB')
plt.subplot(4, 2, 7)
librosa.display.specshow(specs[6], sr=config['sr'],
                         hop_length=config['hop_length'],
                         y_axis='linear',
                         x_axis='time')
plt.title(labels[6])
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.savefig('plots/example_specs_linear_scale.png', dpi=300)
