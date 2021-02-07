# %%
from DLNet_functions import preprocess_wrapper
from pathlib import Path
import json
import glob
import tensorflow as tf
assert tf.__version__ >= "2.0"
# autotune computation
AUTOTUNE = tf.data.experimental.AUTOTUNE

import numpy as np
import soundfile as sf
import os
import glob
import tqdm
import json
import librosa
from librosa import display
from pathlib import Path
import IPython.display as pd
import matplotlib.pyplot as plt

# %%
# Create Config for preprocessing and pipeline parameters

config = {'sr': 44100,
          'audio_length': 1,
          'mono': True,
          'n_mels': 64,
          'n_fft': 1024,
          'hop_length': 256,
          'win_length': 512,
          'window': 'hann',
          'center': True,
          'pad_mode': 'reflect',
          'power': 2.0,
          }

# save classes from foldernames
folders = glob.glob('_data/*_wav/')
config['classes'] = sorted(set([Path(f).parts[-1] for f in folders]))

# save number of frames from length in samples divided by fft hop length
config['n_frames'] = int(
    config['sr']*config['audio_length']/config['hop_length']) + 1

# save input shape for model
config['input_shape'] = (config['n_mels'], config['n_frames'], 1)

# save config
with open('DLNet_config.json', 'w+') as fp:
    json.dump(config, fp, sort_keys=True, indent=4)

#%%
# Generate mp3_32k dataset and uncompressed wav dataset:
wrapper = preprocess_wrapper(config)
dataset_mp3_32k = wrapper.gen_tf_dataset('_data/compressed_wav/mp3_32k')
dataset_uncompr = wrapper.gen_tf_dataset('_data/uncompr_wav')
dataset_full = wrapper.gen_tf_dataset('_data/*_wav/*')
dataset_combi = dataset_mp3_32k.concatenate(dataset_uncompr)
# %%
# get all wav files
fps = glob.glob('_data/*_wav/*/*.wav', recursive=True)
fps_random = []
np.random.seed(9)

# setup subplot 
nrows, ncols = 2, 2
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 6))

# plot some audio waveforms
for r in range(nrows):
    for c in range(ncols):
        fp_random = fps[np.random.randint(len(fps))]
        audio, sr = librosa.core.load(fp_random, sr=None)
        ax[r][c].plot(audio, c='k')
        # ax[r][c].axis('off')
        ax[r][c].set_title(Path(fp_random).parts[-2:])
        if r == 0:
            ax[r][c].set_xticks([])
        # save random audio filepaths
        fps_random.append(fp_random)

# %%
# setup subplot 
nrows, ncols = 4, 2
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 12))

# plot some audio waveforms
for i, fp_random in enumerate(fps_random):
    audio, sr = librosa.core.load(fp_random, sr=None)

    # calculate stft
    stft = librosa.stft(audio, n_fft=config['n_fft'], hop_length=config['hop_length'], win_length=config['win_length'])
    
    # calculate melspec
    melspec = librosa.feature.melspectrogram(audio, n_fft=config['n_fft'],
    hop_length=config['hop_length'], n_mels=config['n_mels'], fmax=int(config['sr']/2))
    melspec = librosa.amplitude_to_db(melspec, ref=np.max)

    # calculate magnitude and scale to dB
    magspec = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    # plot with librosa
    librosa.display.specshow(magspec, x_axis='time', y_axis='linear', sr=sr, hop_length=256, ax=ax[i][0])
    librosa.display.specshow(melspec, x_axis='time', y_axis='mel', sr=sr, hop_length=256, ax=ax[i][1])
    
    # adjustments
    # ax[i][1].set_yticks([])
    ax[i][1].set_ylabel(Path(fp_random).parts[-2], rotation=270, labelpad=20)
    ax[i][1].yaxis.set_label_position("right")
    
    # settings for all axises but bottom ones
    if not i == len(fps_random) - 1:
        ax[i][0].set_xticks([])
        ax[i][1].set_xticks([])
        ax[i][0].set_xlabel('')
        ax[i][1].set_xlabel('')
    
    # settings for upper axises
    if i == 0:
        ax[i][0].set_title('stft')
        ax[i][1].set_title('mel spectrogram')   

# adjust whitespace in between subplots        
plt.subplots_adjust(hspace=0.1, wspace=0.1)

print('Melspec shape: %s' % (str(melspec.shape)))
print('Stft shape: %s' % (str(stft.shape)))
print(f'Total data points in mel-spectrogram: {melspec.shape[0]*melspec.shape[1]}')
print(f'Total data points in stft-spectrogram: {stft.shape[0]*stft.shape[1]}')
print(f'-> Data Reduction by factor: {(stft.shape[0]*stft.shape[1]) / (melspec.shape[0]*melspec.shape[1])}')
print()

# show tensor types and shapes in dataset (we need this to load the dataset later)
print(dataset_full.element_spec)

# %%
# load a dataset from disk

dataset = tf.data.experimental.load(f'./_data/dataset',
                                    (tf.TensorSpec(shape=(config['n_mels'], config['n_frames'], 1), dtype=tf.float32, name=None),
                                     tf.TensorSpec(shape=(len(config['classes']),), dtype=tf.uint8, name=None)),
                                    compression='GZIP')

# shuffle before splitting in train and eval dataset
dataset = dataset.shuffle(buffer_size=18000)
dataset = dataset.cache()

# take first 80% from dataset
train_dataset = dataset.take(14400)
train_dataset = train_dataset.shuffle(buffer_size=18000)
train_dataset = train_dataset.batch(64)
train_dataset = train_dataset.prefetch(AUTOTUNE)


# take last 20% samples from dataset
eval_dataset = dataset.skip(14400).batch(64).prefetch(AUTOTUNE)


# %%
# create model architecture
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=config['input_shape']))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.GaussianDropout(0.25))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.GaussianDropout(0.25))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation="relu"))
model.add(tf.keras.layers.GlobalMaxPool2D())
model.add(tf.keras.layers.Dense(len(config['classes']), activation="sigmoid"))

# compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# fit model
model.fit(train_dataset, epochs=5)
model.evaluate(eval_dataset)


# %%
