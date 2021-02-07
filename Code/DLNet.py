# %%
## Import necessary packages
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

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
## Create Config for preprocessing and pipeline parameters

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
classes = sorted(set([Path(f).parts[-1] for f in folders]))
config['classes'] = classes

# save number of frames from length in samples divided by fft hop length
config['n_frames'] = int(config['sr']*config['audio_length']/config['hop_length']) + 1

# save input shape for model
config['input_shape'] = (config['n_mels'], config['n_frames'], 1)

# save config 
with open('DLNet_config.json', 'w+') as fp:
    json.dump(config, fp, sort_keys=True, indent=4)

# %%
# VISUALIZE WAVEFORMS
# get all wav files
fps = glob.glob('_data/*_wav/_*/*.wav', recursive=True)
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
# VISUALIZE SPECTROGRAMS
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


# %%
# Preprocessing
# generate mel-filter matrix
mel_filter = librosa.filters.mel(config['sr'], 
                                 config['n_fft'], 
                                 n_mels=config['n_mels'], 
                                 fmin=0.0, 
                                 fmax=None, 
                                 htk=False, 
                                 norm='slaney', 
                                 dtype=np.float32)


# Groundtruth extraction from folder name
def folder_name_to_one_hot(file_path):
    
    label = Path(file_path).parts[-3]
    label_idx = classes.index(label)
    
    # get one hot encoded array
    one_hot = tf.one_hot(label_idx, len(config['classes']), on_value=None, off_value=None, 
                         axis=None, dtype=tf.uint8, name=None)
    return one_hot

def load_and_preprocess_data(file_path):
    # path string is saved as byte array in tf.data.dataset -> convert back to str
    if type(file_path) is not str:
        file_path = file_path.numpy()
        file_path = file_path.decode('utf-8')
    
    
    # load audio data 
    y, _ = librosa.core.load(file_path, sr=config['sr'], mono=config['mono'], offset=0.0, duration=None, 
                             dtype=np.float32, res_type='kaiser_best')



    # calculate stft from audio data
    stft = librosa.core.stft(y, n_fft=config['n_fft'], hop_length=config['hop_length'], 
                             win_length=config['win_length'], window=config['window'], 
                             center=config['center'], dtype=np.complex64, pad_mode=config['pad_mode'])

    # calculate magnitude and scale to dB
    magspec = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    # filter stft with mel-filter
    mel_spec = mel_filter.dot(np.abs(stft).astype(np.float32) ** config['power'])
    
    # add channel dimension for conv layer compatibility
    magspec = np.expand_dims(magspec, axis=-1)

    # add channel dimension for conv layer compatibility
    mel_spec = np.expand_dims(mel_spec, axis=-1)
    
    # get ground truth from file_path string
    one_hot = folder_name_to_one_hot(file_path)
    
    return magspec, one_hot

# there is a TF bug where we get an error if the size of the tensor from a py.function is not set manualy
# when called from a map()-function.
def preprocessing_wrapper(file_path):
    magspec, one_hot = tf.py_function(load_and_preprocess_data, [file_path], [tf.float32, tf.uint8])
    
    magspec.set_shape([config['n_fft'], config['n_frames'], 1])
    one_hot.set_shape([len(config['classes'])])
    return magspec, one_hot


# %%
# Offline preprocessing -> ability to save preprocessed dataset tensors on harddrive

# autotune computation
AUTOTUNE = tf.data.experimental.AUTOTUNE

# folder with the training data
wavset = '_data/*_wav/_*/*.wav'
# define a dataset of file paths
dataset = tf.data.Dataset.list_files(wavset)
# run the preprocessing via map
dataset = dataset.map(preprocessing_wrapper, num_parallel_calls=AUTOTUNE)
# save dataset to disk
#!rm -rf ./_data/processed/train
tf.data.experimental.save(dataset=dataset, path=f'./_data/dataset', compression='GZIP')
# show tensor types and shapes in dataset (we need this to load the dataset later)
print(dataset.element_spec)



# %%
# load a dataset from disk

dataset = tf.data.experimental.load(f'./_data/dataset', 
                                    (tf.TensorSpec(shape=(config['n_mels'], config['n_frames'], 1), dtype=tf.float32, name=None), 
                                     tf.TensorSpec(shape=(len(config['classes']),), dtype=tf.uint8, name=None)), 
                                    compression='GZIP')

# shuffle before splitting in train and eval dataset
dataset = dataset.shuffle(buffer_size=np.int(len(dataset)))
dataset = dataset.cache()

# take first 80% from dataset
train_dataset = dataset.take(np.int(4/5*len(dataset)))
train_dataset = train_dataset.shuffle(buffer_size= np.int(len(dataset)))
train_dataset = train_dataset.batch(64)
train_dataset = train_dataset.prefetch(AUTOTUNE)


# take last 20% samples from dataset 
eval_dataset = dataset.skip(np.int(4/5*len(dataset))).batch(64).prefetch(AUTOTUNE)


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

# Define metrics
metrics = [tf.keras.metrics.TrueNegatives(),
           tf.keras.metrics.TruePositives(),
           tf.keras.metrics.FalseNegatives(),
           tf.keras.metrics.FalsePositives(),
           tf.keras.metrics.Precision(),
           tf.keras.metrics.Recall(),
           tf.keras.metrics.CategoricalAccuracy()
          ]

# compile model
n_epochs = 10
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# fit model
history = model.fit(train_dataset, epochs=n_epochs, validation_data=eval_dataset)

# %% 
# setup plot
fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(16,4))

# plot loss
ax[0].plot(range(n_epochs), history.history['loss'])
ax[0].plot(range(n_epochs), history.history['val_loss'])
ax[0].set_ylabel('loss'), ax[0].set_title('train_loss vs val_loss')

# plot accuracy
ax[1].plot(range(n_epochs), history.history['categorical_accuracy'])
ax[1].plot(range(n_epochs), history.history['val_categorical_accuracy'])
ax[1].set_ylabel('accuracy'), ax[1].set_title('train_acc vs val_acc')

# plot adjustement
for a in ax:
    a.grid(True)
    a.legend(['train','val'], loc=4)
    a.set_xlabel('num of Epochs')
    
plt.show()

# %%
