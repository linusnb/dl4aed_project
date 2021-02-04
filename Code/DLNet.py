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
          'n_fft': 2048,
          'hop_length': 256,
          'win_length': 2048,
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

    # filter stft with mel-filter
    mel_spec = mel_filter.dot(np.abs(stft).astype(np.float32) ** config['power'])
    
    # add channel dimension for conv layer compatibility
    mel_spec = np.expand_dims(mel_spec, axis=-1)
    
    # get ground truth from file_path string
    one_hot = folder_name_to_one_hot(file_path)
    
    return mel_spec, one_hot

# there is a TF bug where we get an error if the size of the tensor from a py.function is not set manualy
# when called from a map()-function.
def preprocessing_wrapper(file_path):
    mel_spec, one_hot = tf.py_function(load_and_preprocess_data, [file_path], [tf.float32, tf.uint8])
    
    mel_spec.set_shape([config['n_mels'], config['n_frames'], 1])
    one_hot.set_shape([len(config['classes'])])
    return mel_spec, one_hot


# %%
# Offline preprocessing -> ability to save preprocessed dataset tensors on harddrive

# autotune computation
AUTOTUNE = tf.data.experimental.AUTOTUNE

# folder with the training data
wavset = '_data/*_wav/*/*.wav'
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
dataset = dataset.shuffle(buffer_size=18000)
dataset = dataset.cache()

# take first 80% from dataset
train_dataset = dataset.take(14400)
train_dataset = train_dataset.shuffle(buffer_size= 18000)
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



