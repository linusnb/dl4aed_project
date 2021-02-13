# %%
from pathlib import Path
import json
import glob
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
CALCULATE_MEL = False
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
              'calculate_mel': CALCULATE_MEL,
              'filter_signal': False,
              'random_seed': RANDOM_SEED
              }


# save number of frames from length in samples divided by fft hop length
config['n_frames']: int = int(
    config['sr']*config['audio_length']/config['hop_length']) + 1

# save input shape for model
if CALCULATE_MEL:
    config['input_shape']: (int, int, int) = (config['n_mels'],
                                              config['n_frames'], 1)
else:
    config['input_shape']: (int, int, int) = (int(config['n_fft']/2 + 1),
                                              config['n_frames'], 1)


# save config
with open('DLNet_config.json', 'a') as fp:
    json.dump(config, fp, sort_keys=True, indent=4)

# Creater wrapper object:
ds_config: str = 'dl4aed_project/Code/_data/dataset_config.json'
wrapper: PreprocessWrapper = PreprocessWrapper(config, ds_config)


# %%
# Create dataset from MedleyDB
train_aac, test_aac = wrapper.tf_dataset_from_codec('_data/MedleyDB/compressed_wav/ogg_vbr')
train_wav, test_wav = wrapper.tf_dataset_from_codec('_data/MedleyDB/uncompr_wav')
test_dataset = test_wav.concatenate(test_aac)
train_dataset = train_wav.concatenate(train_aac)

# train_data = '/home/linus/tubCloud/Documents/MedleyDB_temp_olf_version/compressed_wav/mp3_32k/*.wav'
# more_data = '/home/linus/tubCloud/Documents/MedleyDB_temp_olf_version/uncompr_wav/*.wav'

# dataset = tf.data.Dataset.list_files(train_data)
# dataset = dataset.concatenate(tf.data.Dataset.list_files(more_data))
# dataset = dataset.map(wrapper.preprocessing_wrapper,
#                           num_parallel_calls=AUTOTUNE)                                

# %%
# VISUALIZE WAVEFORMS
# get all wav files
fps = glob.glob('_data/MedleyDB/compressed_wav/**/*.wav', recursive=True)
fps_random = []

# setup subplot
nrows, ncols = 2, 2
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 6))

# plot some audio waveforms
for r in range(nrows):
    for c in range(ncols):
        fp_random = fps[np.random.randint(0, len(fps))]
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
specs_c = [None]*4
specs_uc = [None]*4
uncompr_file_path = [None]*4
for i, file in enumerate(fps_random):
    spec_c, _ = wrapper.load_and_preprocess_data(file)
    path, name = os.path.split(file)
    _, folder = os.path.split(path)
    uncompr_file = os.path.join(DATA_PATH, 'MedleyDB', 'uncompr_wav',
                                folder, name)
    uncompr_file_path[i] = uncompr_file
    spec_uc, _ = wrapper.load_and_preprocess_data(uncompr_file)
    specs_c[i] = spec_c[:, :, 0]
    specs_uc[i] = spec_uc[:, :, 0]

plt.figure(figsize=(15, 10))
plt.subplot(4, 2, 1)
librosa.display.specshow(specs_c[0], sr=config['sr'],
                         hop_length=config['hop_length'],
                         y_axis='log')
plt.title(fps_random[0])
plt.colorbar(format='%+2.0f dB')
plt.subplot(4, 2, 2)
librosa.display.specshow(specs_uc[0], sr=config['sr'],
                         hop_length=config['hop_length'],
                         y_axis='log')
plt.title(uncompr_file_path[0])
plt.colorbar(format='%+2.0f dB')
plt.subplot(4, 2, 3)
librosa.display.specshow(specs_c[1], sr=config['sr'],
                         hop_length=config['hop_length'],
                         y_axis='log')
plt.title(fps_random[1])
plt.colorbar(format='%+2.0f dB')
plt.subplot(4, 2, 4)
librosa.display.specshow(specs_uc[1], sr=config['sr'],
                         hop_length=config['hop_length'],
                         y_axis='log')
plt.title(uncompr_file_path[1])
plt.colorbar(format='%+2.0f dB')
plt.subplot(4, 2, 5)
librosa.display.specshow(specs_c[2], sr=config['sr'],
                         hop_length=config['hop_length'],
                         y_axis='log')
plt.title(fps_random[2])
plt.colorbar(format='%+2.0f dB')
plt.subplot(4, 2, 6)
librosa.display.specshow(specs_uc[2], sr=config['sr'],
                         hop_length=config['hop_length'],
                         y_axis='log')
plt.title(uncompr_file_path[2])
plt.colorbar(format='%+2.0f dB')
plt.subplot(4, 2, 7)
librosa.display.specshow(specs_c[3], sr=config['sr'],
                         hop_length=config['hop_length'],
                         x_axis='time',
                         y_axis='log')
plt.title(fps_random[3])
plt.colorbar(format='%+2.0f dB')
plt.subplot(4, 2, 8)
librosa.display.specshow(specs_uc[3], sr=config['sr'],
                         hop_length=config['hop_length'],
                         x_axis='time',
                         y_axis='log')
plt.title(uncompr_file_path[3])
plt.colorbar(format='%+2.0f dB')
plt.show()

# %% Prepare dataset
train_size = len(train_dataset)
test_size = len(test_dataset)
eval_size = int(.1*train_size)
batch_size = 64

# Shuffel train data:
train_dataset = train_dataset.shuffle(buffer_size=train_size)

# Split train into train and eval set:
eval_dataset = train_dataset.take(eval_size)
eval_dataset = eval_dataset.batch(batch_size).prefetch(AUTOTUNE)

# Train dataset
train_dataset = train_dataset.skip(eval_size)
train_dataset = train_dataset.shuffle(train_size - eval_size)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(AUTOTUNE)

# Prepare test dataset
test_dataset = test_dataset.batch(batch_size).prefetch(AUTOTUNE)

# # Split dataset in 80:20 (test:train)
# buff_size: int = len(dataset)
# train_size: int = int(.8*buff_size)
# test_size: int = buff_size - train_size
# # shuffle before splitting in train and eval dataset
# dataset = dataset.shuffle(buffer_size=buff_size)
# dataset = dataset.cache()

# # take first 80% from dataset
# train_dataset = dataset.take(train_size)
# train_dataset = train_dataset.shuffle(buffer_size=train_size)
# train_dataset = train_dataset.batch(64)
# train_dataset = train_dataset.prefetch(AUTOTUNE)

# # take last 20% samples from dataset
# test_dataset = dataset.skip(test_size).shuffle(test_size)
# test_dataset = test_dataset.batch(64).prefetch(AUTOTUNE)
# eval_dataset = test_dataset


# %%
# Build model architecture
MODEL_TYPE = ModelType.BASIC_CNN
model_builder = ModelBuilder(MODEL_TYPE, config['input_shape'], 
                             config['classes'])
model = model_builder.get_model()

# Define metrics
metrics = [tf.keras.metrics.TrueNegatives(),
           tf.keras.metrics.TruePositives(),
           tf.keras.metrics.FalseNegatives(),
           tf.keras.metrics.FalsePositives(),
           tf.keras.metrics.Precision(),
           tf.keras.metrics.Recall(),
           tf.keras.metrics.CategoricalAccuracy()
           ]
# %%
# compile model
n_epochs = 2
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# fit model
history = model.fit(train_dataset, epochs=n_epochs,
                    validation_data=eval_dataset)

model.evaluate(test_dataset,batch_size=64)

# %%
# setup plot
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))

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
