# %%
from DLNet_functions import preprocess_wrapper
from pathlib import Path
import json
import glob
import tensorflow as tf
assert tf.__version__ >= "2.0"
# autotune computation
AUTOTUNE = tf.data.experimental.AUTOTUNE

# %%
# Create Config for preprocessing and pipeline parameters

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
