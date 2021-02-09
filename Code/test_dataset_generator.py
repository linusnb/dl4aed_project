#%%
# import matplotlib.pyplot as plt
# import numpy as np
# # from librosa import display
# from pathlib import Path
import pytest
import glob
from dataset_generator import Dataset
import librosa
import os

@pytest.fixture
def datapath():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'test_dataset/_data')


@pytest.fixture
def database(datapath):
    return Dataset(os.path.join(datapath, "dataset_config.json"), 'new_ds')


def test_init(database):
    assert isinstance(database, Dataset)


def test_encode_audio(database, datapath):
    input = os.path.join(datapath, 'new_ds', "seed_files/test_file.wav")
    output_path = os.path.join(datapath, 'new_ds', "compressed_wav/mp3_160k")
    if not os.path.isdir(output_path):
        os.makedirs(os.path.join(output_path))
    output = os.path.join(output_path, "test_file.mp3")
    codec = database._codec_settings['mp3_160k']
    codec_out = codec['codec']
    channels_out = codec['channels']
    bitrate_out = codec['bitrate']
    sampling_rate_out = codec['sampling_rate']
    database.encode(input, output, codec_out, channels_out, bitrate_out,
                    sampling_rate_out)
    assert os.path.isfile(output) is True
    assert codec_out == 'libmp3lame'
    assert channels_out == 2
    assert bitrate_out == 160000
    assert sampling_rate_out == 44100


def test_decode_audio(database, datapath):
    input = os.path.join(datapath, "compressed_wav/mp3_160k/test_file.mp3")
    output = os.path.join(datapath, "compressed_wav/mp3_160k/test_file.wav")
    database.decode(input, output)
    assert os.path.isfile(output) == True
    # Read output file
    data_out, sr_out = librosa.load(output,
                                    sr=database._db_format['sampling_rate'],
                                    mono=False)
    assert sr_out == database._db_format['sampling_rate']
    assert data_out.shape[0] == database._db_format['channels']


def test_split_audio(database, datapath):
    input = os.path.join(datapath, "compressed_wav/mp3_160k/test_file.wav")
    # Generate chunks
    database.split_audio(input)
    # Read chunks:
    data_path = os.path.join(datapath,
                             "compressed_wav/mp3_160k/test_file_c*.wav")
    fps = glob.glob(data_path, recursive=True)
    for file in fps:
        chunk, sr = librosa.core.load(file,
                                      sr=database._db_format['sampling_rate'],
                                      mono=False)
        assert chunk.shape[1] == database._db_format['sampling_rate']*database._chunk_size
        assert chunk.shape[0] == database._db_format['channels']
        # Clean up
        os.remove(file)

def test_add_item(database, datapath):
    new_file = os.path.join(datapath, "seed_files/test_file.wav")
    expt_new_file = os.path.join(datapath, "compressed_wav/mp3_128k/test_file_c1.wav")
    os.chdir(datapath)
    database.add_item(new_file, 'mp3_128k')
    assert os.path.isfile(expt_new_file)
    # Read chunks:
    data_path = os.path.join(datapath,
                             "compressed_wav/mp3_128k/test_file_c*.wav")
    fps = glob.glob(data_path, recursive=True)
    for file in fps:
        # Clean up
        os.remove(file)



# #%%
# # Get files
# fps = glob.glob('_data/compressed_wav/**/*.wav', recursive=True)
# fps_random = []
# np.random.seed(9)

# # setup subplot 
# nrows, ncols = 2, 2
# fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 6))

# # plot some audio waveforms
# for r in range(nrows):
#     for c in range(ncols):
#         fp_random = fps[np.random.randint(len(fps))]
#         audio, sr = librosa.core.load(fp_random, sr=44100)
#         ax[r][c].plot(audio, c='k')
#         print(np.size(audio))
#         # ax[r][c].axis('off')
#         ax[r][c].set_title(Path(fp_random).parts[-2:])
#         if r == 0:
#             ax[r][c].set_xticks([])
#         # save random audio filepaths
#         fps_random.append(fp_random)
# #%%
# # Get files
# fps = glob.glob('_data/uncompr_wav/**/*.wav', recursive=True)
# fps_random = []
# # np.random.seed(9)

# # setup subplot 
# nrows, ncols = 2, 2
# fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 6))

# # plot some audio waveforms
# for r in range(nrows):
#     for c in range(ncols):
#         fp_random = fps[np.random.randint(len(fps))]
#         audio, sr = librosa.core.load(fp_random, sr=44100)
#         ax[r][c].plot(audio, c='k')
#         print(np.size(audio))
#         # ax[r][c].axis('off')
#         ax[r][c].set_title(Path(fp_random).parts[-2:])
#         if r == 0:
#             ax[r][c].set_xticks([])
#         # save random audio filepaths
#         fps_random.append(fp_random)
# # %%
# fps = glob.glob('_data/uncompr_wav/*.wav', recursive=True)
# sr = 44100
# audio_duration = np.array([np.size(librosa.core.load(fp_i, sr=sr)[0]) for fp_i in fps])


# # %%
# fps = glob.glob('_data/compressed_wav/**/*.wav', recursive=True)
# sr = 44100
# compr_audio_duration = np.array([np.size(librosa.core.load(fp_i, sr=sr)[0]) for fp_i in fps])

