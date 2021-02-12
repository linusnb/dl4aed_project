import pytest
from DLNet_functions import PreprocessWrapper
import glob
import tensorflow as tf
import os


@pytest.fixture
def wrapper():
    CALCULATE_MEL = False
    config: {} = {'sr': 44100,
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
                  'calculate_mel': CALCULATE_MEL,
                  'filter_signal': False
                  }

    # save number of frames from length in samples divided by fft hop length
    config['n_frames']: int = int(
        config['sr']*config['audio_length']/config['hop_length']) + 1

    # save input shape for model
    if CALCULATE_MEL:
        config['input_shape']: (int, int, int) = (config['n_mels'],
                                                  config['n_frames'], 1)
    else:
        config['input_shape']: (int, int, int) = (config['n_fft'], 
                                                  config['n_frames'], 1)
    ds_config = "dl4aed_project/Code/_data/dataset_config.json"
    return PreprocessWrapper(config, ds_config, binary=False)


def test_init():
    CALCULATE_MEL = True
    config: {} = {'sr': 44100,
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
                  'calculate_mel': CALCULATE_MEL,
                  'filter_signal': False
                  }

    # save number of frames from length in samples divided by fft hop length
    config['n_frames']: int = int(
        config['sr']*config['audio_length']/config['hop_length']) + 1

    # save input shape for model
    if CALCULATE_MEL:
        config['input_shape']: (int, int, int) = (config['n_mels'],
                                                  config['n_frames'], 1)
    else:
        config['input_shape']: (int, int, int) = (config['n_fft'], 
                                                  config['n_frames'], 1)
    ds_config = "dl4aed_project/Code/_data/dataset_config.json"
    assert isinstance(PreprocessWrapper(config, ds_config, binary=False),
                      PreprocessWrapper) is True


def test_folder_name_to_one_hot(wrapper):
    file_path = '_data/uncompr_wav/CatMartino_IPromise_MIX_c1.wav'
    one_hot = wrapper.folder_name_to_one_hot(file_path)
    assert isinstance(one_hot, tf.Tensor) is True
    wrong_file_path = '_data/nolabel/CatMartino_IPromise_MIX_c1.wav'
    with pytest.raises(Exception) as error:
        assert wrapper.folder_name_to_one_hot(wrong_file_path)
    assert str(error.value) == 'Data cannot be labeled.'


def test_load_and_preprocess_data(wrapper):
    file_path = '/home/linus/tubCloud/Documents/3.Semester/DL4AED/dl4aed_project/Code/_data/MedleyDB/compressed_wav/aac_128/1/1.wav'
    spec, one_hot = wrapper.load_and_preprocess_data(file_path)
