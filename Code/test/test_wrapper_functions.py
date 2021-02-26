from time import strftime
import os
import tensorflow as tf
import glob
import pytest
from wrapper_functions import PreprocessWrapper


@pytest.fixture
def wrapper():
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

    ds_config = "dl4aed_project/Code/_data/dataset_config.json"
    return PreprocessWrapper(config, ds_config)


def test_init():
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
    ds_config = "dl4aed_project/Code/_data/dataset_config.json"
    wrapper = PreprocessWrapper(config, ds_config)
    assert isinstance(wrapper, PreprocessWrapper) is True


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
