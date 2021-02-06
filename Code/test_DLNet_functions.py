import pytest
from DLNet_functions import preprocess_wrapper
import glob
import tensorflow as tf

@pytest.fixture
def wrapper():
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
    config['classes'] = ['compressed_wav', 'uncompr_wav']

    # save number of frames from length in samples divided by fft hop length
    config['n_frames'] = int(
        config['sr']*config['audio_length']/config['hop_length']) + 1

    # save input shape for model
    config['input_shape'] = (config['n_mels'], config['n_frames'], 1)
    return preprocess_wrapper(config)


def test_folder_name_to_one_hot(wrapper):
    file_path = '_data/uncompr_wav/CatMartino_IPromise_MIX_c1.wav'
    one_hot = wrapper.folder_name_to_one_hot(file_path)
    assert isinstance(one_hot, tf.Tensor) is True
    wrong_file_path = '_data/nolabel/CatMartino_IPromise_MIX_c1.wav'
    with pytest.raises(Exception) as error:
        assert wrapper.folder_name_to_one_hot(wrong_file_path)
    assert str(error.value) == 'Data cannot be labeled.'