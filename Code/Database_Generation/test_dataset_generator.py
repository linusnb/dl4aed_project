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
def db(datapath):
    return Dataset(os.path.join(datapath, "dataset_config.json"), 'new_ds')


def test_init(db):
    assert isinstance(db, Dataset)


def test_encode_audio(db, datapath):
    input = os.path.join(datapath, 'new_ds', "seed_files/test_file.wav")
    for codec in db._codec_list:
        output_path = os.path.join(datapath, 'new_ds', codec)
        if not os.path.isdir(output_path):
            os.makedirs(os.path.join(output_path))
        output = "test_file."+db._codec_settings[codec].get('format')
        output = os.path.join(output_path, output)
        db.encode(input, output, codec)
        out_wav, _ = os.path.splitext(output)
        out_wav = out_wav + '.wav'
        db.encode(output, out_wav, 'lossless_wav')
        assert os.path.isfile(output) is True
        assert os.path.isfile(out_wav) is True


def test_split_audio(db, datapath):
    input = os.path.join(datapath, 'new_ds', 'mp3_160k', 'test_file.wav')
    # Generate chunks
    db.split_audio(input)
    # Read chunks:
    data_path = os.path.join(datapath, 'new_ds', 'mp3_160k', '*.wav')
    fps = glob.glob(data_path, recursive=True)
    for file in fps:
        chunk, sr = librosa.core.load(file,
                                      sr=db._wav_format['sampling_rate'],
                                      mono=False)
        assert chunk.shape[1] == db._wav_format['sampling_rate']*db._chunk_size
        assert chunk.shape[0] == db._wav_format['channels']
        # Clean up
        os.remove(file)


def test_add_item(db, datapath):
    new_file = os.path.join(datapath, 'new_ds', 'seed_files', 'new_test_file.wav')
    expt_new_file = os.path.join(datapath, 'new_ds', '**', '4', '*.wav')
    db.add_item(new_file)
    fps = glob.glob(expt_new_file, recursive=True)
    # assert len(fps) == len(db._codec_list)*50
    # Read chunks:
    for file in fps:
        # Clean up
        os.remove(file)
