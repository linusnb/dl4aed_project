import os
from os.path import split
import subprocess
import json
import numpy as np
import soundfile as sf
import glob
import matplotlib.pyplot as plt
import random


class Dataset:

    def __init__(self, json_file: str, root_specifc: str) -> None:
        """__init__
        Constructs dataset object from json config file.

        Parameters
        ----------
        json_file : str
            Path to json config file.
        """
        # Read json Dataset_config
        with open(json_file, "r") as read_file:
            config = json.load(read_file)
        # Root dir:
        self._root_dir, _ = os.path.split(json_file)
        # Add root specification
        self._root_dir = os.path.join(self._root_dir, root_specifc)
        # Get seed path:
        self._seed_dir = os.path.join(self._root_dir,
                                      config['reference_audio_path'])
                                      # Get location of uncompressed waves
        self._uncompr_dir = os.path.join(self._root_dir,
                                         config['uncompr_audio_path'])
        # Get location of compressed waves
        self._compr_dir = os.path.join(self._root_dir,
                                       config['compressed_audio_path'])
        # Get chunk size
        self._chunk_size = config['segment_length']
        # Get number of chunks
        self._n_chunks = config['number of min chunks']
        # Get list of codec settings:
        self._key_list = list(config.keys())
        self._codec_list = self._key_list[5:]
        # Get dict with codec settings:
        self._codec_settings = {key: value for key, value in config.items()
                                if key in self._codec_list}
        self._wav_format = self._codec_settings['uncompr_wav']

    def encode(self, input: str, output: str, codec_name: dict) -> bool:
        """encode_audio
        Encodes audio into specified format by codec_out.

        Parameters
        ----------
        input : str
            Input file with complete path and file extension
        output : str
            Output file with complete path and file extension
        Returns
        -------
        bool
            True, if file was created.
        """
        ffmpeg_command = self._codec_settings[codec_name].get('ffmpeg_command')
        subprocess.run(ffmpeg_command.format(input, output),
                       capture_output=True, shell=True)
        if os.path.isfile(output):
            return True
        else:
            raise ValueError(f'Failed to create encoded file {output}.')

    def split_audio(self, input: str) -> bool:
        """split_audio
        Splits audio file into chunks depending on size set in json config.
        Wav file is cropped to fit integer number of chunks.

        Parameters
        ----------
        input : str
            Path to wav file.
        """
        # Read input
        data, samplerate = sf.read(input)
        n_samples = data.shape[0]
        n_channels = data.shape[1]
        # Number of chunks
        n_chunks = int(np.floor(n_samples/(self._chunk_size*samplerate)))
        # Crop data
        data = data[:int(n_chunks*self._chunk_size*samplerate)]
        # Normalise data
        data /= np.max(np.abs(data))
        # Update number of sample
        n_samples = data.shape[0]
        # Reshape cropped data into arrays of chunk_length:
        chunk_arr = np.reshape(data, (n_chunks, int(n_samples/n_chunks),
                               n_channels))
        # Filter depending on RMS value: chunk rms > .5*data_rms
        filter = np.sqrt(np.mean(chunk_arr**2, axis=(1, 2))) > \
                        (.5*np.sqrt(np.mean(chunk_arr**2)))
        chunk_arr = chunk_arr[filter]
        if chunk_arr.shape[0] > self._n_chunks:
            # Pick random 50 samples from filtert array:
            chunk_idx = set(np.arange(chunk_arr.shape[0]))
            samples_idx = random.sample(chunk_idx, 50)
        else:
            # Delete input file:
            os.remove(input)
            return False
        # Output file path:
        # Remove file extension from input file path
        in_file_path, _ = os.path.splitext(input)
        out_file_path, _ = os.path.split(in_file_path)
        # Only return chunks with rms greater then half of average rms
        for idx, sample_idx in enumerate(samples_idx):
            n_file_name = os.path.join(out_file_path, f"{idx+1}.wav")
            # Write wav file:
            sf.write(n_file_name, chunk_arr[sample_idx, :], samplerate,
                     self._wav_format['codec_sf'])
        # Delete input file:
        os.remove(input)
        return True

    def add_item(self, input_file: str) -> None:
        """add_item
        Add item to the dataset with specified codec. If file exists in seed
        list, file will not be added.

        Parameters
        ----------
        input_file : str
            Path to new input file.

        Raises
        ------
        ValueError
            [description]
        """
        # Get filename of input file
        # Remove path extension of input file
        _, in_f_name_wav = os.path.split(input_file)
        # Remove file extension from input file name
        in_f_name, _ = os.path.splitext(in_f_name_wav)

        # Check if filename is in seed_list
        seed_file = os.path.join(self._seed_dir, "seed_list.txt")
        with open(seed_file) as f:
            lines = f.readlines()
            n_seeds = len(lines)
            seed_name = in_f_name_wav+'\n'
            if seed_name in lines:
                raise ValueError(f"File with same name alread in seed \
                                 list:{in_f_name_wav}")

        is_split = False
        # Iterate over codec list:
        for codec in self._codec_list:
            # Decoded output file path
            # Encode to codec_specifier
            if codec != 'uncompr_wav':
                # Output path: dataset_name/compressed_wav/codec_name/
                # seed_number
                enc_output_path = os.path.join(self._compr_dir, codec,
                                            str(n_seeds+1))
                # If compressed output directory not existing -> make directory
                if not os.path.isdir(enc_output_path):
                    os.makedirs(os.path.join(enc_output_path))
                # Encodec output file path
                enc_out_file = os.path.join(enc_output_path, in_f_name + '.' +
                                            self._codec_settings[codec]
                                            .get('format'))
                self.encode(input_file, enc_out_file, codec)
                # decode to wav
                dec_out_file = os.path.join(enc_output_path, in_f_name_wav)
                self.encode(enc_out_file, dec_out_file, 'uncompr_wav')
                # Remove encoded file:
                os.remove(enc_out_file)
            # Convert to wav
            else:
                dec_out_path = os.path.join(self._uncompr_dir, str(n_seeds+1))
                # If uncompressed dir does not exist:
                if not os.path.isdir(dec_out_path):
                    os.makedirs(dec_out_path)
                # Uncompressed file path:
                dec_out_file = os.path.join(dec_out_path, in_f_name_wav)
                self.encode(input_file, dec_out_file, 'uncompr_wav')
            # Split decoded file:
            is_split = self.split_audio(dec_out_file)
            if is_split:
                print(f'Added {dec_out_file} to {enc_output_path}.')

        if is_split:
            # Append seed name to text file
            with open(seed_file, 'a') as f:
                f.write(in_f_name_wav)
                f.write("\n")
            # Delete seed:
            os.remove(input_file)
            return True
        else:
            return(f'Item: {input_file} not added to dataset.')

    def add_dir(self, input_dir: str, codec_specifier: str) -> None:
        """add_dir
        Add a directory of wav files to the dataset with specified codec.

        Parameters
        ----------
        input_dir : str
            Path to input directory.
        codec_specifier : str
            Codec specifier, item from list of possible codecs.
        """
        # apply add_item on directory
        # Get all wav files in dir:
        fps = glob.glob(input_dir+'/*.wav', recursive=True)
        for file in fps:
            self.add_item(file, codec_specifier)

    def get_codec_specifiers(self) -> []:
        """get_codec_specifiers
        Returns all possible codec specifiers.

        Returns
        -------
        list
            List of all possible codec specifiers.
        """
        return self._codec_list

    def get_stats(self) -> {}:
        n_chunk_arr = np.zeros(len(self._codec_list))
        for idx, codec in enumerate(self._codec_list):
            if codec != 'uncompr_wav':
                path = os.path.join(self._compr_dir, codec, '**/*.wav')
                n_chunk_arr[idx] = len(glob.glob(path, recursive=True))
            else:
                path = os.path.join(self._uncompr_dir, '**/*.wav')
                n_chunk_arr[idx] = len(glob.glob(path, recursive=True))
        n_total_chunks = np.sum(n_chunk_arr)
        print(f'Number of total chunks:{n_total_chunks}')
        plt.bar(self._codec_list, n_chunk_arr)
        plt.xticks(rotation=45)
        plt.title(f'Dataset in {self._root_dir}')
        plt.show()
        return dict(zip(self._codec_list, n_chunk_arr.T)), n_total_chunks
