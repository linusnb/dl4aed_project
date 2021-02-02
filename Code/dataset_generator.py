import os
import subprocess
import json
import numpy as np
import soundfile as sf
import glob
import matplotlib.pyplot as plt
class Dataset:

    def __init__(self, json_file):
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
        root_dir, _ = os.path.split(json_file)
        # Get seed path:
        self._seed_dir = os.path.join(root_dir, config['reference_audio_path'])
        # Get location of uncompressed waves
        self._uncompr_dir = os.path.join(root_dir,
                                         config['uncompr_audio_path'])
        # Get location of compressed waves
        self._compr_dir = os.path.join(root_dir,
                                       config['compressed_audio_path'])
        # Get chunk size
        self._chunk_size = config['segment_length']
        # Get list of codec settings:
        self._key_list = list(config.keys())
        self._codec_list = self._key_list[4:]
        # Get dict with codec settings:
        self._codec_settings = {key: value for key, value in config.items()
                                if key in self._codec_list}
        self._db_format = self._codec_settings['db_format']

    def encode(self, input, output, codec_out, channels_out, bitrate_out,
               sampling_rate_out):
        """encode_audio
        Encodes audio into specified format by codec_out.

        Parameters
        ----------
        input : str
            Input file with complete path and file extension
        output : str
            Output file with complete path and file extension
        codec_out : str
            Encoder
        channels_out : int
            Number of channels of codec
        bitrate_out : int
            Bitrate for codec setting
        sampling_rate_out : int
            Sampling rate of codec

        Returns
        -------
        subprocess
            Subprocess running an ffmpeg command.
        """
        return subprocess.run([f"ffmpeg -y -i {input} \
                              -acodec {codec_out} \
                              -ac {channels_out} \
                              -ab {bitrate_out} \
                              -ar {sampling_rate_out} \
                              {output}"],
                              capture_output=True, shell=True)

    def decode(self, input, output):
        """decode
        Decodes input to db_format and writes file to output.

        Parameters
        ----------
        input : str
            Input file with complete path and file extension
        output : str
            Output file with complete path without file extension

        Returns
        -------
        [type]
            Command prompt for ffmpeg
        """
        return subprocess.run([f"ffmpeg -i {input} \
                               -acodec {self._db_format['codec_ffmpeg']} \
                               -ac {self._db_format['channels']} \
                               -ar {self._db_format['sampling_rate']} \
                               {output}"],
                              capture_output=True, shell=True)

    def split_audio(self, input):
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
        # Update number of sample
        n_samples = data.shape[0]
        # Reshape cropped data into arrays of chunk_length:
        chunk_arr = np.reshape(data, (n_chunks, int(n_samples/n_chunks),
                               n_channels))
        # Filter depending on RMS value: chunk rms > .5*data_rms
        filter = np.sqrt(np.mean(chunk_arr**2, axis=(1, 2))) > \
                        (.5*np.sqrt(np.mean(chunk_arr**2)))
        # Output file path:
        # Remove file extension from input file path
        in_file_path, _ = os.path.splitext(input)
        # Only return chunks with rms greater then half of average rms
        for idx, chunk in enumerate(chunk_arr[filter]):
            n_file_name = f"{in_file_path}" + f"_c{idx+1}.wav"
            # Write wav file:
            sf.write(n_file_name, chunk, samplerate,
                     self._db_format['codec_sf'])

    def add_item(self, input_file, codec_specifier):
        """add_item
        Add item to the dataset with specified codec. If file exists in seed
        list, file will not be added.

        Parameters
        ----------
        input_file : str
            Path to new input file.
        codec_specifier : str
            Codec specifier, item from list of possible codecs.

        Raises
        ------
        ValueError
            [description]
        """
        if codec_specifier not in self._codec_list:
            raise ValueError("Invalid codec specifier.")
        # Get filename of input file
        # Remove path extension of input file
        _, in_filename_wav = os.path.split(input_file)
        # Check if filename is in seed_list
        with open(os.path.join(self._seed_dir, "seed_list.txt")) as f:
            lines = f.readlines()
            seed_name = in_filename_wav+'\n'
            if seed_name in lines:
                raise ValueError(f"File with same name alread in seed \
                                 list:{in_filename_wav}")
        # Remove file extension from input file name
        in_filename, _ = os.path.splitext(in_filename_wav)
        # if encoder is neccessary
        if codec_specifier != 'db_format':
            codec = self._codec_settings[codec_specifier]
            # Output path
            enc_output_path = os.path.join(self._compr_dir, codec_specifier)
            # If compressed output directory not existing -> make directory
            if not os.path.isdir(enc_output_path):
                os.makedirs(os.path.join(enc_output_path))
            # Label data
            enc_output_file = os.path.join(enc_output_path, in_filename + '.' +
                                           codec['format'])
            # encode to codec_specifier
            self.encode(input_file, enc_output_file, codec['codec'],
                        codec['channels'], codec['bitrate'],
                        codec['sampling_rate'])
            # decode to db_format
            dec_output_file = os.path.join(enc_output_path, in_filename_wav)
            self.decode(enc_output_file, dec_output_file)
            # Remove encoded file:
            os.remove(enc_output_file)
        # if no encoder neccessary
        else:
            dec_output_file = os.path.join(self._uncompr_dir, in_filename_wav)
            self.decode(input_file, dec_output_file)
        # Split decoded file:
        self.split_audio(dec_output_file)
        # Append seed name to text file
        with open(os.path.join(self._seed_dir, "seed_list.txt"), 'a') as f:
            f.write(in_filename_wav)
            f.write("\n")
        # Delete seed:
        os.remove(input_file)

    def add_dir(self, input_dir, codec_specifier):
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
        fps = glob.glob(input_dir+'/**/*.wav', recursive=True)
        for file in fps:
            self.add_item(file, codec_specifier)

    def get_codec_specifiers(self):
        """get_codec_specifiers
        Returns all possible codec specifiers.

        Returns
        -------
        list
            List of all possible codec specifiers.
        """
        return self._codec_list

    def get_stats(self):
        n_chunk_arr = np.zeros(len(self._codec_list))
        for idx, codec in enumerate(self._codec_list):
            if codec != 'db_format':
                path = os.path.join(self._compr_dir, codec, '**/*.wav')
                n_chunk_arr[idx] = len(glob.glob(path, recursive=True))
            else:
                path = os.path.join(self._uncompr_dir, '**/*.wav')
                n_chunk_arr[idx] = len(glob.glob(path, recursive=True))
        n_total_chunks = np.sum(n_chunk_arr)
        plt.bar(self._codec_list, n_chunk_arr)
        plt.xticks(rotation=45)
        return dict(zip(self._codec_list, n_chunk_arr.T)), n_total_chunks
