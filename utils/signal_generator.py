import os

import tensorflow as tf
from scipy import signal


class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
                 num_mel_bins=None, lower_frequency=None, upper_frequency=None,
                 num_coefficients=None, mfcc=False, resampling_rate=None, seconds=4):

        self.labels = labels

        self.sampling_rate = sampling_rate
        self.resampling_rate = resampling_rate

        self.frame_length = frame_length
        self.frame_step = frame_step

        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency

        self.num_mel_bins = num_mel_bins
        self.num_coefficients = num_coefficients

        num_spectrogram_bins = (frame_length) // 2 + 1
        rate = self.resampling_rate if self.resampling_rate else self.sampling_rate

        self.seconds = seconds

        if mfcc is True:
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                self.num_mel_bins, num_spectrogram_bins, rate,
                self.lower_frequency, self.upper_frequency)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft

    def apply_resampling(self, audio):
        audio = signal.resample_poly(audio, 1, self.sampling_rate // self.resampling_rate)
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        return audio

    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)

        if self.resampling_rate:
            audio = tf.numpy_function(self.apply_resampling, [audio], tf.float32)

        return audio, label_id

    def pad(self, audio):

        if self.resampling_rate is not None:
            rate = self.resampling_rate
        else:
            rate = self.sampling_rate

        zero_padding = tf.zeros([self.seconds * rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.seconds * rate])

        return audio

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(audio, frame_length=self.frame_length,
                              frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                                       self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, label

    def read_pad_aug(self, file_path):

        filename = tf.strings.split(file_path, sep='/', maxsplit=-1, name=None)[-1]
        aug_ext = tf.strings.split(filename, sep='_', maxsplit=-1, name=None)[-1]

        # GraphTensor cannot use .numpy()!
        # to do : optimize it
        # check it!!
        if aug_ext == tf.constant('noise.wav', dtype=tf.string):
            clean_path = tf.strings.regex_replace(file_path, "_noise.wav$", ".wav")
        else:
            clean_path = tf.strings.regex_replace(file_path, "_no.wav$", ".wav")

        audio, label = self.read(clean_path)

        if self.resampling_rate is not None:
            rate = self.resampling_rate
        else:
            rate = self.sampling_rate

        if tf.shape(audio) > self.seconds * rate:
            audio = audio[:self.seconds * rate]
        else:
            audio = self.pad(audio)

        if aug_ext == tf.constant('noise.wav', dtype=tf.string):
            audio = self.add_white_noise(audio, 0.1)

        return audio, label

    def preprocess_with_mfcc(self, audio, label):

        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, label

    def add_white_noise(self, signal, noise_factor=0.1):
        mean, var = tf.nn.moments(signal, axes=[0])
        noise = tf.random.normal(tf.shape(signal), mean=0.0, stddev=var ** 2, dtype=tf.dtypes.float32, seed=None,
                                 name=None)
        return signal + noise * noise_factor

    def make_dataset(self, files, train, augumentation):

        ds = tf.data.Dataset.from_tensor_slices(files)

        ds_new = tf.data.Dataset.from_tensor_slices([''])

        # duplicate audios for augumentation
        if augumentation:
            for elem in ds:
                filename_no_aug = tf.strings.regex_replace(elem, ".wav^", "_no.wav")
                filename_noise = tf.strings.regex_replace(elem, ".wav^", "_noise.wav")

                ds_new = ds_new.concatenate(tf.data.Dataset.from_tensor_slices([filename_no_aug]))
                ds_new = ds_new.concatenate(tf.data.Dataset.from_tensor_slices([filename_noise]))

                # skip ''
                ds_new = ds_new.skip(1)
        else:
            ds_new = ds
            
        ds_new = ds_new.map(self.read_pad_aug, num_parallel_calls=4)
        ds_new = ds_new.map(self.preprocess, num_parallel_calls=4)

        ds_new = ds_new.batch(32)
        ds_new = ds_new.cache()

        if train:
            ds_new = ds_new.shuffle(100, reshuffle_each_iteration=True)

        return ds_new