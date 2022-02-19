import argparse
import tensorflow as tf
import time
from scipy import signal
import numpy as np
from subprocess import call


def print_latency(model_path, MFCC_OPTIONS, resampled_rate=16000):
    call('sudo sh -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
         shell=True)

    model = model_path
    rate = 44100

    mfcc = MFCC_OPTIONS['mfcc']
    resize = 32

    length = MFCC_OPTIONS['frame_length']
    stride = MFCC_OPTIONS['frame_step']

    num_mel_bins = MFCC_OPTIONS['num_mel_bins']
    num_coefficients = MFCC_OPTIONS['num_coefficients']

    lower_frequency = 20
    upper_frequency = 4000

    num_frames = 99  # (rate - length) // stride + 1 DOESN'T WORK - PADDING?
    num_spectrogram_bins = length // 2 + 1

    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, rate, lower_frequency,
        upper_frequency)

    if model is not None:
        interpreter = tf.lite.Interpreter(model_path=model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

    inf_latency = []
    tot_latency = []
    for _ in range(100):
        sample = np.array(np.random.random_sample(rate * 4), dtype=np.float32)

        start = time.time()

        if resampled_rate:
            desired_length = int(round(float(len(sample)) /
                                    rate * resampled_rate))
            
            sample = signal.resample(sample, desired_length)

        sample = tf.convert_to_tensor(sample, dtype=tf.float32)

        # STFT
        stft = tf.signal.stft(sample, length, stride,
                              fft_length=length)
        spectrogram = tf.abs(stft)

        if mfcc is False and resize > 0:
            # Resize (optional)
            spectrogram = tf.reshape(spectrogram, [1, num_frames, num_spectrogram_bins, 1])
            spectrogram = tf.image.resize(spectrogram, [resize, resize])
            input_tensor = spectrogram
        else:
            # MFCC (optional)
            mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
            log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
            mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
            mfccs = mfccs[..., :num_coefficients]
            mfccs = tf.reshape(mfccs, [1, num_frames, num_coefficients, 1])
            input_tensor = mfccs

        if model is not None:
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            start_inf = time.time()
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

        end = time.time()
        tot_latency.append(end - start)

        if model is None:
            start_inf = end

        inf_latency.append(end - start_inf)
        time.sleep(0.1)

    print('Inference Latency = {:.2f} ms'.format(np.mean(inf_latency) * 1000.))
    print('Total Latency = {:.2f} ms'.format(np.mean(tot_latency) * 1000.))