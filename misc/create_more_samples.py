from scipy import signal
from scipy.io import wavfile

import wave
import numpy as np

import os


DATASET_OLD_FOLDER = 'data/Dataset_old/'
DATASET_NEW = 'data/Dataset_new/'

for label in os.listdir(DATASET_OLD_FOLDER):

    if not os.path.exists(os.path.join(DATASET_NEW,label)):
        os.mkdir(os.path.join(DATASET_NEW,label))

    for audio_name in os.listdir(os.path.join(DATASET_OLD_FOLDER,label)):

        path = os.path.join(DATASET_OLD_FOLDER, label, audio_name)
        _, audio = wavfile.read(path)

        seconds = len(audio) // 44100

        for s in range(seconds):
            curr_audio = audio[0:44100]
            audio = audio[44100:]

            new_audio_name = audio_name.replace(".wav","_{}.wav".format(s+1))

            path_new = os.path.join(DATASET_NEW, label, new_audio_name)
            wavfile.write(path_new, 44100, curr_audio.astype(np.int16))
