from scipy import signal
from scipy.io import wavfile

import wave
import numpy as np
from tqdm import tqdm

import os

DATASET_OLD_FOLDER = 'assets/audio/data/Dataset/'
DATASET_NEW = 'assets/audio/data/Dataset_enlarged/'

if not os.path.exists(DATASET_NEW):
    os.mkdir(DATASET_NEW)

for label in os.listdir(DATASET_OLD_FOLDER):

    if not os.path.exists(os.path.join(DATASET_NEW,label)):
        os.mkdir(os.path.join(DATASET_NEW,label))

    for audio_name in tqdm(os.listdir(os.path.join(DATASET_OLD_FOLDER,label))):

        if audio_name.split(".")[1] != "wav":
            continue

        path = os.path.join(DATASET_OLD_FOLDER, label, audio_name)

        _, audio = wavfile.read(path)

        seconds = len(audio) // 44100

        # without window of half seconds set:
        # - range(seconds)
        # - audio = audio[44100:]
        for s in range(seconds*2):
            curr_audio = audio[0:44100] 
            audio = audio[22050:] # shift half second

            new_audio_name = audio_name.replace(".wav","_{}.wav".format(s+1))

            path_new = os.path.join(DATASET_NEW, label, new_audio_name)
            wavfile.write(path_new, 44100, curr_audio.astype(np.int16))