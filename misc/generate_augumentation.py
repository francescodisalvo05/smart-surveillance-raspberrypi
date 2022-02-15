from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

from scipy import signal
from scipy.io import wavfile

import numpy as np
import os
import sys

sys.path.append('constants/')
from split import ROOM_DICTIONARY
from path import DATASET, AUGUMENTATION_PATH

classes = list(ROOM_DICTIONARY['all'].keys())

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
])

# check if it exists..
if not os.path.exists(AUGUMENTATION_PATH):
    os.mkdir(AUGUMENTATION_PATH)

for c in classes:
    for filename in os.listdir(DATASET + c):
        rate, audio = wavfile.read(DATASET + c + '/' + filename)
        augmented_samples = augment(samples=audio, sample_rate=rate)
        wavfile.write(DATASET + c + '/' + filename, rate, augmented_samples)

