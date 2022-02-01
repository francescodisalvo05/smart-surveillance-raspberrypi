'''
Remove from the initial folder all the audio files that do not 
belongs to our classes. 
'''

import os
import pandas as pd
from collections import Counter 

AUDIO_FOLDER = '/Volumes/TOSHIBA EXT/DOMESTIC SOUNDS/FSD50K.dev_audio_old/'

df = pd.read_csv('assets/idx_dev.csv', names=['fname','class'])

files_to_keep = df.fname.to_list()
files_to_keep_str = [str(f) for f in files_to_keep]

all_files = os.listdir(AUDIO_FOLDER)
all_files = [f.split(".wav")[0] for f in all_files]

files_to_remove = []
for audio in all_files:
    if audio not in files_to_keep_str:
        files_to_remove.append(audio)
        os.remove(AUDIO_FOLDER + audio + '.wav')

print('Number of total files : {}'.format(len(all_files)))
print('Length of files to keep list : {}'.format(len(files_to_keep)))
print('Number of unique files to keep : {}'.format(len(Counter(files_to_keep))))
print('Length of the files to remove : {}'.format(len(files_to_remove)))
print('')
print('Final dimension : {}'.format(len(all_files) - len(files_to_remove)))

