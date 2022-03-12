import pandas as pd
import numpy as np
import os
import shutil

DATASET_FOLDER = '/Volumes/TOSHIBA EXT/DOMESTIC SOUNDS/Dataset/'
BASE_PATH = 'assets/dataset_idx/'

# files = ['new_idx_dev.csv','new_idx_eval.csv']
files = ['new_idx_dev.csv']

if not os.path.exists(DATASET_FOLDER):
    os.mkdir(DATASET_FOLDER)


for file in files:

    if file == 'new_idx_dev.csv':
        DATA_PATH = '/Volumes/TOSHIBA EXT/DOMESTIC SOUNDS/FSD50K.dev_audio_old/'
    elif file == 'new_idx_eval.csv':
        DATA_PATH = '/Volumes/TOSHIBA EXT/DOMESTIC SOUNDS/FSD50K.eval_audio/'
    
    df = pd.read_csv(BASE_PATH + file, names=['fname', 'label'])

    fnames, labels = df['fname'], df['label']

    for f, l in zip(fnames, labels):

        if not os.path.exists(DATASET_FOLDER + l):
            os.mkdir(DATASET_FOLDER + l)

        # copy from dataset path to dataset folder + class
        src = DATA_PATH + str(f) + '.wav'
        dst = DATASET_FOLDER + l + '/' + str(f) + '.wav'
        try:
            shutil.copyfile(src, dst)
        except:
            print("-- Error with {}".format(src))







