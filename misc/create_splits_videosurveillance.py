from sklearn.model_selection import train_test_split
import pandas as pd
import argparse, os

import sys

sys.path.append('constants/')

from misc import RANDOM_STATE

DATASET = 'data/Dataset/'


DATASET_DEV_IDX = 'assets/dataset_idx/new_idx_dev.csv'
DATASET_EVAL_IDX = 'assets/dataset_idx/new_idx_eval.csv'
SPLIT_BASE_PATH = 'assets/dataset_split/'

##########################################################
######### TRAIN / VAL
##########################################################

df = pd.read_csv(DATASET_DEV_IDX, names=['fname', 'label'])

fnames, classes = [], []

'''
Train frequencies:
Bark : 414,
Door: 1056,
Drill: 169,
Hammer: 165,
Fire: 733,
Gunshot: 384,
Glass: 559,
Slam: 323
'''

freq_dictionary = {
    'Bark' : 0,
    'Door': 0,
    'Drill': 0,
    'Hammer': 0,
    'Fire': 0,
    'Gunshot_and_gunfire': 0,
    'Glass': 0,
    'Slam': 0
}

TRAIN_NAME = 'train_split.txt'
VALIDATION_NAME = 'val_split.txt'
TEST_NAME = 'test_split.txt'

# get only 100 records
# they will be augumented later
for index, row in df.iterrows():
    if df.loc[index, 'label'] in list(freq_dictionary.keys()):
        if freq_dictionary[df.loc[index, 'label']] < 100:
            fnames.append(df.loc[index, 'fname'])
            classes.append(df.loc[index, 'label'])
            freq_dictionary[df.loc[index, 'label']] += 1

df_new = pd.DataFrame({'fname': fnames, 'label': classes})

y = df_new.label
X = df_new.drop(columns=['label'])

# split it into train and validation
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.10, stratify=y, random_state=RANDOM_STATE)

with open(os.path.join(SPLIT_BASE_PATH, TRAIN_NAME), 'w') as file:
    for idx in range(len(x_train)):
        string = './{}{}/{}.wav\n'.format(DATASET, y_train.iloc[idx], x_train.iloc[idx].fname)
        file.write(string)

with open(os.path.join(SPLIT_BASE_PATH, VALIDATION_NAME), 'w') as file:
    for idx in range(len(x_val)):
        string = './{}{}/{}.wav\n'.format(DATASET, y_val.iloc[idx], x_val.iloc[idx].fname)
        file.write(string)

file.close()

##########################################################
######### TEST
##########################################################

df_eval = pd.read_csv(DATASET_EVAL_IDX, names=['fname', 'label'])



fnames, classes = [], []

# get only 100 records
# they will be augumented later
for index, row in df_eval.iterrows():
    if df.loc[index, 'label'] in list(freq_dictionary.keys()):
        fnames.append(df.loc[index, 'fname'])
        classes.append(df.loc[index, 'label'])

df_new_eval = pd.DataFrame({'fname': fnames, 'label': classes})
y = df_new_eval.label
X = df_new_eval.drop(columns=['label'])

with open(os.path.join(SPLIT_BASE_PATH, TEST_NAME), 'w') as file:
    for idx in range(len(X)):
        string = './{}{}/{}.wav\n'.format(DATASET, y.iloc[idx], X.iloc[idx].fname)
        file.write(string)