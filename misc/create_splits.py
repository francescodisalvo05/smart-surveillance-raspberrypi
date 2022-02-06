from sklearn.model_selection import train_test_split
import pandas as pd
import argparse, os

import sys

sys.path.append('constants/')

from split import ROOM_DICTIONARY
from misc import RANDOM_STATE

DATASET = 'data/Dataset/'


DATASET_DEV_IDX = 'assets/dataset_idx/dev_ids.csv'
DATASET_EVAL_IDX = 'assets/dataset_idx/eval_ids.csv'

# It is based on Google Colab.. Fix it
SPLIT_BASE_PATH = 'assets/dataset_split'

parser = argparse.ArgumentParser()

# to do: define all the other choices for room
parser.add_argument('-r','--room',type=str, default='all', choices=['all','bedroom','bathroom','kitchen','office','entrance','workshop'])

args = parser.parse_args()

##########################################################
######### TRAIN / VAL
##########################################################

df = pd.read_csv(DATASET_DEV_IDX, names=['fname', 'label'])

fnames, classes = [], []
room_dictionary = ROOM_DICTIONARY[args.room]

TRAIN_NAME = 'train_{}_split.txt'.format(args.room)
VALIDATION_NAME = 'val_{}_split.txt'.format(args.room)
TEST_NAME = 'test_{}_split.txt'.format(args.room)

# get only 100 records
# they will be augumented later
for index, row in df.iterrows():
    if df.loc[index, 'label'] in list(room_dictionary.keys()):
        if room_dictionary[df.loc[index, 'label']] < 100:
            fnames.append(df.loc[index, 'fname'])
            classes.append(df.loc[index, 'label'])
            room_dictionary[df.loc[index, 'label']] += 1

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

y = df_eval.label
X = df_eval.drop(columns=['label'])

with open(os.path.join(SPLIT_BASE_PATH, TEST_NAME), 'w') as file:
    for idx in range(len(X)):
        string = './{}{}/{}.wav\n'.format(DATASET, y.iloc[idx], X.iloc[idx].fname)
        file.write(string)