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
room_dictionary = ROOM_DICTIONARY[args.room].copy()

TRAIN_NAME = 'train_{}_aug_split.txt'.format(args.room)
VALIDATION_NAME = 'val_{}_aug_split.txt'.format(args.room)
TEST_NAME = 'test_{}_aug_split.txt'.format(args.room)

# get only 100 records
# they will be augumented later
for index, row in df.iterrows():
    if df.loc[index, 'label'] in list(room_dictionary.keys()):
        if room_dictionary[df.loc[index, 'label']] < 300:
            fnames.append(str(df.loc[index, 'fname']))
            classes.append(str(df.loc[index, 'label']))
            room_dictionary[df.loc[index, 'label']] += 1

df_new = pd.DataFrame({'fname': fnames, 'label': classes})

y = df_new.label
X = df_new.drop(columns=['label'])

# split it into train and validation
x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.10, stratify=y, random_state=RANDOM_STATE)

fnames_train, classes_train = list(x_train.fname), list(y_train)
fnames_train_aug, classes_train_aug = fnames_train.copy(), classes_train.copy()

from collections import Counter

# add augumentation
for idx in range(len(fnames_train)):
    if room_dictionary[classes_train[idx]] < 300:
        fnames_train_aug.append(fnames_train[idx] + '_shift')
        classes_train_aug.append(classes_train[idx])

print(Counter(classes_train))
print("-----------------")
print(Counter(classes_train_aug))

with open(os.path.join(SPLIT_BASE_PATH, TRAIN_NAME), 'w') as file:
    for idx in range(len(fnames_train_aug)):
        string = './{}{}/{}.wav\n'.format(DATASET, classes_train_aug[idx], fnames_train_aug[idx])
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
room_dictionary = ROOM_DICTIONARY[args.room].copy()

for index, row in df_eval.iterrows():
    if df.loc[index, 'label'] in list(room_dictionary.keys()):
        if room_dictionary[df.loc[index, 'label']] < 150:
            fnames.append(df.loc[index, 'fname'])
            classes.append(df.loc[index, 'label'])
            room_dictionary[df.loc[index, 'label']] += 1

df_new_eval = pd.DataFrame({'fname': fnames, 'label': classes})
y = df_new_eval.label
X = df_new_eval.drop(columns=['label'])

with open(os.path.join(SPLIT_BASE_PATH, TEST_NAME), 'w') as file:
    for idx in range(len(X)):
        string = './{}{}/{}.wav\n'.format(DATASET, y.iloc[idx], X.iloc[idx].fname)
        file.write(string)