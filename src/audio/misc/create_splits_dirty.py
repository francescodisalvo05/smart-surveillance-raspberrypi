from sklearn.model_selection import train_test_split
import pandas as pd
import argparse, os

import sys

sys.path.append('src/audio/constants/')

from misc import RANDOM_STATE

DATASET = 'assets/audio/data/Dataset/'
SPLIT_BASE_PATH = 'assets/audio/dataset_split/'

TRAIN_NAME = 'train_split_dirty.txt'
VALIDATION_NAME = 'val_split_dirty.txt'
TEST_NAME = 'test_split.txt'

# read test file from the cleaned one
x_clean = []
with open(os.path.join(SPLIT_BASE_PATH, TEST_NAME), 'r') as file:
    lines = file.readlines()
    for line in lines:
        x_clean.append(line.strip().split("/")[-1])
file.close()

fnames, classes = [], []

for folder in ['Bark','Crash','Door','Doorbell','Drill','Other','Speech']:
    for file in os.listdir(os.path.join(DATASET,folder)):
        classes.append(folder)
        fnames.append(file)

df = pd.DataFrame({'fname':fnames, 'label':classes})

y = df.label
X = df.drop(columns=['label'])

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.13, stratify=y, random_state=RANDOM_STATE) 

with open(os.path.join(SPLIT_BASE_PATH, TRAIN_NAME), 'w') as file:
    for idx in range(len(x_train)):
        if x_train.iloc[idx].fname not in x_clean:
            string = './{}{}/{}\n'.format(DATASET, y_train.iloc[idx], x_train.iloc[idx].fname)
            file.write(string)
file.close()

with open(os.path.join(SPLIT_BASE_PATH, VALIDATION_NAME), 'w') as file:
    for idx in range(len(x_val)):
        if x_train.iloc[idx].fname not in x_clean:
            string = './{}{}/{}\n'.format(DATASET, y_val.iloc[idx], x_val.iloc[idx].fname)
            file.write(string)
file.close()