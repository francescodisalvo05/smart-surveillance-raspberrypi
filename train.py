import argparse
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot

from utils.data import get_data
from utils.model import Model
from constants.split import ROOM_DICTIONARY

TRAIN_PATH = ''
TEST_PATH = ''


def main(args):
    MFCC_OPTIONS = {
        'frame_length': 240, 'frame_step': 120, 'mfcc': True, 'lower_frequency': 20,
        'upper_frequency': 4000, 'num_mel_bins': 40, 'num_coefficients': 10
    }

    labels = list(ROOM_DICTIONARY[args.room].keys())

    train_ds, val_ds, test_ds = get_data(
        room=args.room,
        labels=labels,
        mfcc_options=MFCC_OPTIONS,
        resampling=None)

    model = Model(model_name='DS-CNN', n_classes=len(labels), alpha=1, pruning=None)

    learning_rate = 0.001
    epochs = args.epochs
    input_shape = [32, 1469, 10, 1]

    model.train_model(train_ds, val_ds, learning_rate, input_shape, epochs)

    # if save model..
    model.save_tf('models_tf/model_test')

    model.save_tflite('models_tflite/model_test_tflite/model.tflite')

    cm = model.make_inference(test_ds)

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)

    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Room - {}'.format(args.room))
    plt.tight_layout()

    plt.savefig('heatmap.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--room', type=str, default='all', choices=['all', 'bedroom'])
    parser.add_argument('-e', '--epochs', type=int, default=20)

    args = parser.parse_args()

    main(args)



