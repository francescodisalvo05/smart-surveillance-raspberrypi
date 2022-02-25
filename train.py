import argparse
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot

from utils.data import get_data
from utils.model import Model
from constants.split import ROOM_DICTIONARY
from constants.misc import RANDOM_STATE

import tensorflow as tf

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

def main(args):

    '''16 kHz
    MFCC_OPTIONS = {
        'frame_length': 640 * 2, 'frame_step': 320 * 2, 'mfcc': True, 'lower_frequency': 20,
        'upper_frequency': 4000, 'num_mel_bins': 40, 'num_coefficients': 10
    }
    resampling_rate = 16000
    '''

    '''44,1 kHz'''
    MFCC_OPTIONS = {
        'frame_length': 1764 *2 , 'frame_step': 882 *2, 'mfcc': True, 'lower_frequency': 20,
        'upper_frequency': 4000, 'num_mel_bins': 40, 'num_coefficients': 10
    }
    resampling_rate = None

    labels = list(['Bark',
    'Door',
    'Drill',
    'Hammer',
    'Gunshot',
    'Glass'])
    

    train_ds, val_ds, test_ds = get_data(
        labels=labels,
        mfcc_options=MFCC_OPTIONS,
        resampling=resampling_rate)

    for elem in train_ds:
      input_shape = elem[0].shape.as_list()
      print(input_shape)
      break

    learning_rate = 0.001
    epochs = args.epochs

    model = Model(model_name='DS-CNN', 
                  n_classes=len(labels),
                  input_shape=input_shape, 
                  alpha=0.8,
                  pruning=True)    

    model.train_model(train_ds, val_ds, learning_rate, input_shape, epochs)

    # if save model..
    model.save_tf('models_tf/model_test')

    model.save_tflite('models_tflite/model_test_tflite/model.tflite')

    cm, f1_score = model.make_inference(test_ds)

    model.plot_stats(f1_score, cm, MFCC_OPTIONS, labels, resampling_rate)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--epochs', type=int, default=20)

    args = parser.parse_args()

    main(args)



