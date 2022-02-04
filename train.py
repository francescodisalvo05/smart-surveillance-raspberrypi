import argparse
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot

from utils.data import get_data
from utils.model import Model

TRAIN_PATH = ''
TEST_PATH = ''

def main(args):

    MFCC_OPTIONS = {
        'frame_length': 240, 'frame_step': 120, 'mfcc': True, 'lower_frequency': 20, 
        'upper_frequency': 4000, 'num_mel_bins': 40, 'num_coefficients': 10
    }

    

    labels = [
        'Speech','Alarm','Drawer_open_or_close','Door','Crying_and_sobbing',
        'Mechanical_fan', 'Ringtone', 'Sink_(filling_or_washing)', 'Water_tap_and_faucet',
        'Microwave_oven', 'Printer', 'Scissors', 'Computer_keyboard', 'Doorbell', 'Keys_jangling'
        'Knock', 'Ringtone', 'Packing_tape_and_duct_tape', 'Hammer', 'Sawing', 'Boiling', 'Toilet_flush'
    ]

    train_ds, val_ds ,test_ds= get_data(
        room = args.room,
        labels = labels,
        mfcc_options=MFCC_OPTIONS,
        resampling=None)

    model = Model(model_name='DS-CNN', n_classes=len(labels),alpha=1,pruning=None)

    learning_rate = 0.01
    epochs = args.epochs
    input_shape = [32,1469,10,1]

    model.train_model(train_ds, val_ds, learning_rate, input_shape, epochs)

    # if save model..
    model.save_tf('models_tf/model_test')

    model.save_tflite('models_tflite/model_test_tflite/model.tflite')

    cm = model.make_inference(test_ds)


    ax = plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)
    plt.savefig('heatmap.png')

    print(accuracy)

    



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-r','--room', type=str, default='all', choices=['all','bedroom'])
    parser.add_argument('-e','--epochs', type=int, default=20)
    
    args = parser.parse_args()

    main(args)

    

