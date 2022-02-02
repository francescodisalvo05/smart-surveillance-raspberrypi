import argparse
import numpy as np
import os

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
        'Knock', 'Ringtone', 'Packing_tape_and_duct_tape', 'Hammer', 'Sawing'
    ]

    train_ds, val_ds = get_data(
        labels = labels, 
        train_path = args.train_path,
        test_path=args.test_path,
        mfcc_options=MFCC_OPTIONS,
        resampling=None)

    model = Model(model_name='DS-CNN', n_classes=len(labels))

    learning_rate = 0.01
    epochs = 30
    input_shape = [32,366,10,1] 

    model.train_model(train_ds, val_ds, learning_rate, input_shape, epochs)

    # if save model..
    model.save_tf('models_tf/model_test')

    model.save_tflite('models_tflite/model_test.tflite')

    # to do : make inference? 



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path',type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    
    args = parser.parse_args()

    main(args)

    

