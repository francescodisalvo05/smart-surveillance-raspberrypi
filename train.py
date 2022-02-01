import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
import zlib
from scipy import signal
from signal_generator import SignalGenerator


TRAIN_PATH = ''
TEST_PATH = ''

def get_data(options,resampling):

    csv_path, _ = os.path.splitext() # to be determined

    data_dir = '' # to be determined 

    with open(TRAIN_PATH ,"r") as fp:
        train_files = [line.rstrip() for line in fp.readlines()]  
    
    with open(TEST_PATH ,"r") as fp:
        test_files = [line.rstrip() for line in fp.readlines()]  
    
    

    labels = [
    'Speech','Alarm','Drawer_open_or_close','Door','Crying_and_sobbing',
    'Mechanical_fan', 'Ringtone', 'Sink_(filling_or_washing)', 'Water_tap_and_faucet',
    'Microwave_oven', 'Printer', 'Scissors', 'Computer_keyboard', 'Doorbell', 'Keys_jangling'
    'Knock', 'Ringtone', 'Packing_tape_and_duct_tape', 'Hammer', 'Sawing'
    ]

    generator = SignalGenerator(labels, sampling_rate=44100, resampling_rate=resampling, **options)
    train_ds = generator.make_dataset(train_files, True)
    train_ds = train_ds.take(4184)  #making 80-20 train-validation split
    val_ds = train_ds.skip(4184)
    test_ds = generator.make_dataset(test_files, False)

    return train_ds, val_ds, test_ds

def save_tf(model):

    SAVING_TF_PATHDIR = ''

    if not os.path.isdir('models_tf/'):
        os.mkdir('models_tf/')

    if not os.path.isdir(SAVING_TF_PATHDIR):
        os.mkdir(SAVING_TF_PATHDIR)

    model.save(SAVING_TF_PATHDIR)

    return SAVING_TF_PATHDIR


def get_model(model_name,**params):

    if model_name == "MLP":

        model = keras.Sequential([keras.layers.Flatten(input_shape=(6,2)),
                              keras.layers.Dense(units=128 * params['alpha'], activation='relu'),
                              keras.layers.Dense(units=128 * params['alpha'], activation='relu'),
                              keras.layers.Dense(units=params['step'] * 2),
                              keras.layers.Reshape([params['step'], 2])])
    
    elif model_name == "DS-CNN":

        model = keras.Sequential([
                  keras.layers.Conv2D(filters=int(params['alpha'],*256), kernel_size=[3, 3], strides=params['alpha'], use_bias=False),
                  keras.layers.BatchNormalization(momentum=0.1),
                  keras.layers.ReLU(),

                  keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
                  keras.layers.Conv2D(filters=int(params['alpha']*256), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
                  keras.layers.BatchNormalization(momentum=0.1),
                  keras.layers.ReLU(),

                  keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
                  keras.layers.Conv2D(filters=int(params['alpha']*256), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
                  keras.layers.BatchNormalization(momentum=0.1),
                  keras.layers.ReLU(),

                  keras.layers.GlobalAveragePooling2D(),
                  keras.layers.Dense(units=params['units'])
              ])

    return model

def train_model(model,train_ds,val_ds,**params):
        

    model.build(input_shape=params['learning_rate'])
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=params['learning_rate']),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=keras.metrics.SparseCategoricalAccuracy())

    model.fit(train_ds, epochs=params['learning_rate'], validation_data=val_ds)   

    TF_PATHDIR = save_tf(model)

def save_tflite(TF_PATH,optimization=None):

    TFLITE_PATHDIR =  'models_tflite/'
    TFLITE_MODEL_PATH =  TFLITE_PATHDIR + ''

    if not os.path.isdir('models_tflite/'):
        os.mkdir('models_tflite/')

    if not os.path.isdir(TFLITE_PATHDIR):
        os.mkdir(TFLITE_PATHDIR)

    converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)
    converter.experimental_enable_resource_variables = True

    if optimization is not None:
        converter.optimizations = optimization

    tflite_m = converter.convert()

    # save tflite model
    with open(TFLITE_MODEL_PATH, 'wb') as fp:
        fp.write(tflite_m)

    # compress the tflite model and save it
    TFLITE_PATH_COMPRESSED = TFLITE_MODEL_PATH + ".zlib"
    with open(TFLITE_PATH_COMPRESSED, 'wb') as fp:
        compressed_tflite_model = zlib.compress(tflite_m, level=9)
        fp.write(compressed_tflite_model)

    return os.path.getsize(TFLITE_PATH_COMPRESSED) / 1024, TFLITE_MODEL_PATH

def make_inference(test_ds,TFLITE_MODEL_PATH):
    
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    test_ds = test_ds.unbatch().batch(1)

    total_count, correct_count = 0, 0

    for x, y in test_ds:
        # give the input
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()

        # predict and get the current ground truth
        curr_prediction_logits = interpreter.get_tensor(output_details[0]['index']).squeeze()
        curr_label = y.numpy().squeeze()

        curr_prediction = np.argmax(curr_prediction_logits)

        if curr_prediction == curr_label:
          correct_count += 1
        
        total_count += 1

    return correct_count / total_count
    

