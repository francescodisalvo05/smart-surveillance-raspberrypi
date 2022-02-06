import os
import zlib
import numpy as np

from scipy import signal
from utils.signal_generator import SignalGenerator

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras


class Model():

    def __init__(self, model_name=None, alpha=1, n_classes=None, pruning=False):

        self.model_name = model_name
        self.n_classes = n_classes

          #initialize optimization
        self.pruning = pruning
        self.alpha = alpha 

        self.model = self.set_model()

      
        # initialize
        self.tflite_path = None
        self.model_path = None


    def set_model(self): # units = num classes
        
        strides=[2,1]
        units = self.n_classes

        if self.model_name == "DS-CNN":

            model = keras.Sequential([
                  keras.layers.Conv2D(filters=int(self.alpha*256), kernel_size=[3, 3], strides=strides, use_bias=False),
                  keras.layers.BatchNormalization(momentum=0.1),
                  keras.layers.ReLU(),

                  keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
                  keras.layers.Conv2D(filters=int(self.alpha*256), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
                  keras.layers.BatchNormalization(momentum=0.1),
                  keras.layers.ReLU(),

                  keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
                  keras.layers.Conv2D(filters=int(self.alpha*256), kernel_size=[1, 1], strides=[1, 1], use_bias=False),
                  keras.layers.BatchNormalization(momentum=0.1),
                  keras.layers.ReLU(),

                  keras.layers.GlobalAveragePooling2D(),
                  keras.layers.Dense(units=units)
              ])
              
        elif self.model_name == "VGG":
            model = keras.Sequential([

            # Input shape to be specified 
                keras.layers.Conv2D(input_shape=(),filters=64,kernel_size=(3,3),padding="same", activation="relu"),
                keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"),
                keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),  # shapes problem here
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
                keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
                keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
                keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
                keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2))

                keras.layers.Flatten(),
                keras.layers.Dense(units=4096,activation="relu"),
                keras.layers.Dense(units=4096,activation="relu"),
                keras.layers.Dense(units=2, activation="softmax")
            ])
        else:
            raise ValueError('{} not implemented'.format(self.model_name))  

        return model  


    def train_model(self, train_ds, val_ds, learning_rate, input_shape, num_epochs):
            
        pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.30,
                                                                               final_sparsity=0.85,
                                                                               begin_step=len(train_ds) * 5,
                                                                               end_step=len(train_ds) * 15)}
        if self.pruning:
            prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
            model = prune_low_magnitude(self.model,**pruning_params)
            callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
        
            self.model.build(input_shape=input_shape)
            self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=keras.metrics.SparseCategoricalAccuracy())

            self.model.fit(train_ds, epochs=num_epochs, validation_data=val_ds, callbacks=callbacks)   
            model = tfmot.sparsity.keras.strip_pruning(model)
        
        else:
            self.model.build(input_shape=input_shape)
            self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=keras.metrics.SparseCategoricalAccuracy())

            self.model.fit(train_ds, epochs=num_epochs, validation_data=val_ds)   


    def save_tf(self, path=''):

        self.model_path = path

        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)

        self.model.save(self.model_path)



    def save_tflite(self, path='', optimization=None):
        
        #if not os.path.isdir(path):
        #    os.mkdir(path)

        self.tflite_path = path

        converter = tf.lite.TFLiteConverter.from_saved_model(self.model_path)
        converter.experimental_enable_resource_variables = True

        if optimization is not None:
            converter.optimizations = optimization

        tflite_m = converter.convert()

        # save tflite model
        with open(self.tflite_path, 'wb') as fp:
            fp.write(tflite_m)

       

    def make_inference(self, test_ds):
        
        interpreter = tf.lite.Interpreter(model_path=self.tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        test_ds = test_ds.unbatch().batch(1)

        predictions, labels = [], []

        for x, y in test_ds:
            # give the input
            interpreter.set_tensor(input_details[0]["index"], x)
            interpreter.invoke()

            # predict and get the current ground truth
            curr_prediction_logits = interpreter.get_tensor(output_details[0]['index']).squeeze()
            curr_label = y.numpy().squeeze()

            curr_prediction = np.argmax(curr_prediction_logits)

            predictions.append(curr_prediction)
            labels.append(curr_label)

        confusion_matrix = tf.math.confusion_matrix(labels, predictions) # add names!

        return confusion_matrix