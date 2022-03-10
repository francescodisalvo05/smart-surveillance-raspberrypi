import os
import zlib
import numpy as np

from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns

from utils.signal_generator import SignalGenerator
from utils.inference_latency import print_latency
from utils.ResNet import Residual
from utils.MobileNetV2 import MobileNetV2

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
from sklearn.metrics import f1_score

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

class Model():

    def __init__(self, model_name=None, alpha=1, n_classes=None, pruning=False, input_shape=None):

        self.model_name = model_name
        self.n_classes = n_classes
        self.input_shape = (input_shape[1],input_shape[2],input_shape[3])

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
                  keras.layers.Conv2D(filters=int(self.alpha*256), kernel_size=[3, 3], strides=strides, use_bias=False, input_shape=self.input_shape),
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
        elif self.model_name == "MobileNet":
            model = keras.Sequential([
                  keras.layers.Conv2D(filters=int(self.alpha*32), kernel_size=[3, 3], strides=strides, use_bias=False, input_shape=self.input_shape),
                  keras.layers.BatchNormalization(momentum=0.1),
                  keras.layers.ReLU(),

                  keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", use_bias=False),
                  keras.layers.BatchNormalization(momentum=0.1),
                  keras.layers.ReLU(),
                  keras.layers.Conv2D(filters=int(self.alpha*64), kernel_size=[1, 1], strides=[1, 1], padding="same", use_bias=False),
                  keras.layers.BatchNormalization(momentum=0.1),
                  keras.layers.ReLU(),

                  keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[2, 2], padding="same", use_bias=False),
                  keras.layers.BatchNormalization(momentum=0.1),
                  keras.layers.ReLU(),
                  keras.layers.Conv2D(filters=int(self.alpha*128), kernel_size=[1, 1], strides=[1, 1], padding="same", use_bias=False),
                  keras.layers.BatchNormalization(momentum=0.1),
                  keras.layers.ReLU(),

                  keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", use_bias=False),
                  keras.layers.BatchNormalization(momentum=0.1),
                  keras.layers.ReLU(),
                  keras.layers.Conv2D(filters=int(self.alpha*128), kernel_size=[1, 1], strides=[1, 1], padding="same", use_bias=False),
                  keras.layers.BatchNormalization(momentum=0.1),
                  keras.layers.ReLU(),

                  #keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[2, 2], padding="same", use_bias=False),
                  #keras.layers.BatchNormalization(momentum=0.1),
                  #keras.layers.ReLU(),
                  #keras.layers.Conv2D(filters=int(self.alpha*256), kernel_size=[1, 1], strides=[1, 1], padding="same", use_bias=False),
                  #keras.layers.BatchNormalization(momentum=0.1),
                  #keras.layers.ReLU(),

                  #keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], padding="same", use_bias=False),
                  #keras.layers.BatchNormalization(momentum=0.1),
                  #keras.layers.ReLU(),
                  #keras.layers.Conv2D(filters=int(self.alpha*256), kernel_size=[1, 1], strides=[1, 1], padding="same", use_bias=False),
                  #keras.layers.BatchNormalization(momentum=0.1),
                  #keras.layers.ReLU(),

                  keras.layers.GlobalAveragePooling2D(),
                  keras.layers.Dense(units=units)
            ])
        elif self.model_name == "MobileNetV2":
            model = MobileNetV2(self.alpha, self.input_shape, units)
        elif self.model_name == "NotWorkingMobileNetV2":
            model = keras.Sequential([
                  keras.layers.Conv2D(filters=int(self.alpha*32), kernel_size=[3, 3], strides=strides, use_bias=False, input_shape=self.input_shape),
                  keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999),  
                  keras.layers.ReLU(6.),

                  Residual(filters=int(self.alpha*16)),
                  
                  keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[2, 2], padding="same", use_bias=False),
                  keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999),
                  keras.layers.ReLU(6.),
                  keras.layers.Conv2D(filters=int(self.alpha*16), kernel_size=[1, 1], strides=[1, 1], padding="same", use_bias=False),
                  keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999),
                  keras.layers.ReLU(6.),

                  Residual(filters=int(self.alpha*24)),
                
                  keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[2, 2], padding="same", use_bias=False),
                  keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999),
                  keras.layers.ReLU(6.),
                  keras.layers.Conv2D(filters=int(self.alpha*24), kernel_size=[1, 1], strides=[1, 1], padding="same", use_bias=False),
                  keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999),
                  keras.layers.ReLU(6.),

                  keras.layers.GlobalAveragePooling2D(),
                  keras.layers.Dense(units=units)
            ])
        elif self.model_name == "SimpleNet":
            model = keras.Sequential([
                  keras.layers.Conv2D(filters=int(self.alpha*32), kernel_size=[3, 3], strides=strides, use_bias=False, input_shape=self.input_shape),
                  keras.layers.ReLU(),
                  keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2), padding="same"),

                  keras.layers.Conv2D(filters=int(self.alpha*64), kernel_size=[3, 3], strides=[1, 1], padding="same", use_bias=False),
                  keras.layers.ReLU(),
                  keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2), padding="same"),

                  keras.layers.Conv2D(filters=int(self.alpha*128), kernel_size=[3, 3], strides=[1, 1], padding="same", use_bias=False),
                  keras.layers.ReLU(),
                  keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2), padding="same"),

                  keras.layers.Conv2D(filters=int(self.alpha*256), kernel_size=[3, 3], strides=[1, 1], padding="same", use_bias=False),
                  keras.layers.ReLU(),
                  keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2), padding="same"),

                  keras.layers.Dropout(0.25),
                  keras.layers.Flatten(),
                  keras.layers.Dense(units=128),
                  keras.layers.ReLU(),
                  keras.layers.Dropout(0.5),
                  keras.layers.Dense(units=units)
            ])
        elif self.model_name == "VGG":
            model = keras.Sequential([

            # Input shape to be specified 
                keras.layers.Conv2D(input_shape=self.input_shape,filters=64,kernel_size=(3,3),padding="same", activation="relu"),
                keras.layers.Conv2D(filters=64,kernel_size=(3,3), padding="same", activation="relu"),
                keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2), padding="same"),  # shapes problem here
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
                keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
                keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2), padding="same"),
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
                keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
                keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2), padding="same"),
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
                keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2), padding="same"),
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
                keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
                keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2), padding="same"),

                keras.layers.Flatten(),
                keras.layers.Dense(units=512,activation="relu"),
                keras.layers.Dense(units=256,activation="relu"),
                keras.layers.Dense(units=units, activation="softmax")
            ])
        else:
            raise ValueError('{} not implemented'.format(self.model_name))  

        return model  


    def train_model(self, train_ds, val_ds, learning_rate, input_shape, num_epochs):

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='models_tflite', monitor='val_sparse_categorical_accuracy', save_best_only=True)
        lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        callbacks = [cp_callback, lr_callback]
            
        pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.30,
                                                                               final_sparsity=0.40,
                                                                               begin_step=len(train_ds) * 5,
                                                                               end_step=len(train_ds) * 15)}
        if self.pruning:
            
            prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
            model = prune_low_magnitude(self.model,**pruning_params)
            pr_callback = tfmot.sparsity.keras.UpdatePruningStep()
            callbacks.append(pr_callback)
        
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

        confusion_matrix = tf.math.confusion_matrix(labels, predictions)  # add names!

        f1 = f1_score(labels, predictions, average='macro')

        return confusion_matrix, f1

    def plot_stats(self, f1, cm, MFCC_OPTIONS, labels, resampling_rate):

        print("\n ==== STATS ====")

        # f1 score
        print("F1 Score = {} %".format(round(f1 * 100, 2)))

        # dimension
        print("Model size = {} kB".format(round(os.path.getsize(self.tflite_path) / 1024, 2)))

        # latency
        # print_latency(self.model_path, MFCC_OPTIONS, resampling_rate) -> FIX IT!!

        # confusion matrix
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, fmt='g', ax=ax)
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Room - {}'.format('Danger'))
        plt.tight_layout()
        plt.savefig('heatmap_{}.png'.format('Danger'))
