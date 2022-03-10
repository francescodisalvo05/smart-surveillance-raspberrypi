import tensorflow as tf
from tensorflow import keras

class Stride1Block(keras.layers.Layer):
  def __init__(self, filters):
    super(Stride1Block, self).__init__()
    filters1, filters2 = filters
    self.conv2a = keras.layers.Conv2D(filters=filters1,
                                      kernel_size=1,
                                      padding='same',
                                      use_bias=False,
                                      activation=None)
    self.batch_norm = keras.layers.BatchNormalization(momentum=0.1)
    self.relu6 = keras.layers.ReLU(6.)
    self.dwconv2 = keras.layers.DepthwiseConv2D(kernel_size=3,
                                      strides=1,
                                      padding='same',
                                      use_bias=False,
                                      activation=None)
    self.conv2b = keras.layers.Conv2D(filters=filters2,
                                      kernel_size=1,
                                      padding='same',
                                      use_bias=False,
                                      activation=None)
      
  def call(self, inputs):
    x = self.conv2a(inputs)
    x = self.batch_norm(x)
    x = self.relu6(x)
    x = self.dwconv2(x)
    x = self.batch_norm(x)
    x = self.relu6(x)
    x = self.conv2b(x)
    x = self.batch_norm(x)
    x += inputs
    return x
			
class Stride2Block(keras.layers.Layer):
  def __init__(self, filters):
    super(Stride2Block, self).__init__()
    filters1, filters2 = filters
    self.conv2a = keras.layers.Conv2D(filters=filters1,
                                      kernel_size=1,
                                      padding='same',
                                      use_bias=False,
                                      activation=None)
    self.batch_norm = keras.layers.BatchNormalization(momentum=0.1)
    self.relu6 = keras.layers.ReLU(6.)
    self.zero_pad2 = keras.layers.ZeroPadding2D()
    self.dwconv2 = keras.layers.DepthwiseConv2D(kernel_size=3,
                                      strides=2,
                                      padding='valid',
                                      use_bias=False,
                                      activation=None)
    self.conv2b = keras.layers.Conv2D(filters=filters2,
                                      kernel_size=1,
                                      padding='same',
                                      use_bias=False,
                                      activation=None)
      
  def call(self, inputs):
    x = self.conv2a(inputs)
    x = self.batch_norm(x)
    x = self.relu6(x)
    x = self.zero_pad2(x)
    x = self.dwconv2(x)
    x = self.batch_norm(x)
    x = self.relu6(x)
    x = self.conv2b(x)
    x = self.batch_norm(x)
    return x
			
class MobileNetV2(keras.Model):
  def __init__(self, alpha, i_shape, units):
    super(MobileNetV2, self).__init__()
    self.alpha = alpha
    self.i_shape = i_shape
    self.units = units
    self.conv1 = keras.layers.Conv2D(filters=int(self.alpha*32),
                                      kernel_size=[3, 3], 
                                      strides=[2, 1],
                                      use_bias=False, input_shape=self.i_shape)
    self.batch_norm = keras.layers.BatchNormalization(momentum=0.1)
    self.relu6 = keras.layers.ReLU(6.)
    self.block1 = Stride1Block(filters=(int(self.alpha*32), int(self.alpha*32)))
    self.block2 = Stride2Block(filters=(int(self.alpha*64), int(self.alpha*64)))
    self.block3 = Stride1Block(filters=(int(self.alpha*64), int(self.alpha*64)))
    self.block4 = Stride2Block(filters=(int(self.alpha*128), int(self.alpha*128)))
    self.block5 = Stride1Block(filters=(int(self.alpha*128), int(self.alpha*128)))
    self.glopool = keras.layers.GlobalAveragePooling2D()
    self.dense = keras.layers.Dense(units=self.units)

  def call(self, inputs):
    x = self.conv1(inputs)
    x = self.batch_norm(x)
    x = self.relu6(x)
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.glopool(x)
    x = self.dense(x)
    return x
