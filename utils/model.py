import tensorflow.keras as K
import tensorflow as tf
import tensorflow.keras as tk

#Model for CIFAR10
def base_model():
  pre_trained_model =  tf.keras.applications.densenet.DenseNet121(include_top=False , weights='imagenet' , input_shape=(32,32,3))
  for layers in pre_trained_model.layers:
    layers.trainable = True
  last_layer = pre_trained_model.get_layer('relu')
  x = (last_layer.output)
  x = K.layers.GlobalMaxPool2D()(x)
  x = K.layers.Dropout(0.4)(x)
  x = K.layers.BatchNormalization(name='BN_Dense_2')(x)
  x = K.layers.Dense(NUM_CLASS,activation= None)(x)
  model = K.Model(pre_trained_model.input, x)
  return model