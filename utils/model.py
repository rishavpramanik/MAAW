import tensorflow.keras as K
import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.layers as tkl

#Model for CIFAR10
def base_model_cifar(NUM_CLASS):
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

def base_model_aptos():
  input_tensor = tkl.Input(shape=(224,224,3))
  DenseNet121_model = tk.applications.DenseNet121(include_top=False,weights='imagenet',input_tensor=input_tensor) 
  p  = tkl.GlobalAveragePooling2D()(DenseNet121_model.output)
  d11 = tkl.Dense(units = 512, activation = 'relu',kernel_regularizer= tk.regularizers.l2(0.0001),name='Dense_last')(p) # 256 0.0001
  d11 = tkl.Dropout(0.3)(d11)
  d11 = tkl.BatchNormalization(name='BN_Dense_2')(d11)
  o1 = tkl.Dense(units = 5, activation = None)(d11)
  model = tk.Model(inputs = input_tensor,outputs = [o1,d11])
  return model

def base_model_fmnist():
  model = K.Sequential()
  model.add(K.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
  model.add(K.layers.MaxPooling2D((2, 2)))
  model.add(K.layers.Flatten())
  model.add(tkl.Dense(100, activation='relu', kernel_initializer='he_uniform'))
  model.add(K.layers.Dropout(0.4))
  model.add(K.layers.BatchNormalization(name='BN_Dense_2'))
  model.add(tkl.Dense(10, activation=None))
  return model

def base_model_ham():
  pre_trained_model =  tf.keras.applications.inception_v3.InceptionV3(include_top=False , weights='imagenet' , input_shape=(224,224,3))
  for layers in pre_trained_model.layers:
    layers.trainable = True
  last_layer = pre_trained_model.get_layer('mixed10')
  x = tkl.MaxPool2D(pool_size=(2,2))(last_layer.output)
  x = tkl.GlobalMaxPool2D()(x) ##############
  x = tkl.Dense(256, activation='relu')(x)
  x = tkl.Dropout(0.2)(x)
  x = tk.layers.BatchNormalization(name='BN_Dense_1')(x)
  x = tkl.Dense(128, activation='relu')(x)
  x = tkl.Dropout(0.1)(x)
  x = tk.layers.BatchNormalization(name='BN_Dense_2')(x)
  x = K.layers.Dense(7,activation= None)(x)
  model = tk.Model(pre_trained_model.input, x)
  return model

def getmodel(runtimename, NUM_CLASS):
    if runtimename == 'cifar10' or runtimename == 'cifar50' or runtimename == 'cifar100':
      return base_model_cifar(NUM_CLASS);
    if runtimename == 'fmnist10' or runtimename == 'fmnist50' or runtimename == 'fmnist100':
      return base_model_fmnist();
    if runtimename == 'ham10000':
      return base_model_ham();
    if runtimename == 'aptos':
      return base_model_aptos();