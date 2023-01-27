from keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras as K
import tensorflow as tf
import tensorflow.keras as tk
from tensorflow.python.ops.script_ops import numpy_function
import numpy as np

def fmnistutil(irs):
  def makeImablanced(data,num_classes, irs):
    n = int(data.shape[0]/num_classes)
    unb_data = data.T[0:int(n*irs[0])];
    prev = n;
    for i in range(1,num_classes):
      unb_data = np.concatenate([unb_data, data.T[prev:(prev+int(n*irs[i]))]]);
      prev = n*(i+1);
    return unb_data.T;



  (x_train, y_train), (x_test, y_test) = K.datasets.fashion_mnist.load_data()
  y_train = np.expand_dims(y_train,axis=1)
  y_test = np.expand_dims(y_test,axis=1)
  sorted_index_train = np.argsort(y_train, axis=0).T[0]
  sorted_index_test = np.argsort(y_test, axis=0).T[0]

  imb_sorted_index_train = makeImablanced(sorted_index_train,10,irs);
  imb_sorted_index_test =  makeImablanced(sorted_index_test,10,irs);

  x_train_imb = x_train[imb_sorted_index_train]
  y_train_imb = y_train[imb_sorted_index_train] 

  x_test_imb = x_test#[imb_sorted_index_test]       ##################### NO SLICING IN TEST
  y_test_imb = y_test#[imb_sorted_index_test]



  def batch_generator(X, Y, batch_size = 128):
      indices = np.arange(len(X)) 
      batch=[]
      while True:
              # it might be a good idea to shuffle your data before each epoch
              np.random.shuffle(indices) 
              for i in indices:
                  batch.append(i)
                  if len(batch)==batch_size:
                      yield X[batch], Y[batch]
                      batch=[]


  batch_size_train = 256
  y_train_imb = K.utils.to_categorical(y_train_imb)               
  x_train_imb = np.expand_dims((x_train_imb)/255.0,axis=-1)       
  train_generator = batch_generator(x_train_imb, y_train_imb, batch_size = 128)   

  batch_size_test = 256
  y_test_imb = K.utils.to_categorical(y_test_imb)
  x_test_imb = np.expand_dims((x_test_imb)/255.0,axis=-1)
  test_generator = batch_generator(x_test_imb, y_test_imb, batch_size = 128)

  auto_t_steps_per_epoch = x_train_imb.shape[0]//batch_size_train
  auto_v_steps_per_epoch = x_test_imb.shape[0]//batch_size_test
  return train_generator,test_generator,auto_t_steps_per_epoch,auto_v_steps_per_epoch,10



def data_generator(runtimename, BATCH_SIZE, HEIGHT, WIDTH):
    train_datagen = ImageDataGenerator(
        rescale = 1./ 255.
    )
 
    valid_datagen = ImageDataGenerator(
        rescale = 1./ 255.
    )

    if runtimename == 'aptos':
        pass
    if runtimename == 'cifar10':
        train_generator = train_datagen.flow_from_directory(
            'Data/cifar10/cifar10_train',
            batch_size = BATCH_SIZE,
            class_mode = 'categorical',
            target_size = (HEIGHT,WIDTH)
        );

        validation_generator = valid_datagen.flow_from_directory(
            'Data/cifar10/cifar10_test',
            batch_size = BATCH_SIZE,
            class_mode = 'categorical',
            target_size = (HEIGHT,WIDTH)
        );
        return train_generator,validation_generator,train_generator.n//BATCH_SIZE,validation_generator.n//BATCH_SIZE,train_generator.num_classes;

    if runtimename == 'cifar50':
        train_generator = train_datagen.flow_from_directory(
            'Data/cifar50/cifar50_train',
            batch_size = BATCH_SIZE,
            class_mode = 'categorical',
            target_size = (HEIGHT,WIDTH)
        );

        validation_generator = valid_datagen.flow_from_directory(
            'Data/cifar50/cifar50_test',
            batch_size = BATCH_SIZE,
            class_mode = 'categorical',
            target_size = (HEIGHT,WIDTH)
        );
        return train_generator,validation_generator,train_generator.n//BATCH_SIZE,validation_generator.n//BATCH_SIZE,train_generator.num_classes;
    if runtimename == 'cifar100':
        train_generator = train_datagen.flow_from_directory(
            'Data/cifar100/cifar100_train',
            batch_size = BATCH_SIZE,
            class_mode = 'categorical',
            target_size = (HEIGHT,WIDTH)
        );

        validation_generator = valid_datagen.flow_from_directory(
            'Data/cifar100/cifar100_test',
            batch_size = BATCH_SIZE,
            class_mode = 'categorical',
            target_size = (HEIGHT,WIDTH)
        );
        return train_generator,validation_generator,train_generator.n//BATCH_SIZE,validation_generator.n//BATCH_SIZE,train_generator.num_classes;
        
    if runtimename == 'fmnist10':
        return fmnistutil([0.1, 0.1, 0.1, 0.1, 0.01, 1.0, 1.0, 1.0, 1.0, 1.0])
    if runtimename == 'fmnist50':
        return fmnistutil([0.02, 0.02, 0.02, 0.02, 0.02, 1.0, 1.0, 1.0, 1.0, 1.0])
    if runtimename == 'fmnist100':
        return fmnistutil([0.01, 0.01, 0.01, 0.01, 0.01, 1.0, 1.0, 1.0, 1.0, 1.0])
    if runtimename == 'ham10000':
        train_generator = train_datagen.flow_from_directory(
            'Data/ham10000/Reduced_Train_Unbalanced_DataSet',
            batch_size = BATCH_SIZE,
            class_mode = 'categorical',
            target_size = (HEIGHT,WIDTH)
        );

        validation_generator = valid_datagen.flow_from_directory(
            'Data/ham10000/Reduced_Test_Unbalanced_DataSet',
            batch_size = BATCH_SIZE,
            class_mode = 'categorical',
            target_size = (HEIGHT,WIDTH)
        );
        return train_generator,validation_generator,train_generator.n//BATCH_SIZE,validation_generator.n//BATCH_SIZE,train_generator.num_classes;

    if runtimename == 'aptos':
        datagen=ImageDataGenerator(rescale=1./255., validation_split=0.2)
        src_dir = 'Data/aptos'

        train_generator=datagen.flow_from_directory(
            src_dir,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            target_size=(HEIGHT, WIDTH),
            subset='training')

        valid_generator=datagen.flow_from_directory(
            src_dir,
            batch_size=BATCH_SIZE,
            class_mode="categorical",    
            target_size=(HEIGHT, WIDTH),
            subset='validation')  
        return train_generator,validation_generator,train_generator.n//BATCH_SIZE,validation_generator.n//BATCH_SIZE,train_generator.num_classes;