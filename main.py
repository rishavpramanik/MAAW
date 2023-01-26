import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import tensorflow.keras as K
import tensorflow.keras as tk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from tqdm import tqdm
import tensorflow_datasets as tfds
import matplotlib.ticker as mticker
import tensorflow.keras.backend as K_
from tensorflow.python.ops.script_ops import numpy_function

#############################################################################################

def makeImablanced(data,num_classes, irs):
  n = int(data.shape[0]/num_classes)
  unb_data = data.T[0:int(n*irs[0])];
  prev = n;
  for i in range(1,num_classes):
    unb_data = np.concatenate([unb_data, data.T[prev:(prev+int(n*irs[i]))]]);
    prev = n*(i+1);
  return unb_data.T;



(x_train, y_train), (x_test, y_test) = K.datasets.fashion_mnist.load_data()

irs = [0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0];
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


############################################################################################# MODEL

def softmax_Layer():
  input = K.Input(shape=(None,10));
  x = tf.keras.layers.Softmax()(input)
  model = K.Model(input, x)
  return model

softMaxLayer = softmax_Layer()

def base_model():
  model = K.Sequential()
  model.add(K.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
  model.add(K.layers.MaxPooling2D((2, 2)))
  model.add(K.layers.Flatten())
  model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
  model.add(K.layers.Dropout(0.4))
  model.add(K.layers.BatchNormalization(name='BN_Dense_2'))
  model.add(Dense(10, activation=None))
	# compile model
	# opt = SGD(lr=0.01, momentum=0.9)
	# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def modifiedCOS(theta):
  theta = tf.convert_to_tensor(theta,dtype=tf.float32)
  return tf.where(theta<=3.14159,tf.math.cos(theta),-tf.math.cos(theta)-2)

# Angle Shifted Predictions
def getShiftedPred(y_true,y_pred, a_pred, weight_last_layer,bias_last_layer,maj_wt, min_wt):
  N = y_pred.shape[0]
  num_labels = y_pred[0].shape[0]

  y_true_oh = tf.one_hot(y_true,num_labels)

  a_pred = tf.convert_to_tensor(a_pred)
  weight_last_layer = tf.convert_to_tensor(weight_last_layer)
  bias_last_layer = tf.convert_to_tensor(bias_last_layer)

  norm_a_pred_ = tf.expand_dims(tf.linalg.norm(a_pred,axis=1),axis=1)
  norm_weight_last_layer_ = tf.expand_dims(tf.linalg.norm(weight_last_layer,axis=0),axis=0)
  pr_ = tf.matmul(a_pred,weight_last_layer)
  cos_t = ((pr_ / norm_a_pred_)/ norm_weight_last_layer_)
  th_ = tf.math.acos(cos_t)

  M = tf.where(y_true_oh ==1 , maj_wt, min_wt)
  th_= M*th_;
  cos_t = modifiedCOS(th_)
  z = tf.convert_to_tensor((norm_weight_last_layer_ * norm_a_pred_)*cos_t + bias_last_layer)

  soft_prediction = softMaxLayer(z) 

  return soft_prediction

############################################################################################# LOSS
def sparseCategorical_LSM_DWB_Loss(y_true,y_pred, a_pred, weight_last_layer,bias_last_layer,maj_wt=1.0, min_wt=1.0):
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.convert_to_tensor(y_true)
  a_pred = tf.convert_to_tensor(a_pred)

  num_labels = y_pred[0].shape[0]
  y_true_oh = tf.one_hot(y_true,num_labels)

  fin_preds = getShiftedPred(y_true,y_pred, a_pred, weight_last_layer,bias_last_layer,maj_wt, min_wt)
  fin_preds = K_.clip(fin_preds, K_.epsilon(), 1 - K_.epsilon())

  f = tf.convert_to_tensor([tf.math.count_nonzero(y_true==i) for i in range(num_labels)],dtype=tf.float32)+1e-9
  f = tf.constant([float(f[i]) for i in y_true])

  # wt = 1./f
  #wt = K_.log(tf.math.divide(K_.max(f), f))+1
  wt = 1-1/f
  wt_scaled_pow = tf.pow(tf.math.subtract(1.0, fin_preds), tf.transpose(wt * tf.transpose(y_true_oh)))
  temp = tf.multiply(y_true_oh, K_.log(fin_preds))
  reg_term = 0#fin_preds * (tf.math.subtract(1,fin_preds))

  temp = tf.multiply(wt_scaled_pow, temp)
  loss = tf.math.subtract(temp, reg_term)

  xcent = -1./f*tf.reduce_sum(loss,axis=-1)
  xcent_mean =  tf.reduce_mean(xcent)

  return xcent_mean


class SparseCategorical_LSM_DWB_Loss:
  def __init__(self, maj_wt=1.0, min_wt=1.0):
    self.maj_wt = maj_wt
    self.min_wt = min_wt
    pass
  def __call__(self, y_true,y_pred, a_pred, weight_last_layer,bias_last_layer):
    return sparseCategorical_LSM_DWB_Loss(y_true,y_pred, a_pred, weight_last_layer,bias_last_layer,self.maj_wt, self.min_wt);

############################################################################################# TRAIN

train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric   = tf.keras.metrics.CategoricalAccuracy()

def run_optimizer(model,optimizer,loss_object, x , y_true):
    x = tf.convert_to_tensor(x)
    with tf.GradientTape(watch_accessed_variables=True) as tape:
      w = model.trainable_weights[-2]
      b = model.trainable_weights[-1]
      a = tk.Model(model.input, model.get_layer('BN_Dense_2').output)(x)

      logits = model(x)
      y_pred = softMaxLayer(logits)
      loss_val = loss_object(y_true=y_true , y_pred=y_pred, a_pred=a, weight_last_layer=w,bias_last_layer=b)
    grad = tape.gradient(loss_val , model.trainable_weights)
    optimizer.apply_gradients(grads_and_vars = zip(grad , model.trainable_weights))

    return logits , loss_val

def train_data_for_one_epoch(model,optimizer ,loss_object, train_data, t_steps_per_epoch):
  losses = []
  pbar = tqdm(total=t_steps_per_epoch, position=0, leave=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ')

  for step in range(1,t_steps_per_epoch+1):
      (x_batch_train, y_batch_train) = next(train_data)
      y_batch_train_1d = tf.argmax(y_batch_train,axis=1);
      logits, loss_value = run_optimizer(model,optimizer,loss_object, x_batch_train, y_batch_train_1d)
      
      losses.append(loss_value)
      
      train_acc_metric(y_batch_train, softMaxLayer(logits))

      pbar.set_description("Training loss for step %s: %.4f" % (int(step), float(loss_value)))
      pbar.update()
  return losses

def perform_validation(model,optimizer ,loss_object, test_data,v_steps_per_epoch):
  losses = []
  for step in range(v_steps_per_epoch):
      x_batch_test, y_batch_test = next(test_data);
      y_batch_test_1d = tf.argmax(y_batch_test,axis=1);

      val_logits = model(x_batch_test)
      w = model.trainable_weights[-2]
      b = model.trainable_weights[-1]
      a = tk.Model(model.input, model.get_layer('BN_Dense_2').output)(x_batch_test) 

      val_loss = loss_object(y_true=y_batch_test_1d, y_pred=softMaxLayer(val_logits), a_pred=a, weight_last_layer=w,bias_last_layer=b)
      losses.append(val_loss)
      val_acc_metric(y_batch_test, softMaxLayer(val_logits))
  return losses


def lr_scheduler(lr,epoch):
  if epoch == 5:
    return 0.0001
  return lr

def train(model,loss_object, train , test , epochs = 3,t_steps_per_epoch=128,v_steps_per_epoch=128,val_acc_threshold=1.0):
    history = {}
    history['train_loss'] = []
    history['val_loss'] = []

    history['train_acc'] = []
    history['val_acc'] = []

    val_epoch_loss   = []

    lr = 0.001;

    for epoch in range(epochs):
        lr = lr_scheduler(lr,epoch)
        optimizer = K.optimizers.Adam(learning_rate=lr)

        print('Start of epoch %d' % (epoch,))
        train_losses = train_data_for_one_epoch(model ,optimizer,loss_object, train_data=train,t_steps_per_epoch=t_steps_per_epoch)
        train_acc    = train_acc_metric.result()
        history['train_acc'].append(train_acc.numpy())
        train_acc_metric.reset_states()

        val_losses   = perform_validation(model,optimizer ,loss_object, test_data=test,v_steps_per_epoch=v_steps_per_epoch)
        val_acc      = val_acc_metric.result()
        history['val_acc'].append(val_acc.numpy())
        val_acc_metric.reset_states()

        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))

        print('\n Epoch %s: Train loss: %.4f  Validation Loss: %.4f,\
         Train Accuracy: %.4f, Validation Accuracy %.4f' % (epoch, float(np.mean(train_losses)), float(np.mean(val_losses)),
                                                            float(train_acc), float(val_acc)))
        if float(val_acc) >= val_acc_threshold:
          print("___ Desired ACC Reached!!")
          break;
    history['model'] = model
    return history


############################################################################################# Model Executuion

model = base_model() 

loss_object = SparseCategorical_LSM_DWB_Loss()
history = train(model,loss_object, train_generator , test_generator,epochs = 10,t_steps_per_epoch=auto_t_steps_per_epoch,v_steps_per_epoch=auto_v_steps_per_epoch,val_acc_threshold=0.89)

loss_object = SparseCategorical_LSM_DWB_Loss(maj_wt=1.10,min_wt=0.95)
history = train(model,loss_object, train_generator , test_generator,epochs = 10,t_steps_per_epoch=auto_t_steps_per_epoch,v_steps_per_epoch=auto_v_steps_per_epoch,val_acc_threshold=0.89)
