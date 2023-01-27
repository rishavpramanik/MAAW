import tensorflow.keras as K
import tensorflow as tf
import tensorflow.keras as tk
import numpy as np
from tqdm import tqdm

def modifiedcos(theta):
  theta = tf.convert_to_tensor(theta,dtype=tf.float32)
  return tf.where(theta<=3.14159,tf.math.cos(theta),-tf.math.cos(theta)-2)

def softmax_Layer(temp=1,NUM_CLASS=10):
  input = K.Input(shape=(None,NUM_CLASS))
  x = tf.keras.layers.Softmax()(input/temp)
  model = tk.Model(input, x)
  return model

def lr_scheduler(lr,epoch):
  if epoch == 5:
    return lr*10;
  return lr;

def run_optimizer(model,optimizer,loss_object, x , y_true):
    x = tf.convert_to_tensor(x)
    with tf.GradientTape(watch_accessed_variables=True) as tape:
      w = model.trainable_weights[-2]
      b = model.trainable_weights[-1]
      a = tk.Model(model.input, model.get_layer('BN_Dense_2').output)(x)

      logits = model(x)
      softMaxLayer = softmax_Layer()
      y_pred = softMaxLayer(logits)
      L = loss_object(y_true=y_true , y_pred=y_pred, a_pred=a, weight_last_layer=w,bias_last_layer=b)

    grad = tape.gradient(L , model.trainable_weights)
    optimizer.apply_gradients(grads_and_vars = zip(grad , model.trainable_weights))
    del tape
    return logits , L 

def train_data_for_one_epoch(model,optimizer ,loss_object, train_acc_metric, train_data, t_steps_per_epoch):
  losses = []
  pbar = tqdm(total=t_steps_per_epoch, position=0, leave=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ')

  for step in range(1,t_steps_per_epoch+1):
      (x_batch_train, y_batch_train) = next(train_data)
      y_batch_train_1d = tf.argmax(y_batch_train,axis=1)
      logits, loss_value = run_optimizer(model,optimizer,loss_object, x_batch_train, y_batch_train_1d)
      
      losses.append(loss_value)
      train_acc_metric(y_batch_train, softmax_Layer()(logits))

      pbar.set_description("Training loss for step %s: %.4f" % (int(step), float(loss_value)))
      pbar.update()
  return losses

def perform_validation(model,optimizer ,loss_object, val_acc_metric, test_data,v_steps_per_epoch):
  losses = []
  for step in range(v_steps_per_epoch):
      x_batch_test, y_batch_test = next(test_data)
      y_batch_test_1d = tf.argmax(y_batch_test,axis=1)

      val_logits = model(x_batch_test)
      w = model.trainable_weights[-2]
      b = model.trainable_weights[-1]
      a = tk.Model(model.input, model.get_layer('BN_Dense_2').output)(x_batch_test) 

      val_loss = loss_object(y_true=y_batch_test_1d, y_pred=softmax_Layer()(val_logits), a_pred=a, weight_last_layer=w,bias_last_layer=b)
      losses.append(val_loss)
      val_acc_metric(y_batch_test, softmax_Layer()(val_logits))
  return losses

def train(model,loss_object, train , test ,train_acc_metric, val_acc_metric, epochs = 3,t_steps_per_epoch=128,v_steps_per_epoch=128,val_acc_threshold=1.0,INIT_LEARNING_RATE=0.0001):
    history = {}
    history['train_loss'] = []
    history['val_loss'] = []

    history['train_acc'] = []
    history['val_acc'] = []


    lr = INIT_LEARNING_RATE

    for epoch in range(epochs):
        lr = lr*0.98;
        lr = lr_scheduler(lr,epoch)

        optimizer = tk.optimizers.Adam(learning_rate=lr)

        print('Start of epoch %d' % (epoch,))
        train_losses = train_data_for_one_epoch(model ,optimizer,loss_object,train_acc_metric, train_data=train,t_steps_per_epoch=t_steps_per_epoch)
        train_acc    = train_acc_metric.result()
        history['train_acc'].append(train_acc.numpy())
        train_acc_metric.reset_states()

        val_losses   = perform_validation(model,optimizer ,loss_object,val_acc_metric, test_data=test,v_steps_per_epoch=v_steps_per_epoch)
        val_acc      = val_acc_metric.result()
        history['val_acc'].append(val_acc.numpy())
        val_acc_metric.reset_states()

        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))

        print('\n Epoch %s: Train loss: %.4f  Validation Loss: %.4f,\
         Train Accuracy: %.4f, Validation Accuracy %.4f' % (epoch, float(np.mean(train_losses)), float(np.mean(val_losses)),
                                                            float(train_acc), float(val_acc)))
        if float(val_acc) >= val_acc_threshold:
          print("___ Desired Metric Reached!!")
          break;
    history['model'] = model
    return history


def getAutoSettings(runtimename):
  stage_paramaters = []
  batch_size = None;
  height = None
  width = None
  init_lr = None


  if runtimename == "cifar10":
    stage_paramaters.append([1.00,1.00,2,0.70])
    stage_paramaters.append([1.02,0.99,2,0.72])
    stage_paramaters.append([1.04,0.98,2,0.73])
    stage_paramaters.append([1.07,0.97,2,0.75])
    batch_size = 256;
    height = 32;
    width = 32;
    init_lr = 1e-4;
  
  if runtimename == "cifar50":
    stage_paramaters.append([1.00,1.00,10,0.98])
    stage_paramaters.append([1.02,0.99,10,0.98])
    stage_paramaters.append([1.04,0.98,10,0.98])
    stage_paramaters.append([1.07,0.97,10,0.98])
    batch_size = 256;
    height = 32;
    width = 32;
    init_lr = 1e-4;

  if runtimename == "cifar100":
    stage_paramaters.append([1.00,1.00,10,0.98])
    stage_paramaters.append([1.02,0.99,10,0.98])
    stage_paramaters.append([1.04,0.98,10,0.98])
    stage_paramaters.append([1.07,0.97,10,0.98])
    batch_size = 256;
    height = 32;
    width = 32;
    init_lr = 1e-4;

  if runtimename == "fmnist10":
    stage_paramaters.append([1.00,1.00,10,0.88])
    stage_paramaters.append([1.05,0.98,10,0.89])
    stage_paramaters.append([1.10,0.96,10,0.89])
    batch_size = 256;
    height = 28;
    width = 28;
    init_lr = 1e-3;

  if runtimename == "fmnist50":
    stage_paramaters.append([1.00,1.00,10,0.84])
    stage_paramaters.append([1.05,0.98,10,0.85])
    stage_paramaters.append([1.10,0.96,10,0.85])
    batch_size = 256;
    height = 28;
    width = 28;
    init_lr = 1e-3;
    
  if runtimename == "fmnist100":
    stage_paramaters.append([1.00,1.00,10,0.80])
    stage_paramaters.append([1.05,0.98,10,0.81])
    stage_paramaters.append([1.10,0.96,10,0.82])
    batch_size = 256;
    height = 28;
    width = 28;
    init_lr = 1e-3;

  if runtimename == "ham10000":
    stage_paramaters.append([1.00,1.00,20,0.80])
    stage_paramaters.append([1.02,0.99,20,0.82])
    stage_paramaters.append([1.04,0.98,20,0.84])
    batch_size = 32;
    height = 224;
    width = 224;
    init_lr = 1e-4;

  if runtimename == "aptos":
    stage_paramaters.append([1.00,1.00,10,0.80])
    stage_paramaters.append([1.05,0.99,10,0.81])
    stage_paramaters.append([1.10,0.98,10,0.82])
    stage_paramaters.append([1.15,0.97,10,0.83])
    stage_paramaters.append([1.20,0.97,10,0.83])
    batch_size = 64;
    height = 224;
    width = 224;
    init_lr = 1e-4;

  return batch_size, height, width, init_lr, stage_paramaters;