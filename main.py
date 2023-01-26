import argparse
import tensorflow as tf
import tensorflow.keras as tk
import numpy as np
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
import zipfile
from utils.model import *
from utils.metric import *
from MAAW import *

parser = argparse.ArgumentParser()
parser.add_argument('--data_directory', type=str, default = './', help='Directory where the image data is stored')
parser.add_argument('--epochs', type=int, default = 10, help='Number of Epochs of training')
parser.add_argument('--batch_size', type=int, default = 32, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default = 0.0001, help='Learning Rate')
parser.add_argument('--stepLR', type=int, default=5, help='Step size for Step LR scheduler')
args = parser.parse_args()

runtimename = 'cifar10'
model_save_dir = '/content/LSMDWB_Models/'
base_path_drive = '/content/drive/MyDrive/LSMDWB_DataSet/'
test_dir = '/content/cifar100_test'
train_dir = '/content/cifar100_train'

with zipfile.ZipFile(base_path_drive+"cifar100_test.zip","r") as zip_ref:
    zip_ref.extractall(test_dir)

with zipfile.ZipFile(base_path_drive+"cifar100_train.zip","r") as zip_ref:
    zip_ref.extractall(train_dir)

"""

##Data Batching"""

BATCH_SIZE = 256
HEIGHT = 32
WIDTH = 32
INIT_LEARNING_RATE = 1e-4

train_datagen = ImageDataGenerator(
    rescale = 1./ 255.
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    target_size = (HEIGHT,WIDTH)
);


valid_datagen = ImageDataGenerator(
    rescale = 1./ 255.
)

validation_generator = valid_datagen.flow_from_directory(
    test_dir,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    target_size = (HEIGHT,WIDTH)
);

auto_t_steps_per_epoch = train_generator.n//BATCH_SIZE
auto_v_steps_per_epoch = validation_generator.n//BATCH_SIZE
NUM_CLASS = train_generator.num_classes
test_generator = validation_generator
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric   = tf.keras.metrics.CategoricalAccuracy()

def train_data_for_one_epoch(model,optimizer ,loss_object, train_data, t_steps_per_epoch):
  losses = []
  pbar = tqdm(total=t_steps_per_epoch, position=0, leave=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ')

  for step in range(1,t_steps_per_epoch+1):
      (x_batch_train, y_batch_train) = next(train_data)
      y_batch_train_1d = tf.argmax(y_batch_train,axis=1)
      logits, loss_value = run_optimizer(model,optimizer,loss_object, x_batch_train, y_batch_train_1d)
      
      losses.append(loss_value)
      softMaxLayer = softmax_Layer()
      train_acc_metric(y_batch_train, softMaxLayer(logits))

      pbar.set_description("Training loss for step %s: %.4f" % (int(step), float(loss_value)))
      pbar.update()
  return losses

def perform_validation(model,optimizer ,loss_object, test_data,v_steps_per_epoch):
  losses = []
  for step in range(v_steps_per_epoch):
      x_batch_test, y_batch_test = next(test_data)
      y_batch_test_1d = tf.argmax(y_batch_test,axis=1)

      val_logits = model(x_batch_test)
      w = model.trainable_weights[-2]
      b = model.trainable_weights[-1]
      a = tk.Model(model.input, model.get_layer('BN_Dense_2').output)(x_batch_test) 

      val_loss = loss_object(y_true=y_batch_test_1d, y_pred=softMaxLayer(val_logits), a_pred=a, weight_last_layer=w,bias_last_layer=b)
      losses.append(val_loss)
      val_acc_metric(y_batch_test, softMaxLayer(val_logits))
  return losses

def train(model,loss_object, train , test , epochs = 3,t_steps_per_epoch=128,v_steps_per_epoch=128,val_acc_threshold=1.0):
    history = {}
    history['train_loss'] = []
    history['val_loss'] = []

    history['train_acc'] = []
    history['val_acc'] = []


    lr = INIT_LEARNING_RATE

    for epoch in range(epochs):
        lr = lr_scheduler(lr,epoch)
        optimizer = tk.optimizers.Adam(learning_rate=lr)

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
          print("___ Desired Metric Reached!!")
          break;
    history['model'] = model
    return history

"""## Run"""

model = base_model()

x_test, y_test = (next(test_generator))

for i in range(auto_v_steps_per_epoch):
 x_test_temp, y_test_temp = (next(test_generator))
 x_test = np.concatenate((x_test,x_test_temp))
 y_test = np.concatenate((y_test,y_test_temp))

MODELS = []
HISTORYS = []
STAGE = 0

while True:
  print("========== STAGE "+str(STAGE)+" ==========")
  STAGE = STAGE+1
  alpha = float(input("alpha: "))
  beta = float(input("beta: "))
  max_epoch = int(input("max_epoch: "))
  val_acc_thresh = float(input("val_acc_thresh: "))
  loss_object = SparseCategorical_LSM_DWB_Loss(maj_wt=alpha,min_wt=beta)
  history = train(model,loss_object, train_generator , validation_generator,epochs = max_epoch,t_steps_per_epoch=5,v_steps_per_epoch=5,val_acc_threshold=val_acc_thresh)
  MODELS.append(model)
  HISTORYS.append(history)
  model.save('~ '+model_save_dir+runtimename+str(alpha)+"_"+str(beta)+".h5 ~")
  print("Model Saved As "+model_save_dir+runtimename+str(alpha)+"_"+str(beta)+".h5")
  computePerformance(x_test, y_test, model)
  if input('Do You Want To Continue? y/n') != 'y':
    break