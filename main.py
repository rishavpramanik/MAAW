import argparse
import tensorflow as tf
import tensorflow.keras as tk
import numpy as np
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
import zipfile
from utils.model import *
from utils.metric import *
from utils.data_generator import *
from MAAW import *

parser = argparse.ArgumentParser()
parser.add_argument('--runtimename', type=str, default = './', help='Directory where the image data is stored')
parser.add_argument('--auto', type=bool, default = False, help='Autometic Parameter Setting According To Runtime')
parser.add_argument('--model_save', type=bool, default = False, help='Autometic Model Save')
parser.add_argument('--batch_size', type=int, default = 256, help='Batch size for training')
parser.add_argument('--height', type=int, default = 32, help='Height Of Image')
parser.add_argument('--width', type=int, default = 32, help='Width Of Image')
parser.add_argument('--init_learning_rate', type=float, default = 0.0001, help='Initial Learning Rate')
args = parser.parse_args()

runtimename = args.runtimename;
model_save_dir = 'saved_models/'

"""

##Data Batching"""
print("=========================== Data Batching ==========================")
print("Current Run Time : "+runtimename)
BATCH_SIZE = args.batch_size;
HEIGHT = args.height;
WIDTH = args.width;
INIT_LEARNING_RATE = args.init_learning_rate

BATCH_SIZE, HEIGHT, WIDTH, INIT_LEARNING_RATE, stage_paramaters = getAutoSettings(runtimename);
train_generator,validation_generator,auto_t_steps_per_epoch,auto_v_steps_per_epoch, NUM_CLASS = data_generator(runtimename, BATCH_SIZE, HEIGHT, WIDTH)

print("----- Automatic Paramters -----")
print("Batch Size : ",BATCH_SIZE)
print("Height : ",HEIGHT)
print("Width : ",WIDTH)
print("INIT_LEARNING_RATE : ",INIT_LEARNING_RATE)
print("Stage Paramters: ",stage_paramaters)
print("-------------------------------")
auto_t_steps_per_epoch = 3
auto_v_steps_per_epoch = 3

test_generator = validation_generator

train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric   = tf.keras.metrics.CategoricalAccuracy()

"""## Run"""
print("=========================== Model Warming UP ==========================")
model = getmodel(runtimename, NUM_CLASS)

x_test, y_test = (next(test_generator))

for i in range(auto_v_steps_per_epoch):
 x_test_temp, y_test_temp = (next(test_generator))
 x_test = np.concatenate((x_test,x_test_temp))
 y_test = np.concatenate((y_test,y_test_temp))

MODELS = []
HISTORYS = []
STAGE = 0

print("=========================== Model Run ==========================")

for alpha,beta,max_epoch,val_acc_thresh in stage_paramaters:
  print("========== STAGE "+str(STAGE)+" ==========")
  STAGE = STAGE+1
  print("****** Current Parameters : alpha = "+str(alpha)+", beta = "+str(beta)+", max_epoch = "+str(max_epoch)+", val_acc_thresh = "+str(val_acc_thresh)+" *******")

  loss_object = SparseCategorical_LSM_DWB_Loss(maj_wt=alpha,min_wt=beta)
  history = train(model,loss_object, train_generator , validation_generator, train_acc_metric, val_acc_metric, epochs = max_epoch,t_steps_per_epoch=auto_t_steps_per_epoch,v_steps_per_epoch=auto_v_steps_per_epoch,val_acc_threshold=val_acc_thresh,INIT_LEARNING_RATE=INIT_LEARNING_RATE)
  MODELS.append(model)
  HISTORYS.append(history)
  if args.model_save == True:
    model.save(model_save_dir+runtimename+"_"+str(int(alpha*100))+"_"+str(int(beta*100))+".h5")
    print("[ Model Saved As "+model_save_dir+runtimename+"_"+str(alpha)+"_"+str(beta)+".h5 ]")
  computePerformance(x_test, y_test, model);
