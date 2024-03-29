import tensorflow.keras as K
import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.backend as K_

# Helper functions
from utils.helper import *

# Angle Shifted Predictions
def getShiftedPred(y_true,y_pred, a_pred, weight_last_layer,bias_last_layer,maj_wt, min_wt):
  """
  This function calculates the angle shifted predictions using the following steps:
  1. One-hot encode the true labels
  2. Calculate the norm of the predictions and the last layer weights
  3. Calculate the dot product of the predictions and the last layer weights
  4. Calculate the cosine of the angle between the predictions and the last layer weights
  5. Apply the majority and minority class weights to the angle
  6. Use the modified cosine function to calculate the final cosine
  7. Calculate the softmax predictions by applying the final cosine to the last layer weights and adding the bias
  """
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
  th_= M*th_
  cos_t = modifiedcos(th_)
  softMaxLayer = softmax_Layer()
  z = tf.convert_to_tensor((norm_weight_last_layer_ * norm_a_pred_)*cos_t + bias_last_layer)
  soft_prediction = softMaxLayer(z) 
  return soft_prediction

def sparseCategorcial_maaw_loss(y_true,y_pred, a_pred, weight_last_layer,bias_last_layer,maj_wt=1.0, min_wt=1.0):
  """
  y_true: the true labels for the data
  y_pred: the predicted labels from the model
  a_pred: the activations from the penultimate layer of the model
  weight_last_layer: the weights of the last layer of the model
  bias_last_layer: the biases of the last layer of the model
  maj_wt: a weighting factor for the majority class (default value is 1.0)
  min_wt: a weighting factor for the minority class (default value is 1.0)
  """
  #Convert inputs to tensor
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.convert_to_tensor(y_true)
  a_pred = tf.convert_to_tensor(a_pred)
  
  # Number of labels
  num_labels = y_pred[0].shape[0]
  
  # One hot encoding of y_true
  y_true_oh = tf.one_hot(y_true,num_labels)
  
  # Get shifted predictions
  fin_preds = getShiftedPred(y_true,y_pred, a_pred, weight_last_layer,bias_last_layer,maj_wt, min_wt)
  
  #clip the values to avoid Nans
  fin_preds = K_.clip(fin_preds, K_.epsilon(), 1 - K_.epsilon())
  
  #Calculating frequency of each class
  f = tf.convert_to_tensor([tf.math.count_nonzero(y_true==i) for i in range(num_labels)],dtype=tf.float32)+1e-9
  
  #Convert f to float
  f = tf.constant([float(f[i]) for i in y_true])
  
  #Calculating class weight
  wt = 1-1/f
  
  #Calculating scaled power of class weight
  wt_scaled_pow = tf.pow(tf.math.subtract(1.0, fin_preds), tf.transpose(wt * tf.transpose(y_true_oh)))
  
  #Calculating Loss
  temp = tf.multiply(y_true_oh, K_.log(fin_preds))
  loss = tf.multiply(wt_scaled_pow, temp)
  
  #Calculating cross entropy
  xcent = -1./f*tf.reduce_sum(loss,axis=-1)
  
  #Calculating mean of cross entropy
  xcent_mean =  tf.reduce_mean(xcent)
  return xcent_mean


class SparseCategorcial_MAAW_Loss:
  def __init__(self, maj_wt=1.0, min_wt=1.0):
    self.maj_wt = maj_wt
    self.min_wt = min_wt
    pass
  def __call__(self, y_true,y_pred, a_pred, weight_last_layer,bias_last_layer):
    return sparseCategorcial_maaw_loss(y_true,y_pred, a_pred, weight_last_layer,bias_last_layer,self.maj_wt, self.min_wt)
