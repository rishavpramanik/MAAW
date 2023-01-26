import tensorflow.keras as K
import tensorflow as tf
import tensorflow.keras as tk

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
    return 0.0001
  return lr

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