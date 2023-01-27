import sklearn.metrics
import imblearn.metrics
from sklearn.metrics import cohen_kappa_score
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils.helper import *

def computePerformance(x, y,model):
    print("========== Validation Starts ==========")
    y_pred = model.predict(x)
    y_pred = softmax_Layer()(y_pred)
    y_pred_classes = np.argmax(y_pred, axis = 1) 
    y_true_classes = np.argmax(y, axis = 1) 
    
    print('*********** avg_pre: ',sklearn.metrics.precision_score(y_true_classes, y_pred_classes, average='macro'));
    print('*********** avg_rec: ',sklearn.metrics.recall_score(y_true_classes, y_pred_classes, average='macro'));
    print('*********** avg_f1: ',sklearn.metrics.f1_score(y_true_classes, y_pred_classes, average='macro'));
    print('*********** avg_spe: ',imblearn.metrics.specificity_score(y_true_classes, y_pred_classes, average='macro'));
    print('*********** avg_geo: ',imblearn.metrics.geometric_mean_score(y_true_classes, y_pred_classes, average='macro'));

    mcc = sklearn.metrics.matthews_corrcoef(y_true = y_true_classes, y_pred = y_pred_classes)
    kp = cohen_kappa_score(y_true_classes, y_pred_classes)
    auc_metric = tf.keras.metrics.AUC()
    auc_metric.update_state(y, y_pred) 
    auc = auc_metric.result().numpy()


    print('*********** MCC: ',mcc)
    print('*********** Cohen Kappa Score: ',kp)
    print('*********** AUC/ROC: ',auc)

"""#### Loss And Accuracy"""

def plot_metrics(history, metric_name, title):
    plt.title(title)
    plt.plot(history['train_' + metric_name],color='blue',label='train_' +metric_name)
    plt.plot(history['val_' + metric_name],color='green',label='val_' + metric_name)
    plt.show()