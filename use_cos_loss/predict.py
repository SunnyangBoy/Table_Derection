from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import math
import os.path as osp
import numpy as np
import tensorflow.keras.backend as K


def cosloss(y_true, y_pred):
    y_true = tf.Print(y_true, ['y_true: ', y_true])
    y_pred2 = y_pred * 2. * math.pi
    y_pred2 = tf.Print(y_pred2, ['y_pred: ', y_pred2])
    loss = K.mean(2. * (1. - tf.cos(0.5 * (y_pred2 - y_true))))
    loss = tf.Print(loss, ['my loss: ', loss])
    return loss


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 路径参数
h5_dir = '/home/ubuntu/cs/table_derection_tf2/use_cos_loss/saved_model'
h5_file = 'model_51.hdf5'
h5_file_path = osp.join(h5_dir, h5_file)

# 加载模型
h5_model = load_model(h5_file_path,  custom_objects={'cosloss': cosloss})

# image_input = np.zeros((600, 600, 3))
# image_input = np.expand_dims(image_input, 0)
image_input = np.random.rand(1, 600, 600, 3)
print(h5_model.predict(image_input))
