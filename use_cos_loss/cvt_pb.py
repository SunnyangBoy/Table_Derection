from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import math
import os.path as osp
import tensorflow.keras.backend as K
from tensorflow.python.framework import graph_util, graph_io


def cosloss(y_true, y_pred):
    y_true = tf.Print(y_true, ['y_true: ', y_true])
    y_pred2 = y_pred * 2. * math.pi
    y_pred2 = tf.Print(y_pred2, ['y_pred: ', y_pred2])
    loss = K.mean(2. * (1. - tf.cos(0.5 * (y_pred2 - y_true))))
    loss = tf.Print(loss, ['my loss: ', loss])
    return loss


def freeze_graph(graph, session, output):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, ".", "frozen_model.pb", as_text=False)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 路径参数
h5_dir = '/home/ubuntu/cs/table_derection_tf2/use_cos_loss/saved_model_bnwrong'
h5_file = 'model_51.hdf5'
h5_file_path = osp.join(h5_dir, h5_file)

# 重要的环境配置，在图中有BN这种层时非常重要，设置为推理模式
K.clear_session()
K.set_learning_phase(0)  # this line most important
session = K.get_session()

# 加载模型
h5_model = load_model(h5_file_path,  custom_objects={'cosloss': cosloss})

'''
g_list = tf.global_variables()
bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
print(bn_moving_vars)
'''

# 打印输出输出结点，在load pb时用到
INPUT_NODE = h5_model.inputs[0].op.name
OUTPUT_NODE = [out.op.name for out in h5_model.outputs]
print('输入结点名称：', INPUT_NODE)
print('输出结点名称：', OUTPUT_NODE)

# 生成pb
freeze_graph(session.graph, session, [out.op.name for out in h5_model.outputs])

print('h5 cvt pb finished!!!')