import tensorflow as tf
import cv2
import os
import time
import math
import numpy as np
from tensorflow.python.framework import graph_util


def infer_with_pb_model(pb_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        # 读取pb模型
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")

            # 运行tensorflow 进行预测
            with tf.Session() as sess:
                # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字
                names = ['output_angle/Sigmoid:0']
                output_tensors = []
                for n in names:
                    output_tensors.append(tf.get_default_graph().get_tensor_by_name(n))

                # image_input = np.zeros((600, 600, 3))
                # image_input = np.expand_dims(image_input, 0)
                image_string = tf.io.read_file("/home/ubuntu/cs/table_derection_tf2/use_cos_loss/train_data/rot_images/00070_10.jpg")  # 读取原始文件
                image_decoded = tf.image.decode_jpeg(image_string)  # 解码JPEG图片
                # image_emhance = tf.image.per_image_standardization(image_emhance)
                # image_emhance = tf.expand_dims(image_emhance, axis=0)
                # image_emhance = tf.image.resize_bilinear(image_emhance, [600, 600])
                # image_input = image_emhance.eval()
                image_resized = tf.image.resize(image_decoded, [600, 600]) / 255.0

                print(sess.run(output_tensors, feed_dict={"input_1:0": image_resized}))
                print('finished !!!')


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
infer_with_pb_model("/home/ubuntu/cs/table_derection_tf2/use_cos_loss/saved_model_tf2/pb/saved_model.pb")