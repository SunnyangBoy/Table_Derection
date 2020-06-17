import os
import numpy as np
import tensorflow as tf
from model import resnet18
import tensorflow_hub as hub
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

summary_writer = tf.summary.create_file_writer('./tensorboard')

batch_size = 8
epoches = 50
learning_rate = 0.0001


def dataloder(img_root, txt_path):
    img_paths = []
    labels = []
    with open(txt_path, 'r') as reader:
        lines = reader.readlines()
        for line in lines:
            line = line.split(';')
            img_name = line[0]
            angel = line[1][:-1]
            img_path = os.path.join(img_root, img_name)
            img_paths.append(img_path)
            labels.append(int(angel))
    return img_paths, labels


def decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename)  # 读取原始文件
    image_decoded = tf.image.decode_jpeg(image_string)  # 解码JPEG图片

    image_emhance = tf.image.random_brightness(image_decoded, 0.3)
    image_emhance = tf.image.random_contrast(image_emhance, 0.7, 1.3)
    image_emhance = tf.image.central_crop(image_emhance, 0.6)

    image_resized = tf.image.resize(image_emhance, [600, 600]) / 255.0

    label = tf.expand_dims(label, axis=-1)
    label = tf.cast(label, dtype=tf.float32)
    return image_resized, label


class MeanSquaredError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        '''
        sum = tf.constant(0)
        for i, ang in enumerate(y_pred):
            if ang[0] > 0 and ang[0] < 360:
                thresh = (y_true[i][0] + 180) % 360
                if thresh < y_true[i][0]:
                    if ang[0] > 0 and ang[0] < thresh:
                        sum += tf.square(ang[0] + 360 - y_true[i][0])
                    else:
                        sum += tf.square(ang[0] - y_true[i][0])
                else:
                    if ang[0] > thresh and ang[0] < 360:
                        sum += tf.square(2 * thresh - ang[0] - y_true[i][0])
                    else:
                        sum += tf.square(ang[0] - y_true[i][0])
            else:
                sum += tf.square(ang[0] - y_true[i][0])
        loss = sum
        '''
        #min_y_pred = tf.where(y_pred >= 0, y_pred, 0)
        #max_y_pred = tf.where(min_y_pred <= 360, min_y_pred, 360)
        a = abs(y_pred - y_true)
        b = 360 - abs(y_pred - y_true)
        tmp = tf.concat([a, b], -1)
        loss = tf.keras.backend.min(tmp, -1, keepdims=True)
        loss = tf.reduce_mean(tf.square(loss))
        # loss = tf.reduce_mean(tf.square(y_true-y_pred))
        return loss


if __name__ == '__main__':
    img_root = '/root/ais/shanghai_bank_poc/train_rotated3/'
    txt_path = '/root/ais/shanghai_bank_poc/angles3.txt'

    # 数据加载
    train_filenames, train_labels = dataloder(img_root, txt_path)

    train_filenames = tf.constant(train_filenames)
    train_labels = tf.constant(train_labels)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(
        map_func=decode_and_resize,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # 取出前buffer_size个数据放入buffer，并从其中随机采样，采样后的数据用后续数据替换
    train_dataset = train_dataset.shuffle(buffer_size=400)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    #with strategy.scope():
    model = tf.keras.Sequential([
        hub.KerasLayer("https://hub.tensorflow.google.cn/google/imagenet/resnet_v2_50/feature_vector/4",
                       trainable=True, arguments=dict(batch_norm_momentum=0.997)),
        tf.keras.layers.Dense(units=1024, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=1)
    ])
    model.build([None, 600, 600, 3])  # Batch input shape.

    # model = resnet18()
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    checkpoint = tf.train.Checkpoint(myModel=model)
    batch_index = 0
    for epoch in range(epoches):
        for step, (images, labels) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                pre_angles = model(images)
                #my_loss = MeanSquaredError()
                #loss = my_loss(labels, pre_angles)
                #loss = tf.where(pre_angles[pre_angles < 0].numpy().size > 0 or pre_angles[pre_angles > 360].numpy().size > 0, tf.reduce_mean(tf.keras.losses.MSE(labels, pre_angles)), my_loss(labels, pre_angles))
                loss = tf.reduce_mean(tf.keras.losses.MSE(labels, pre_angles))
                batch_index += 1

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 30 == 0:
                print('pre_angles: ', pre_angles)
                print('labels: ', labels)
                print('epoch{}  iter:{}/{}  per_loss:{:.4f}'.format(epoch, step, int(len(train_labels) / batch_size),
                                                                    loss))

            with summary_writer.as_default():  # 希望使用的记录器
                tf.summary.scalar("loss", loss, step=batch_index)

        if epoch % 5 == 0:
            path = checkpoint.save('./checkpoints2/model.ckpt')  # 保存模型参数到文件
            print("model saved to %s" % path)
