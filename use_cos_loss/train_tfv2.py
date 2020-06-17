import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import random
import math
from tensorflow.keras.mixed_precision import experimental as mixed_precision


def dataloder(img_root, txt_path):
    img_paths = []
    labels = []
    info = []
    with open(txt_path, 'r') as reader:
        lines = reader.readlines()
        for line in lines:
            info.append(line)
        random.shuffle(info)
    for ele in info:
        line = ele.split(';')
        img_name = line[0]
        angel = float(line[1][:-1])
        img_path = os.path.join(img_root, img_name)
        img_paths.append(img_path)
        labels.append(angel)
    return img_paths, labels


def decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename)    # 读取原始文件
    image_emhance = tf.image.decode_jpeg(image_string)  # 解码JPEG图片

    # image_emhance = tf.image.random_brightness(image_emhance, 0.3)
    # image_emhance = tf.image.random_contrast(image_emhance, 0.7, 1.3)
    image_resized = tf.image.resize(image_emhance, [600, 600]) / 255.0

    label = tf.expand_dims(label, axis=-1)
    label = tf.cast(label, dtype=tf.float32)
    return image_resized, label


def make_dataset(img_root, txt_path, batch_size):
    train_filenames, train_labels = dataloder(img_root, txt_path)
    train_filenames = tf.constant(train_filenames)
    train_labels = tf.constant(train_labels)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(
        map_func=decode_and_resize,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=2000)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return train_dataset


def lr_scheduler(epoch):
    start = 0.0001
    end = 0.00001
    return start-epoch*(start-end)/epochs


# 必须定义为函数
def CosLoss(y_true, y_pred):
    # tf.print('y_true: ', y_true)
    y_pred2 = y_pred * 2. * math.pi
    # tf.print('y_pred: ', y_pred2)
    loss = tf.reduce_mean(4. * (1. - tf.cos(0.5 * (y_pred2 - y_true))))
    # tf.print('my loss: ', loss)
    return loss


if __name__ == '__main__':
    img_root = '/home/ubuntu/cs/table_derection_tf2/use_cos_loss/train_data/rot_images'
    txt_path = '/home/ubuntu/cs/table_derection_tf2/use_cos_loss/train_data/train_angles.txt'
    epochs = 100
    batch_size = 8

    train_dataset = make_dataset(img_root, txt_path, batch_size)

    callbacks = [
        # tf.keras.callbacks.ModelCheckpoint(filepath='./saved_model_tf2/model_batch{batch}.h5', save_freq=5000),
        tf.keras.callbacks.LearningRateScheduler(lr_scheduler),
        tf.keras.callbacks.TensorBoard(log_dir='./tensorboard', update_freq='batch')]

    my_model = tf.keras.Sequential([hub.KerasLayer("https://hub.tensorflow.google.cn/google/imagenet/resnet_v2_50/feature_vector/4", trainable=True, arguments=dict(batch_norm_momentum=0.997)),
                                 tf.keras.layers.Dense(units=1024, activation=tf.nn.relu),
                                 tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid, name='output_angle')])

    my_model.build([None, 600, 600, 3])
    my_model.summary()

    my_model.compile(optimizer=tf.keras.optimizers.Adam(),
                     loss=CosLoss)

    my_model.fit(train_dataset, epochs=epochs, callbacks=callbacks)

    tf.saved_model.save(my_model, "./saved_model_tf2/pb")

