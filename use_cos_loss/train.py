import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import random
import math


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

    image_emhance = tf.image.random_brightness(image_emhance, 0.3)
    image_emhance = tf.image.random_contrast(image_emhance, 0.7, 1.3)
    image_emhance = tf.image.per_image_standardization(image_emhance)   # 不能直接 / 255！！！

    label = tf.expand_dims(label, axis=-1)
    label = tf.cast(label, dtype=tf.float32)
    return image_emhance, label


def make_dataset(img_root, txt_path, batch_size, epochs):
    train_filenames, train_labels = dataloder(img_root, txt_path)
    train_filenames = tf.constant(train_filenames)
    train_labels = tf.constant(train_labels)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(map_func=decode_and_resize)

    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.repeat(epochs)

    return train_dataset


def lr_scheduler(epoch):
    '''
        imply learning rate linear decay scheduler(from start to end).
    '''
    start = 0.001
    end = 0.0001
    return start-epoch*(start-end)/epochs


'''
class my_model(tf.keras.Model):
    def __init__(self, resnet):
        x = resnet.output
        # x = tf.keras.layers.Flatten()(x) 由于有 GlobalAveragePooling2D()
        # x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)(x)
        # x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid, name='output_angle')(x)

        super(my_model, self).__init__(resnet.input, x, name="my_model")

    def call(self, inputs):
        return self.output(inputs) * 2.0 * math.pi
'''


# 必须定义为函数
def cosloss(y_true, y_pred):
    y_true = tf.Print(y_true, ['y_true: ', y_true])
    y_pred2 = y_pred * 2. * math.pi
    y_pred2 = tf.Print(y_pred2, ['y_pred: ', y_pred2])
    loss = K.mean(4. * (1. - tf.cos(0.5 * (y_pred2 - y_true))))
    loss = tf.Print(loss, ['my loss: ', loss])
    return loss


if __name__ == '__main__':
    img_root = '/home/ubuntu/cs/table_derection_tf2/use_cos_loss/train_data/rot_images'
    txt_path = '/home/ubuntu/cs/table_derection_tf2/use_cos_loss/train_data/train_angles.txt'
    epochs = 200
    batch_size = 8
    save_period = 3
    steps = 3460
    learn_rate = 1e-5
    input_shape = (600, 600, 3)    # 必须指定输入维度

    train_dataset = make_dataset(img_root, txt_path, batch_size, epochs)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath='./saved_model/model_{epoch:02d}.hdf5', period=save_period),
        # tf.keras.callbacks.LearningRateScheduler(lr_scheduler),
        tf.keras.callbacks.TensorBoard(log_dir='./tensorboard')
    ]

    resnet = tf.keras.applications.ResNet50(weights=None, include_top=False, input_shape=input_shape, pooling='avg')
    # model = my_model(resnet)
    x = resnet.output
    # x = tf.keras.layers.Flatten()(x) 由于有 GlobalAveragePooling2D()
    # x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid, name='output_angle')(x)
    model = tf.keras.Model(inputs=resnet.input, outputs=x)

    # ops = tf.get_default_graph().get_operations()
    # update_ops = [x for x in ops if ("AssignMovingAvg" in x.name and x.type == "AssignSubVariableOp")]
    # tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_ops)
    # print(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    # bn_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(tf.GraphKeys.UPDATE_OPS):
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(learn_rate),
                  loss=cosloss,
                  metrics=["accuracy"])
    model.fit(train_dataset, epochs=epochs, callbacks=callbacks, steps_per_epoch=steps)


