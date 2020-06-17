import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import random
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
        if angel >= 250.0 and angel <= 360:
            angel = angel - 360     # !!!
        img_path = os.path.join(img_root, img_name)
        img_paths.append(img_path)
        labels.append(angel)
    return img_paths, labels


def decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename) # 读取原始文件
    image_emhance = tf.image.decode_jpeg(image_string) # 解码JPEG图片

    image_emhance = tf.image.random_brightness(image_emhance, 0.3)
    image_emhance = tf.image.random_contrast(image_emhance, 0.7, 1.3)
    image_emhance = tf.image.central_crop(image_emhance, 0.6)

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
    # 取出前buffer_size个数据放入buffer，并从其中随机采样，采样后的数据用后续数据替换
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return train_dataset


def lr_scheduler(epoch):
    '''
        imply learning rate linear decay scheduler(from start to end).
    '''
    start = 0.001
    end = 0.0001
    return start-epoch*(start-end)/epochs


class my_model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.hub_model = hub.KerasLayer("https://hub.tensorflow.google.cn/google/imagenet/resnet_v2_50/feature_vector/4",
                                        trainable=True,
                                        arguments=dict(batch_norm_momentum=0.997))
        self.dense = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=1)

    #@tf.function
    def call(self, inputs):
        features = self.hub_model(inputs)
        hidden = self.dense(features)
        out = self.out(hidden)
        return out


if __name__ == '__main__':
    img_root = '/home/ubuntu/cs/table_derection_tf2/use_cos_loss/train_data/rot_images'
    txt_path = '/home/ubuntu/cs/table_derection_tf2/use_cos_loss/train_data/train_angles.txt'
    epochs = 50
    #batch_size_per_replica = 8
    batch_size = 8

    '''
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: %d' % strategy.num_replicas_in_sync + '\n' * 5)  # 输出设备数量
    batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
    print('batch_size={0}'.format(batch_size) + '\n' * 5)
    '''

    '''
    # use mixed precision policy.
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    '''

    train_dataset = make_dataset(img_root, txt_path, batch_size)

    #with strategy.scope():
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath='./saved_model3/mymodel_batch{batch}.h5', save_freq=50000),
        tf.keras.callbacks.LearningRateScheduler(lr_scheduler),
        tf.keras.callbacks.TensorBoard(log_dir='/home/ubuntu/cs/table_derection_tf2/tensorboard',
                                       update_freq='batch')
    ]
    model = my_model()

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.MSE)

    model.fit(train_dataset, epochs=epochs, callbacks=callbacks)


    # save the whole model to .pb
    #tf.keras.models.save_model(model, filepath='./saved_model/pb2', save_format='tf')
    tf.saved_model.save(model, "./saved_model3/pb")
    #model.save("saved_model/pb1")

