import os
import cv2
import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

batch_size = 8


def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH))


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

    image_emhance = tf.image.central_crop(image_decoded, 0.7)

    image_resized = tf.image.resize(image_emhance, [600, 600]) / 255.0

    label = tf.expand_dims(label, axis=-1)
    label = tf.cast(label, dtype=tf.float32)
    return image_resized, label, filename


def test_show(filepath, angle):
    filepath = filepath.decode('utf-8')
    image = cv2.imread(filepath)
    # h, w = image.shape[:2]
    # c_w, c_h = w//2, h//2
    # n_w, n_h = int((w * 0.6)/2), int((h * 0.6)/2)
    # crop_img = image[c_h-n_h:c_h+n_h, c_w-n_w:c_w+n_w]
    # resized = cv2.resize(crop_img, (600, 600))
    rotated = rotate_bound(image, -angle)
    cv2.imwrite('./{}'.format(filepath[-10:]), rotated)


if __name__ == '__main__':
    img_root = '/root/ais/shanghai_bank_poc/test_rotated/'
    txt_path = '/root/ais/shanghai_bank_poc/test_angles.txt'

    # 数据加载
    test_filenames, test_labels = dataloder(img_root, txt_path)

    test_filenames = tf.constant(test_filenames)
    test_labels = tf.constant(test_labels)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
    test_dataset = test_dataset.map(
        map_func=decode_and_resize,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # 取出前buffer_size个数据放入buffer，并从其中随机采样，采样后的数据用后续数据替换
    test_dataset = test_dataset.shuffle(buffer_size=400)
    test_dataset = test_dataset.batch(batch_size)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    model = tf.keras.Sequential([
        hub.KerasLayer("https://hub.tensorflow.google.cn/google/imagenet/resnet_v2_50/feature_vector/4",
                       trainable=True, arguments=dict(batch_norm_momentum=0.997)),
        tf.keras.layers.Dense(units=1024, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=1)
    ])
    model.build([None, 600, 600, 3])  # Batch input shape.

    checkpoint = tf.train.Checkpoint(myModel=model)
    checkpoint.restore(tf.train.latest_checkpoint('./checkpoints'))  # 从文件恢复模型参数

    cnt = 0
    correct = 0
    start = time.time()
    for step, (images, labels, filename) in enumerate(test_dataset):
        pre_angles = model.predict(images)

        label = labels.numpy()
        path = filename.numpy()
        for i, pre_angle in enumerate(pre_angles):
            cnt += 1
            if abs(pre_angle[0] - label[i][0]) <= 20:
                correct += 1
            else:
                print('pre_angle: ', pre_angle[0])
                print('label: ', label[i][0])
                print('filename: ', path[i])
                test_show(path[i], pre_angle[0])
    print('#'*30)
    print('correct: ', correct)
    print('acc: ', correct/cnt)
    end = time.time()
    print('time cost', (end - start)/200.0, 's')
