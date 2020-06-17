import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import random
import cv2
import time
from tensorflow.keras.mixed_precision import experimental as mixed_precision

batch_size = 8

def dataloder(img_root, txt_path):
    img_paths = []
    labels = []
    with open(txt_path, 'r') as reader:
        lines = reader.readlines()
        for line in lines:
            line = line.split(';')
            img_name = line[0]
            angel = float(line[1][:-1])
            #if (angel <= 45 and angel >= 0) or (angel <= 360 and angel >= 315) or (angel <= 225 and angel >= 135):
            if angel >=300.0 and angel <= 302.0:
                img_path = os.path.join(img_root, img_name)
                img_paths.append(img_path)
                labels.append(angel)
    return img_paths, labels


def decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename)  # 读取原始文件
    image_decoded = tf.image.decode_jpeg(image_string)  # 解码JPEG图片

    image_emhance = tf.image.central_crop(image_decoded, 0.6)

    image_resized = tf.image.resize(image_emhance, [600, 600]) / 255.0

    label = tf.expand_dims(label, axis=-1)
    label = tf.cast(label, dtype=tf.float32)
    return image_resized, label, filename


def test_show(filepath, angle, label):
    filepath = filepath.decode('utf-8')
    image = cv2.imread(filepath)
    h, w = image.shape[:2]
    c_w, c_h = w//2, h//2
    n_w, n_h = int((w * 0.6)/2), int((h * 0.6)/2)
    crop_img = image[c_h-n_h:c_h+n_h, c_w-n_w:c_w+n_w]
    resized = cv2.resize(crop_img, (600, 600))
    #rotated = rotate_bound(resized, -angle)
    cv2.imwrite('./wrong_images/{}'.format(str(angle)+'_'+str(label)+'.jpg'), resized)#rotated)


if __name__ == '__main__':
    '''
    model = my_model()
    model.build((None, 600, 600, 3))
    model.summary()
    model.load_weights('saved_model/mymodel_epoch50.h5')
    '''

    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    '''
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
    print(gpus, cpus, '\n'*5)
    tf.config.experimental.set_visible_devices(devices=gpus, device_type='GPU')
    '''

    model = tf.saved_model.load('saved_model2/pb/')
    #model = tf.keras.models.load_model('./saved_model/keras')
    #print(list(loaded.signatures.keys()))
    #infer = loaded.signatures["serving_default"]
    #print(infer.structured_outputs)

    img_root = '/home/ubuntu/shanghai_bank_poc/test_rotated'
    txt_path = '/home/ubuntu/shanghai_bank_poc/test_angles.txt'

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

    cnt = 0
    correct = 0
    sum_time = 0
    count = 0
    for step, (images, labels, filename) in enumerate(test_dataset):
        #print(infer(images))
        count += 1
        start = time.time()
        pre_angles = model(images)
        end = time.time()
        sum_time += (end - start)

        preds = pre_angles.numpy()
        tmp_pred = np.where(preds >= -110, preds, -110)
        preds = np.where(tmp_pred <= 250, tmp_pred, 250)
        label = labels.numpy()
        path = filename.numpy()
        for i, pred in enumerate(preds):
            cnt += 1
            if abs(360.0 - pred[0] - label[i][0]) <= 20:
                correct += 1
            elif abs(pred[0] - label[i][0]) <= 20:
                correct += 1
            else:
                print('pre_angle: ', pred[0])
                print('label: ', label[i][0])
                print('filename: ', path[i])
                #test_show(path[i], pred[0], label[i][0])
    print('#' * 30)
    print('correct:{}/{}'.format(correct, cnt))
    print('acc: ', correct / cnt)
    print('pre_time: ', sum_time / count)
