'''
import tensorflow as tf

filename = '/home/ubuntu/shanghai_bank_poc/train_rotated/2-4_8.jpg'
image_string = tf.io.read_file(filename)  # 读取原始文件
image_decoded = tf.image.decode_jpeg(image_string)  # 解码JPEG图片

image_emhance = tf.image.central_crop(image_decoded, 0.8)

encoded_image = tf.image.encode_jpeg(image_emhance)

tf.io.write_file('mage_arr.jpg', encoded_image)

root_dir = '/home/ubuntu/shanghai_bank_poc/train_rotated'

import os

print('files count: ', len(os.listdir(root_dir)))
'''

import cv2

filename = '/home/ubuntu/cs/segmentation_data/scnnning_pdf_1215/img_300010.jpg'

image = cv2.imread(filename)

h, w = image.shape[:2]
c_w, c_h = w//2, h//2
n_w, n_h = int((w * 0.8)/2), int((h * 0.8)/2)
crop_img = image[c_h-n_h:c_h+n_h, c_w-n_w:c_w+n_w]

resized = cv2.resize(crop_img, (600, 600))

cv2.imwrite('test2.jpg', resized)