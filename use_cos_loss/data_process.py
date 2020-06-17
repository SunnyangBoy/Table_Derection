import cv2
import numpy as np
import os
import math

def avg_count(img):
    sum_r = 0
    sum_g = 0
    sum_b = 0
    for i in range(10):
        for j in range(10):
            sum_r += img[i][j][0]
            sum_g += img[i][j][1]
            sum_b += img[i][j][2]
    return sum_r/100, sum_g/100, sum_b/100


def avg(img1, img2, img3, img4):
    r1, g1, b1 = avg_count(img1)
    r2, g2, b2 = avg_count(img2)
    r3, g3, b3 = avg_count(img3)
    r4, g4, b4 = avg_count(img4)
    return (r1+r2+r3+r4)//4, (g1+g2+g3+g4)//4, (b1+b2+b3+b4)//4


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    avg_r, avg_g, avg_b = avg(image[0:10, 0:10], image[0:10, w-10:w], image[h-10:h, 0:10], image[h-10:h, w-10:w])

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(avg_r, avg_g, avg_b))


def crop_image(image, resize):
    ratio = np.random.uniform(0.6, 1.0)
    print('ratio ', ratio)
    h, w = image.shape[:2]
    print(h, w)
    c_w, c_h = w // 2, h // 2
    n_w, n_h = int((w * ratio) / 2), int((h * ratio) / 2)
    if (w * ratio) < resize or (h * ratio) < resize:
        return image
    crop_img = image[c_h - n_h:c_h + n_h, c_w - n_w:c_w + n_w]
    return crop_img


if __name__ == '__main__':
    pre_num = 6

    create_txt = '/home/ubuntu/cs/table_derection_tf2/use_cos_loss/train_data/train_angles.txt'
    create_images = '/home/ubuntu/cs/table_derection_tf2/use_cos_loss/train_data/rot_images'

    root_dir = '/home/ubuntu/cs/table_derection_tf2/use_cos_loss/train_data/images'

    file_names = sorted(os.listdir(root_dir))

    rand_angle = np.random.uniform(0, 359, (len(file_names)) * pre_num)
    # print("rand_angels ", rand_angle)
    rand_angle = rand_angle * math.pi / 180.0
    # print("rand_angels ", rand_angle)
    print('len: ', len(file_names))

    cnt = 0
    resize = 600
    with open(create_txt, 'w') as writer:
        for i, file_name in enumerate(file_names):
            print(file_name)
            img_path = os.path.join(root_dir, file_name)
            image = cv2.imread(img_path)

            new_name = file_name[:-4] + '_10' + '.jpg'
            new_imgPath = os.path.join(create_images, new_name)
            new_image = cv2.resize(image, (resize, resize))
            cv2.imwrite(new_imgPath, new_image)
            line = new_name + ';' + '0.0000' + '\n'
            writer.write(line)

            new_image = rotate_bound(image, 90)
            new_image = cv2.resize(new_image, (resize, resize))
            new_name = file_name[:-4] + '_11' + '.jpg'
            new_imgPath = os.path.join(create_images, new_name)
            cv2.imwrite(new_imgPath, new_image)
            line = new_name + ';' + '1.5708' + '\n'
            writer.write(line)

            new_image = rotate_bound(image, 180)
            new_image = cv2.resize(new_image, (resize, resize))
            new_name = file_name[:-4] + '_12' + '.jpg'
            new_imgPath = os.path.join(create_images, new_name)
            cv2.imwrite(new_imgPath, new_image)
            line = new_name + ';' + '3.1416' + '\n'
            writer.write(line)

            new_image = rotate_bound(image, 270)
            new_image = cv2.resize(new_image, (resize, resize))
            new_name = file_name[:-4] + '_13' + '.jpg'
            new_imgPath = os.path.join(create_images, new_name)
            cv2.imwrite(new_imgPath, new_image)
            line = new_name + ';' + '4.7124' + '\n'
            writer.write(line)

            for j in range(pre_num):
                angle = rand_angle[cnt]
                cnt += 1
                new_image = rotate_bound(image, angle / math.pi * 180.0)
                new_image = crop_image(new_image, resize)
                new_image = cv2.resize(new_image, (resize, resize))
                new_name = file_name[:-4] + '_' + str(j).zfill(2) + '.jpg'
                new_imgPath = os.path.join(create_images, new_name)
                cv2.imwrite(new_imgPath, new_image)
                line = new_name + ';' + str(angle) + '\n'
                writer.write(line)
        writer.close()
