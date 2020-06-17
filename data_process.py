import cv2
import numpy as np
import os
import tensorflow as tf

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


if __name__ == '__main__':
    pre_num = 10

    create_txt = '/home/ubuntu/shanghai_bank_poc/train_angles.txt'
    create_images = '/home/ubuntu/shanghai_bank_poc/train_rotated'

    root_dir = '/home/ubuntu/shanghai_bank_poc/train_images/'

    file_names = sorted(os.listdir(root_dir))

    rand_angle = np.random.uniform(0, 360, (len(file_names)) * pre_num)
    print('len: ', len(file_names))
    print('angels: ', rand_angle)

    cnt = 0
    with open(create_txt, 'w') as writer:
        for i, file_name in enumerate(file_names):
            print(file_name)
            img_path = os.path.join(root_dir, file_name)
            image = cv2.imread(img_path)

            new_image = cv2.pyrDown(image)  # decrease image size
            new_name = file_name[:-4] + '_10' + '.jpg'
            new_imgPath = os.path.join(create_images, new_name)
            cv2.imwrite(new_imgPath, new_image)
            line = new_name + ';' + '0.0' + '\n'
            writer.write(line)

            new_image = rotate_bound(image, 180)
            new_image = cv2.pyrDown(new_image)  # decrease image size
            new_name = file_name[:-4] + '_11' + '.jpg'
            new_imgPath = os.path.join(create_images, new_name)
            cv2.imwrite(new_imgPath, new_image)
            line = new_name + ';' + '180.0' + '\n'
            writer.write(line)

            for j in range(pre_num):
                angle = rand_angle[cnt]    # float
                cnt += 1
                new_image = rotate_bound(image, angle)
                new_image = cv2.pyrDown(new_image)  # decrease image size
                new_name = file_name[:-4] + '_' + str(j).zfill(2) + '.jpg'
                new_imgPath = os.path.join(create_images, new_name)
                cv2.imwrite(new_imgPath, new_image)
                line = new_name + ';' + str(angle) + '\n'
                writer.write(line)
        writer.close()
