# 12/12/2019: 测试科宇相机的预处理

from os import listdir
import imghdr
import os, random
import numpy as np
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D,Reshape,Activation
from keras.models import Model
from keras.utils import np_utils

from keras import optimizers, metrics

from efficientnet import EfficientNetB5
from PIL import Image
from keras.callbacks import ModelCheckpoint, TensorBoard
from matplotlib import pyplot as plt


from imgaug import augmenters as iaa
import imgaug as ia

batch_size = 1
x_mean = np.full((batch_size, 456, 456, 3), (71.35, 69.19, 59.52), np.float32)


dataset_path = 'D:/Dataset/new_scale/data/'  # 最后的‘/’不能少
def datagen(dataset_path, batch_size, x_mean):  # 函数式generator
    while 1:
        imgs = np.zeros((batch_size, 957, 957, 3), dtype=np.uint8)
        labels = np.zeros((batch_size,))
        # 从整个数据集的所有类别中，随机选取batch_size张图片，预处理后yield
        for img_idx in range(batch_size):
            label = np.random.randint(0, 20)  # 随机选取一个类别编号，并作为文件夹索引的一部分
            labels[img_idx] = label
            if label < 10:
                img_path = dataset_path + '0' + str(label)
            else:
                img_path = dataset_path + str(label)
            img_path = img_path + '/' + random.sample(os.listdir(img_path), 1)[0]  # random.sample从指定目录下随机选择文件
            img = Image.open(img_path)
            imgs[img_idx] = img
        ia.seed(np.random.randint(0, 100))
        data_preprocessor1 = iaa.Sequential([iaa.Affine(rotate=(-179, 180))])
        data_preprocessor2 = iaa.Sequential([iaa.Resize({"height": 456, "width": 456}, interpolation='area'),
                                             iaa.Sometimes(0.25, iaa.ContrastNormalization((0.8, 1.2))),  # 25%的图像对比度随机变为0.8或1.2倍
                                             iaa.Sometimes(0.25, iaa.Multiply((0.8, 1.2)))  # 25%的图片像素值乘以0.8-1.2中间的数值,用以增加图片明亮度或改变颜色
                                             ])
        imgs_aug0 = data_preprocessor1.augment_images(imgs)
        imgs_aug1 = imgs_aug0[:, 140:-140, 140:-140, :]
        imgs_aug2 = data_preprocessor2.augment_images(imgs_aug0)
        plt.imshow(np.squeeze(imgs_aug2, 0))
        plt.show()
        # return imgs_aug2.astype(np.float32)-x_mean, np_utils.to_categorical(labels, num_classes=50)
        return imgs_aug2.astype(np.float32), np_utils.to_categorical(labels, num_classes=50)


imgs, labels = datagen(dataset_path, batch_size, x_mean)
# for img in imgs:
# #     cv