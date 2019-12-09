# 12/09/2019: M.F. train1206, 加入多GPU训练代码
# 12/06/2019: 尝试冻结层以增大batch size；将datagen生成batch的过程改为标准的N×K格式
# 08/15/2019: 训练时不做crop只做resize和rotate
# 08/11/2019: 函数式data_generator
# 08/13/2019: t3的训练数据减均值x_mean时，uint8类型问题，改为在输出的时候减均值；另，加一个model.layers.pop()；
#             另，类别数增加到50；另，增加GPU编号选择；

from os import listdir
import imghdr
import os, random
import numpy as np
import tensorflow as tf
from keras.layers import Dense,Dropout,Flatten,Conv2D, MaxPooling2D, Reshape, Activation
from keras.models import Model
from keras.utils import np_utils
from keras import optimizers, metrics

from efficientnet import EfficientNetB5
from PIL import Image
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils.training_utils import multi_gpu_model
import cv2
from imgaug import augmenters as iaa
import imgaug as ia

from triplet_losses import batch_all_triplet_loss, batch_hard_triplet_loss
from triplet_metrics import triplet_accuracy, mean_norm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_num = 1
emb_size = 512
class_num = 45  # SKU种类
N = 12  # 每个batch里SKU的种类
K = 4  # 每种SKU的图片数，不能太多，一般固定取4
batch_size = N * K
x_mean = np.full((N*K, 456, 456, 3), (50.46, 52.25, 43.12), np.float32)


train_path = 'D:/Dataset/lilscale50/Train/'  # 最后的‘/’不能少
val_path = 'D:/Dataset/lilscale50/Val/'
test_path = 'D:/Dataset/lilscale50/Test/'
# 12/06/2019: 选取有限的SKU组成batch，即按照N×k的格式选取batch

def datagen(dataset_path, N, K, x_mean):  # 函数式generator
    while 1:

        # 先从数据集中选出N类SKU
        classes = random.sample(range(class_num), N)
        labels = np.repeat(np.array(classes), K)  # 生成对应的标签
        classes = ['0'+str(ii) if ii < 10 else str(ii) for ii in classes]  # 转换为字符
        imgs = np.zeros((N * K, 957, 957, 3), dtype=np.uint8)  # 训练图片
        img_idx = 0
        for sku_idx in classes:
            sku_path = dataset_path + sku_idx

            img_names = random.sample(os.listdir(sku_path), K)  # 每类SKU取K张图片
            print(img_names)
            print('OK')
            for img_name in img_names:
                imgs[img_idx] = Image.open(sku_path + '/' + img_name)
                img_idx += 1
        # 定义并应用数据增强
        ia.seed(np.random.randint(0, 100))
        data_preprocessor1 = iaa.Sequential([iaa.Affine(rotate=(-179, 180))])
        data_preprocessor2 = iaa.Sequential([iaa.Resize({"height": 456, "width": 456}),
                                             iaa.Sometimes(0.25, iaa.ContrastNormalization((0.8, 1.2))),
                                             # 25%的图像对比度随机变为0.8或1.2倍
                                             iaa.Sometimes(0.25, iaa.Multiply((0.8, 1.2)))
                                             # 25%的图片像素值乘以0.8-1.2中间的数值,用以增加图片明亮度或改变颜色
                                             ])
        imgs = data_preprocessor1.augment_images(imgs)
        imgs = imgs[:, 140:-140, 140:-140, :]
        imgs = data_preprocessor2.augment_images(imgs)
        imgs = imgs.astype(np.float32)-x_mean  # 转为ndarray
        # shuffle
        idxes = np.random.permutation(N*K)  # shuffle图的顺序
        print('type(imgs): ', type(imgs))
        yield imgs[idxes], labels[idxes]

# 测试datagen函数
# imgs, labels = datagen(train_path, N, K, x_mean)
# from matplotlib import pyplot as plt
# for label, img in zip(labels, imgs):
#     print('label: ', label)
#     plt.imshow(img)
#     plt.axis('off')
#     plt.show()

        # 从整个数据集的所有类别中，随机选取batch_size张图片，预处理后yield
        # for img_idx in range(batch_size):
        #     label = np.random.randint(0, 50)  # 随机选取一个类别编号，并作为文件夹索引的一部分
        #     labels[img_idx] = label
        #     if label < 10:
        #         img_path = dataset_path + '0' + str(label)
        #     else:
        #         img_path = dataset_path + str(label)
        #     img_path = img_path + '/' + random.sample(os.listdir(img_path), 1)[0]  # random.sample从指定目录下随机选择文件
        #     img = Image.open(img_path)
        #     imgs[img_idx] = img
        # ia.seed(np.random.randint(0, 100))
        # data_preprocessor1 = iaa.Sequential([iaa.Affine(rotate=(-179, 180))])
        # data_preprocessor2 = iaa.Sequential([iaa.Resize({"height": 456, "width": 456}),
        #                                      iaa.Sometimes(0.25, iaa.ContrastNormalization((0.8, 1.2))),  # 25%的图像对比度随机变为0.8或1.2倍
        #                                      iaa.Sometimes(0.25, iaa.Multiply((0.8, 1.2)))  # 25%的图片像素值乘以0.8-1.2中间的数值,用以增加图片明亮度或改变颜色
        #                                      ])
        # imgs_aug0 = data_preprocessor1.augment_images(imgs)
        # imgs_aug1 = imgs_aug0[:, 140:-140, 140:-140, :]
        # imgs_aug2 = data_preprocessor2.augment_images(imgs_aug1)

        # yield imgs_aug2.astype(np.float32)-x_mean, labels

#  use CPU to instantiate the base model
with tf.device("/cpu:0"):
    model = EfficientNetB5(weights='imagenet', include_top=True)  # 612层
    print('layer count: ', len(model.layers))
    print(model.summary())
    model.layers.pop()
    model.layers.pop()
    x = model.layers[-1].output
    out = Dense(emb_size, activation='linear')(x)
    model = Model(inputs=model.input, output=out)
    # plot_model(model, to_file='model.png',show_shapes=True)
    for i in range(0, len(model.layers) - 2):
        model.layers[i].trainable = False

# print('layer count: ', len(model.layers))
# print(model.summary())
#----------------Multi-GPU model-------------------------#
model = multi_gpu_model(model, gpus=gpu_num)

# sgd = optimizers.SGD(lr=0.0001, decay=0, momentum=0.9)
model.compile(loss=batch_all_triplet_loss, optimizer='adam', metrics=[triplet_accuracy, mean_norm])

filepath = "weights-50classes-alien-{epoch:02d}.h5"

callbacks_list = []
# 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
checkpoint = ModelCheckpoint(filepath)
callbacks_list.append(checkpoint)

tensorboard_callback = TensorBoard(
            log_dir                = './logs',
            histogram_freq         = 0,
            batch_size             = N*K,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
callbacks_list.append(tensorboard_callback)

train_history=model.fit_generator(datagen(train_path, N*K, x_mean), steps_per_epoch=1000,
                                  epochs=100, validation_data=datagen(val_path, N*K, x_mean),
                                  validation_steps=2, verbose=1, callbacks=callbacks_list)

# print('train_history: ', train_history.history)
# loss,accuracy,test = model.evaluate(x_test, y_TestOneHot, batch_size=2, verbose=1)
# print(loss,accuracy,test)
