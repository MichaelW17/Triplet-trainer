# 08/27/2019: 测试曝光和对比度等新的增强方式并plt显示
# 08/15/2019: 训练时不做crop只做resize和rotate
# 08/11/2019: 函数式data_generator
# 08/13/2019: t3的训练数据减均值x_mean时，uint8类型问题，改为在输出的时候减均值；另，加一个model.layers.pop()；
#             另，类别数增加到50；另，增加GPU编号选择；

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
x_mean = np.full((batch_size, 456, 456, 3), (68.27, 64.96, 52.15), np.float32)


train_path = 'D:/Dataset/lilscale50/Train/'  # 最后的‘/’不能少
val_path = 'D:/Dataset/lilscale50/Val/'
test_path = 'D:/Dataset/lilscale50/Test/'
def datagen(dataset_path, batch_size, x_mean):  # 函数式generator
    while 1:
        imgs = np.zeros((batch_size, 957, 957, 3), dtype=np.uint8)
        labels = np.zeros((batch_size,))
        # 从整个数据集的所有类别中，随机选取batch_size张图片，预处理后yield
        for img_idx in range(batch_size):
            label = np.random.randint(0, 50)  # 随机选取一个类别编号，并作为文件夹索引的一部分
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
        data_preprocessor2 = iaa.Sequential([iaa.Resize({"height": 456, "width": 456}),
                                             iaa.Sometimes(0.25, iaa.ContrastNormalization((0.8, 1.2))),  # 25%的图像对比度随机变为0.8或1.2倍
                                             iaa.Sometimes(0.25, iaa.Multiply((0.8, 1.2)))  # 25%的图片像素值乘以0.8-1.2中间的数值,用以增加图片明亮度或改变颜色
                                             ])
        imgs_aug0 = data_preprocessor1.augment_images(imgs)
        imgs_aug1 = imgs_aug0[:, 140:-140, 140:-140, :]
        imgs_aug2 = data_preprocessor2.augment_images(imgs_aug1)
        plt.imshow(np.squeeze(imgs_aug2, 0))
        plt.show()
        yield imgs_aug2.astype(np.float32)-x_mean, np_utils.to_categorical(labels, num_classes=50)

# datagen(train_path, batch_size, x_mean)


model = EfficientNetB5(weights='imagenet')
model.layers.pop()
model.layers.pop()
x = model.layers[-1].output
out = Dense(50, activation=None)(x)
out = Activation('softmax')(out)
model = Model(inputs=model.input, outputs=out)
# plot_model(model, to_file='model.png',show_shapes=True)


sgd = optimizers.SGD(lr=0.0001, decay=0, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=[metrics.mae, metrics.categorical_accuracy])

filepath = "weights-50classes-alien-{epoch:02d}.h5"

callbacks_list = []
# 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
checkpoint = ModelCheckpoint(filepath)
callbacks_list.append(checkpoint)

tensorboard_callback = TensorBoard(
            log_dir                = './logs',
            histogram_freq         = 0,
            batch_size             = batch_size,
            write_graph            = True,
            write_grads            = False,
            write_images           = False,
            embeddings_freq        = 0,
            embeddings_layer_names = None,
            embeddings_metadata    = None
        )
callbacks_list.append(tensorboard_callback)

train_history=model.fit_generator(datagen(train_path, batch_size, x_mean), samples_per_epoch=1000,
                                  epochs=100, validation_data=datagen(val_path, batch_size, x_mean),
                                  validation_steps=2, verbose=1, callbacks=callbacks_list)

# print('train_history: ', train_history.history)
# loss,accuracy,test = model.evaluate(x_test, y_TestOneHot, batch_size=2, verbose=1)
# print(loss,accuracy,test)
