# 12/09/2019: M.F. train1206, ¼ÓÈë¶àGPUÑµÁ·´úÂë
# 12/06/2019: ³¢ÊÔ¶³½á²ãÒÔÔö´óbatch size£»½«datagenÉú³ÉbatchµÄ¹ý³Ì¸ÄÎª±ê×¼µÄN¡ÁK¸ñÊ½
# 08/15/2019: ÑµÁ·Ê±²»×öcropÖ»×öresizeºÍrotate
# 08/11/2019: º¯ÊýÊ½data_generator
# 08/13/2019: t3µÄÑµÁ·Êý¾Ý¼õ¾ùÖµx_meanÊ±£¬uint8ÀàÐÍÎÊÌâ£¬¸ÄÎªÔÚÊä³öµÄÊ±ºò¼õ¾ùÖµ£»Áí£¬¼ÓÒ»¸ömodel.layers.pop()£»
#             Áí£¬Àà±ðÊýÔö¼Óµ½50£»Áí£¬Ôö¼ÓGPU±àºÅÑ¡Ôñ£»

from os import listdir
import imghdr
import os, random
import numpy as np
import tensorflow as tf
from keras.layers import Dense,Dropout,Flatten,Conv2D, MaxPooling2D, Reshape, Activation
from keras.models import Model, load_model
from keras.utils import np_utils
from keras import optimizers, metrics

from efficientnet import EfficientNetB5
from PIL import Image
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import multi_gpu_model
import cv2
from imgaug import augmenters as iaa
import imgaug as ia
import time
from triplet_losses import batch_all_triplet_loss, batch_hard_triplet_loss
from triplet_metrics import triplet_accuracy, mean_norm

os.environ["CUDA_VISIBLE_DEVICES"] = "6, 7"
gpu_num = 2
emb_size = 256
class_num = 20  # SKUÖÖÀà
N = 8  # Ã¿¸öbatchÀïSKUµÄÖÖÀà
K = 8  # Ã¿ÖÖSKUµÄÍ¼Æ¬Êý£¬²»ÄÜÌ«¶à£¬Ò»°ã¹Ì¶¨È¡4
batch_size = N * K
x_mean = np.full((N*K, 456, 456, 3), (67.35, 65.08, 56.17), np.float32)


train_path = '/media/team/RaidDisk/WMH/Scale/new_scale/Train/'  # 最后的‘/’不能少
val_path = '/media/team/RaidDisk/WMH/Scale/new_scale/Val/'
test_path = '/media/team/RaidDisk/WMH/Scale/new_scale/Test/'


def datagen(dataset_path, N, K, x_mean):  # º¯ÊýÊ½generator
    t0 = time.time()
    while 1:

        # ÏÈ´ÓÊý¾Ý¼¯ÖÐÑ¡³öNÀàSKU
        # classes = random.sample(range(class_num), N)  # might have repetitive items
        classes = np.random.choice(class_num, N, replace=False)
        # print('len(classes): ', len(classes))
        labels = np.repeat(classes, K)  # Éú³É¶ÔÓ¦µÄ±êÇ©
        classes = ['0'+str(ii) if ii < 10 else str(ii) for ii in classes]  # ×ª»»Îª×Ö·û
        imgs = np.zeros((N * K, 957, 957, 3), dtype=np.uint8)  # ÑµÁ·Í¼Æ¬
        img_idx = 0
        for sku_idx in classes:
            sku_path = dataset_path + sku_idx

            img_names = random.sample(os.listdir(sku_path), K)  # Ã¿ÀàSKUÈ¡KÕÅÍ¼Æ¬
            # print(sku_idx, img_names)
            for img_name in img_names:
                imgs[img_idx] = Image.open(sku_path + '/' + img_name)
                img_idx += 1
        # ¶¨Òå²¢Ó¦ÓÃÊý¾ÝÔöÇ¿
        ia.seed(np.random.randint(0, 100))
        data_preprocessor1 = iaa.Sequential([iaa.Affine(rotate=(-179, 180))])
        data_preprocessor2 = iaa.Sequential([iaa.Resize({"height": 456, "width": 456}),
                                             iaa.Sometimes(0.25, iaa.ContrastNormalization((0.8, 1.2))),  # contrast
                                             iaa.Sometimes(0.25, iaa.Multiply((0.8, 1.2)))  # brightness
                                             ])
        imgs = data_preprocessor1.augment_images(imgs)
        # imgs = imgs[:, 140:-140, 140:-140, :]
        imgs = data_preprocessor2.augment_images(imgs)
        imgs = imgs.astype(np.float32)-x_mean  # ×ªÎªndarray
        # shuffle
        idxes = np.random.permutation(N*K)  # shuffleÍ¼µÄË³Ðò
        # print('type(imgs): ', type(imgs))
        print('  time for this batch: ', time.time() - t0)
        t0 = time.time()
        yield imgs[idxes], labels[idxes]


#  use CPU to instantiate the base model
with tf.device("/cpu:0"):
    # model = EfficientNetB5(weights='imagenet', include_top=True)  # 612²ã
    model = load_model('new20classes-aug-24.h5')
    model.layers.pop()
    model.layers.pop()
    x = model.layers[-1].output
    out = Dense(emb_size, activation='linear')(x)
    model = Model(inputs=model.input, output=out)
    # plot_model(model, to_file='model.png',show_shapes=True)
    for i in range(0, len(model.layers) - 1):
        model.layers[i].trainable = False

# print('layer count: ', len(model.layers))
# print(model.summary())
#----------------Multi-GPU model-------------------------#
model = multi_gpu_model(model, gpus=gpu_num)

# sgd = optimizers.SGD(lr=0.0001, decay=0, momentum=0.9)
model.compile(loss=batch_all_triplet_loss, optimizer='adam', metrics=[triplet_accuracy, mean_norm])


filepath = "new20classes-triplet-aug-{epoch:02d}.h5"

callbacks_list = []
# ÖÐÍ¾ÑµÁ·Ð§¹ûÌáÉý, Ôò½«ÎÄ¼þ±£´æ, Ã¿ÌáÉýÒ»´Î, ±£´æÒ»´Î
checkpoint = ModelCheckpoint(filepath)
callbacks_list.append(checkpoint)

tensorboard_callback = TensorBoard(
            log_dir                = './logs1213',
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

train_history=model.fit_generator(datagen(train_path, N, K, x_mean), steps_per_epoch=100*class_num*36/64,
                                  epochs=100, validation_data=datagen(val_path, N, K, x_mean),
                                  validation_steps=10*class_num*36/64, verbose=1, callbacks=callbacks_list)

# print('train_history: ', train_history.history)
# loss,accuracy,test = model.evaluate(x_test, y_TestOneHot, batch_size=2, verbose=1)
# print(loss,accuracy,test)
