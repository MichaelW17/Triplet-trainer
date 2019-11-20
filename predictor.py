# 10/xx/2019: 使用训练得到的三重损失模型，计算出测试集的嵌入向量，并画出TSNE图
# from predict3，测试50类模型weights-50classes-retrain-00.h5
# predict1在t3训练模型上的精度很低, predict2在现场生成的测试集上精度99%
# 原因找到了，在datagen中-x_mean的时候，如果是在读取img之后直接减，img是uint8, x_mean也是，二者相减很可能导致负值，这是错误的
# 所以要么在将img转为float32之后再减
from keras.models import load_model

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
from keras.callbacks import ModelCheckpoint

from imgaug import augmenters as iaa
import imgaug as ia
import matplotlib.pyplot as plt

from triplet_losses import batch_all_triplet_loss, batch_hard_triplet_loss
from triplet_metrics import triplet_accuracy, mean_norm
import cv2
from sklearn.manifold import TSNE
from TSNE_plot import scatter


emb_dim = 256
avg_num = 5
x_mean = np.full((avg_num, 456, 456, 3), (50.46, 52.25, 43.12), dtype=np.float32)
# print(x_mean.dtype)

model = load_model('C:/Users/Minghao/Desktop/dipper/weights-45classes-triplet-47.h5',
                   custom_objects={'batch_all_triplet_loss': batch_all_triplet_loss,
                                   'triplet_accuracy': triplet_accuracy, 'mean_norm': mean_norm})

dataset_path = 'D:/Dataset/lilscale50/Test/'  # 最后的‘/’不能少
# print(os.listdir(dataset_path).sort)
def create_high_dim_embs(dataset_path, classes, x_mean, avg_num):
    embs = np.ones((len(classes)*avg_num, emb_dim))
    img_dirs = os.listdir(dataset_path)
    img_dirs.sort()
    for i in classes:
        img_dir = dataset_path + img_dirs[i]
        emb_pool = np.zeros((emb_dim, ))
        imgs = np.zeros((avg_num, 456, 456, 3))
        for j in range(avg_num):  # 取随机5张图片的平均嵌入值
            img_path = img_dir + '/' + random.sample(os.listdir(img_dir), 1)[0]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[140:-140, 140:-140, :]
            img = cv2.resize(img, (456, 456), interpolation=cv2.INTER_CUBIC)
            imgs[j] = img
            # img = np.expand_dims(img, 0)
        predictions = model.predict_on_batch(imgs - x_mean)
        print(predictions.shape)
        # emb_pool += prediction[0]

        embs[i*avg_num:(i+1)*avg_num] = predictions
        # embs[i] = emb_pool / avg_num

    return embs

            # plt.imshow(img[0])
            # plt.axis('off')
            # plt.show()

classes = np.arange(0, 45)
embs = create_high_dim_embs(dataset_path, classes, x_mean, avg_num=avg_num)
print(embs)
print(np.max(embs, axis=1))
print(np.argmax(embs, axis=1))
from TSNE_plot import tsne_plot

# tsne = TSNE(init='pca', random_state=0)
tsne = TSNE()
tsne_embeds = tsne.fit_transform(embs)
scatter(tsne_embeds, np.repeat(classes, avg_num))

