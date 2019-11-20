# 通过简单的数据检查tsne的使用方法是否存在问题
import numpy as np
from keras.utils import np_utils


data = np.arange(0, 10)
data = np.repeat(data, 10)
classes = data
# data = np.expand_dims(data, axis=1)
data = np_utils.to_categorical(data, num_classes=20)

data = data.astype(np.float)
noise = np.random.normal(0, 0.1, (100, 10))
# noise = np.expand_dims(noise, axis=1)
data += noise

print(data)

from sklearn.manifold import TSNE
from TSNE_plot import scatter
tsne = TSNE()
tsne_embeds = tsne.fit_transform(data)
scatter(tsne_embeds, classes)
