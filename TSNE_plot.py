######################################################
# Original implementation by KinWaiCheuk: https://github.com/KinWaiCheuk/Triplet-net-keras
######################################################


from sklearn.manifold import TSNE
import numpy as np
import matplotlib.patheffects as PathEffects
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


def scatter(x, labels, class_num, show_id=True):
    '''

    :param x:  每一行为一个嵌入向量在tsne之后得到的二维表示
    :param labels: 嵌入向量对应的label
    :param len:  类别数
    :return: none
    '''
    # choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", class_num))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[labels.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    if show_id:
    # add a label for each cluster
        txts = []
        for i in range(class_num):
            # Position of each label.
            xtext, ytext = np.median(x[labels == i, :], axis=0)
            txt = ax.text(xtext, ytext, str(i), fontsize=24)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)

    plt.show()



def tsne_plot(name, x_train, y_train,):
    tsne = TSNE()
    train_tsne_embeds = tsne.fit_transform(x_train[:512])
    scatter(train_tsne_embeds, y_train[:512])



