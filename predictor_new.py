# 11/19/2019: M.F. predictor.py，用于计算训练集中各个类的嵌入向量的平均值，作为baseline特征空间
# 10/xx/2019: 使用训练得到的三重损失模型，计算出测试集的嵌入向量，并画出TSNE图
# from predict3，测试50类模型weights-50classes-retrain-00.h5
# predict1在t3训练模型上的精度很低, predict2在现场生成的测试集上精度99%
# 原因找到了，在datagen中-x_mean的时候，如果是在读取img之后直接减，img是uint8, x_mean也是，二者相减很可能导致负值，这是错误的
# 所以要么在将img转为float32之后再减
from keras.models import load_model

from os import listdir
import os, random
import numpy as np
from efficientnet import EfficientNetB5
from collections import defaultdict
from triplet_losses import batch_all_triplet_loss, batch_hard_triplet_loss
from triplet_metrics import triplet_accuracy, mean_norm
import cv2
import json, csv


emb_dim = 256
avg_num = 20
x_mean = np.full((avg_num, 456, 456, 3), (67.35, 65.08, 56.17), dtype=np.float32)
# baseline_space = defaultdict(list)
baseline = []

model = load_model('C:/Users/Minghao/Desktop/dipper/new20classes-triplet-aug-margin50-10.h5',
                   custom_objects={'batch_all_triplet_loss': batch_all_triplet_loss,
                                   'triplet_accuracy': triplet_accuracy, 'mean_norm': mean_norm})

dataset_path = 'D:/Dataset/new_scale/Train/'  # 最后的‘/’不能少
# print(os.listdir(dataset_path).sort)


def create_high_dim_embs(dataset_path, classes, x_mean, avg_num):
    img_dirs = os.listdir(dataset_path)
    img_dirs.sort()
    for i in classes:
        img_dir = dataset_path + img_dirs[i]
        imgs = np.zeros((avg_num, 456, 456, 3))
        for j in range(avg_num):  # 取随机5张图片的平均嵌入值
            img_path = img_dir + '/' + random.sample(os.listdir(img_dir), 1)[0]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #[140:-140, 140:-140, :]
            img = cv2.resize(img, (456, 456), interpolation=cv2.INTER_AREA)
            imgs[j] = img
        predictions = model.predict_on_batch(imgs - x_mean)  # 批量推理
        class_emb = np.mean(predictions, axis=0)  # 取一类图片特征嵌入的均值作为该类的嵌入向量
        # baseline_space[str(i)] = class_emb.tolist()
        baseline.append(class_emb)


classes = np.arange(0, 27)
create_high_dim_embs(dataset_path, classes, x_mean, avg_num=avg_num)
# print(len(baseline_space))
print(len(baseline))
# 保存为json文件
# with open('baseline.json', 'w', encoding='utf-8') as fp:
#     json.dump(baseline_space, fp, indent=4)
# 读取json文件到dict
# with open('data.json', 'r') as f:
#     baseline_space = json.load(f)


# 保存为csv文件
with open('baseline27-margin50.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    counter = 0
    for row in baseline:
        counter += 1
        print('counter: ', counter)
        writer.writerow(row)
csvfile.close()
