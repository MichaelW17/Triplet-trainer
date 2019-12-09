import random
import numpy as np


# print(random.sample([random.randint(0, 10) for _ in range(10)], 3))
classes = [random.randint(0, 20) for _ in range(5)]
r = np.random.permutation(len(classes))
print(classes)
print(r)
classes = np.array(classes)
print(classes[r])

#-----------------------------------------------------------------------------------------
# 通过简单的数据检查tsne的使用方法是否存在问题
# import numpy as np
# from keras.utils import np_utils
#
#
# data = np.arange(0, 10)
# data = np.repeat(data, 10)
# classes = data
# # data = np.expand_dims(data, axis=1)
# data = np_utils.to_categorical(data, num_classes=20)
#
# data = data.astype(np.float)
# noise = np.random.normal(0, 0.1, (100, 10))
# # noise = np.expand_dims(noise, axis=1)
# data += noise
#
# print(data)
#
# from sklearn.manifold import TSNE
# from TSNE_plot import scatter
# tsne = TSNE()
# tsne_embeds = tsne.fit_transform(data)
# scatter(tsne_embeds, classes)


#-----------------------------------------------------------------------------------------
# import csv
# datas = [[1,2,3], [4,5,6]]
# with open('test1.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     counter = 0
#     for row in datas:
#         counter += 1
#         print('counter: ', counter)
#         writer.writerow(row)
# csvfile.close()
#
# def readCSV2List(filePath):
#     try:
#         file = open(filePath, 'r')  # 读取以utf-8
#         context = file.read()  # 读取成str
#         list_result=context.split("\n")  # 以回车符\n分割成单独的行
#         list_result.pop()
#         #每一行的各个元素是以【,】分割的，因此可以
#         length = len(list_result)
#         print('length: ', length)
#         for i in range(length):
#             list_result[i] = list_result[i].split(",")
#         return list_result
#     except Exception :
#         print("文件读取转换失败，请检查文件路径及文件编码是否正确")
#     finally:
#         file.close()  # 操作完成一定要关闭
#
# list1 = readCSV2List('test1.csv')
# print('list: ', list1)



# import json
# import numpy as np
# with open('baseline.json', 'r') as f:
#     baseline_space = json.load(f)
# print(len(baseline_space))
# class0 = baseline_space['0']
# print(np.array(class0).shape)
# print(type(class0))


# import cv2
# import glob
# import json
#
# d = {0: [1,2,3], 1: [4,5,6]}
# with open('data.json', 'w') as fp:
#     json.dump(d, fp)
#
#
# with open('data.json', 'r') as f:
#     data = json.load(f)
#
# print(data)

