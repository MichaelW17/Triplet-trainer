# 用于电子秤项目中计算origin文件夹下所有图片的三通道均值
import os
import numpy as np
from PIL import Image

path = 'D:/Dataset/lilscale50/origin/'
myClasses = os.listdir(path)
print(myClasses)

img_holder = np.zeros((957, 957, 3), dtype=np.float32)  # to add up all the image data
img_num = 0  # number of images accumulated
for i in myClasses:
    count = 0
    mydata = os.listdir(path + i)
    for j in mydata:
        img_num += 1
        img = Image.open(path + i + '/' + j)
        img_holder += img
print('img_num: ', img_num)
img_holder = np.sum(img_holder, axis=(0,1)) / 957**2
img_mean = img_holder / img_num
print(img_mean)
