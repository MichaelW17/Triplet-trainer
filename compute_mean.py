# 用于电子秤项目中计算origin文件夹下所有图片的三通道均值
import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import time

path = 'D:/Dataset/new_scale/Train/'
myClasses = os.listdir(path)
print(myClasses)

# cv2.namedWindow(' ', flags=cv2.WINDOW_NORMAL)
# cv2.resizeWindow(' ', 600, 600)
# -------------------------------先缩放图像为960×960--------------------------------------
# for i in myClasses:
#     mydata = os.listdir(path + i)
#     for j in mydata:
#         img = cv2.imread(path + i + '/' + j)
#         print('img_path: ', path + i + '/' + j)
#         img = cv2.resize(img, (957, 957), interpolation=cv2.INTER_AREA)
#         cv2.imwrite(path + i + '/' + j, img)
#
#         keyvalue = cv2.waitKey(1)
#         if keyvalue == 27:
#             break
#     if keyvalue == 27:
#             break
# cv2.destroyWindow(' ')

t0 = time.time()
# -----------------------------------计算均值----------------------------------------------
img_holder = np.zeros((957, 957, 3), dtype=np.float32)  # to add up all the image data
img_num = 0  # number of images accumulated
for i in myClasses:
    count = 0
    mydata = os.listdir(path + i)
    for j in mydata:
        img = Image.open(path + i + '/' + j)
        for k in range(-30, 30):
            img_holder += img.rotate(k*6)
            img_num += 1
            # plt.imshow(img.rotate(i))
            # plt.show()
print('img_num: ', img_num)
img_holder = np.sum(img_holder, axis=(0, 1)) / 957**2
img_mean = img_holder / img_num
print(img_mean)  # 未经旋转[71.35, 69.19, 59.52]  # 旋转后[67.35 65.08 56.17]
print('time consumed: ', time.time()-t0, 's')