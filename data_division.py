import cv2
import os

path = 'D:/Dataset/new_scale/origin/'
myClasses = os.listdir(path)
print(myClasses)


for i in myClasses:
    count = 0
    mydata = os.listdir(path + i)
    print('class: ', i)
    num = 0
    for j in mydata:
        num += 1
        print('num: ', num)
        img = cv2.imread(path + i + '/' + j)

        if count == 12:
            count = 0
        if count <= 9:
            cv2.imwrite('D:/Dataset/new_scale/Train/' + i + '/' + i + '_' + str(num) + '.jpg', img)
        elif count == 10:
            cv2.imwrite('D:/Dataset/new_scale/Val/'   + i + '/' + i + '_' + str(num) + '.jpg', img)
        elif count == 11:
            cv2.imwrite('D:/Dataset/new_scale/Test/'  + i + '/' + i + '_' + str(num) + '.jpg', img)

        count = count + 1