#! /usr/bin/env python
# coding=utf-8
import numpy as np
import cv2
# import matplotlib.pyplot as plt
from numpy import *
import math
import time
from sklearn.cluster import DBSCAN
from sklearn import metrics, svm
from scipy.spatial import ConvexHull
from collections import deque


def fp_variance(pt, ell):
    vec = [pt[1] - ell[1], pt[0] - ell[0]]
    dis = np.linalg.norm(vec)
    radius = (ell[3] + ell[2]) / float(2)
    return np.abs(dis / float(radius) - 0.5)


##        angle = ell[2]*np.pi/float(180)
##        if vec[0]>0:
##                phase = np.arctan(vec[1]/float(vec[0]))
##        elif vec[0]<0:
##                phase = np.arctan(vec[1]/float(vec[0]))+np.pi
##        else:
##                phase = np.pi/float(2)
##
##        phase = phase - angle
##        v1 = [ell[1][1]*np.cos(angle),ell[1][1]*np.sin(angle)]
##        angle = angle + np.pi/float(2)
##        v2 = [ell[1][0]*np.cos(angle),ell[1][0]*np.sin(angle)]
##        ell_pt =
[np.cos(phase) * v1[0] + np.sin(phase) * v2[0], np.cos(phase) * v1[1] + np.sin(phase) * v2[1]]


##        ell_pt[0] = ell_pt[0]-vec[0]
##        ell_pt[1] = ell_pt[1]-vec[1]
##        return np.linalg.norm(ell_pt)

def fp_LinearRegression(X, Y):
    L = len(X)
    Mat = np.zeros([2, 2])
    b = np.zeros([2, 1])
    for i in range(L):
        Mat[0][1] = Mat[0][1] + X[i]
        Mat[1][1] = Mat[1][1] + X[i] ** 2
        b[0][0] = b[0][0] + Y[i]
        b[1][0] = b[1][0] + Y[i] * X[i]
    Mat[0][0] = L
    Mat[1][0] = Mat[0][1]
    if np.linalg.matrix_rank(Mat) < 2:
        print
        X, Y
        return 0, 0
    sol = np.linalg.solve(Mat, b)
    residue = 0
    for i in range(L):
        residue = residue + (Y[i] - sol[1] * X[i] - sol[0]) ** 2
    # print residue/float(L)
    return sol, np.sqrt(residue / float(L))


def fp_perimeter(contour):
    L = len(contour)
    contour_list = list(contour)  # cyclic appending
    contour_list.append(contour[0])
    # print len(contour_list)
    contour_mat = []
    perimeter = cv2.arcLength(contour, True)
    for i in range(L):
        contour_mat.append(contour_list[i][0])
    return perimeter, contour_mat


def fp_reduction(n, kp):
    dict = {}
    coordinate = []
    s = "{a}\t{b}\n"
    for i in range(len(kp)):
        coor = kp[i].pt
        weight = kp[i].response
        if dict.has_key(s.format(a=int(coor[0] / n), b=int(coor[1] / n))) == True:
            dict[s.format(a=int(coor[0] / n), b=int(coor[1] / n))].append((coor[0], coor[1], weight))
        else:
            dict[s.format(a=int(coor[0] / n), b=int(coor[1] / n))] = [(coor[0], coor[1], weight)]
    # print len(dict)
    for key in dict:
        sumx = 0
        x_para = 0
        sumy = 0
        y_para = 0
        t = key.index("\t")
        n = key.index("\n")
        x = int(key[0:t]) / 2
        y = int(key[t:n]) / 2
        for a in dict[key]:
            sumx = sumx + a[0] * a[2] * (float(abs(a[0] - x)))
            sumy = sumy + a[1] * a[2] * (float(abs(a[1] - y)))
            x_para = x_para + a[2] * (float(abs(a[0] - x)))
            y_para = y_para + a[2] * (float(abs(a[1] - y)))
        coordinate.append((int(sumx / x_para), int(sumy / y_para)))
    return coordinate


def integral_bithresh(img, percen, s):
    outimg = np.zeros_like(img)
    w, h = img.shape
    img = img.astype(np.float32)
    intimg = np.zeros_like(img)

    for i in range(w):
        sum = 0
        for j in range(h):
            sum = img[i][j] + sum
            if i == 0:
                intimg[i][j] = sum
            else:
                intimg[i][j] = intimg[i - 1][j] + sum

    for i in range(w):
        for j in range(h):
            x1 = i - w / s
            x2 = i + w / s
            y1 = j - h / s
            y2 = j + h / s
            if x1 < 0:
                x1 = 0
            if x2 >= w:
                x2 = w - 1

            if y1 < 0:
                y1 = 0
            if y2 >= h:
                y2 = h - 1
            count = (x2 - x1) * (y2 - y1)
            sum = intimg[x2][y2] - intimg[x2][y1] - intimg[x1][y2] + intimg[x1][y1]
            if img[i][j] * count <= (sum * percen):

                outimg[i][j] = 0
            else:
                outimg[i][j] = 255
    return outimg


def fb_colorset(color, min_bound, max_bound):
    X = []
    Y = []
    Z = []
    min_BGR = [255, 255, 255]
    max_BGR = [0, 0, 0]
    for i in range(int(min_bound[0]), int(max_bound[0])):
        for j in range(int(min_bound[1]), int(max_bound[1])):
            BGR = color[j, i]
            for k in range(3):
                if BGR[k] > max_BGR[k]:
                    max_BGR[k] = BGR[k]
                if BGR[k] < min_BGR[k]:
                    min_BGR[k] = BGR[k]
            X.append(int(BGR[0]))
            Y.append(int(BGR[1]))
            Z.append(int(BGR[2]))
    c1, r1 = fp_LinearRegression(X, Y)
    c2, r2 = fp_LinearRegression(X, Z)
    return r1 + r2, sum(max_BGR) - sum(min_BGR)


def fb_ellip(color, imgray, n, m):  # ,n=50,m=0.2
    ##       kernel = np.ones((3,3),np.uint8)
    PCA_thresh = [[-0.02, 0.02], [-0.02, 0.02], [-0.025, 0.04]]
    imgray = cv2.bilateralFilter(imgray, 5, 60, 5)
    thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)
    # cv2.imshow('0',thresh)
    # thresh = cv2.Canny(thresh,600,100,3)#Canny
    # c,thresh = cv2.threshold(thresh,20,255,cv2.THRESH_BINARY)
    # thresh =cv2.erode(thresh,kernel,iterations = 1)
    # thresh = cv2.dilate(thresh,kernel,iterations = 1)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ellipse = []
    ell_pair = []
    cnts = []
    index = []
    features = []
    ratio = []
    for i in range(len(contours)):
        cnt = contours[i]
        LC = len(cnt)
        if LC > n:
            S1 = cv2.contourArea(cnt)
            ell = cv2.fitEllipse(cnt)
            S2 = math.pi * ell[1][0] * ell[1][1]
            if S2 != 0 and (S1 / S2) > m:
                pmr = ell[1][1] / float(ell[1][0])
                if pmr < 3:
                    w = hierarchy[0][i][2]
                    if w > -1:
                        cw = hierarchy[0][w][0]
                        LI = len(contours[w])
                        ##                                               ellipse.append((ell[0][0],ell[0][1],ell[1][1],ell[1][0],ell[2]))
                        ##                                               index.append(i)
                        while cw > -1:
                            LIC = len(contours[cw])
                            if LIC > LI:
                                LI = LIC
                                w = cw
                            cw = hierarchy[0][cw][0]
                        inner = contours[w]
                        IO = LI / float(LC)
                        if IO > 0.1:
                            # print [LI,LC]
                            ##                                                       ellipse.append((ell[0][0],ell[0][1],ell[1][1],ell[1][0],ell[2]))
                            ##                                                       index.append(i)
                            ##                                                       cnts.append(cnt)

                            if LI > 10 and IO > 0.3:
                                cell = cv2.fitEllipse(inner)
                                df = np.array(ell[0]) - np.array(cell[0])
                                cd = np.linalg.norm(df)
                                if cd < 0.05 * ell[1][1] and cell[1][0] / float(cell[1][1]) > 0.7:
                                    ellipse.append((ell[0][0], ell[0][1], ell[1][1], ell[1][0], ell[2]))
                                    index.append(i)
                                    cnts.append(w)
                                    ratio.append(IO)

                            else:
                                ellipse.append((ell[0][0], ell[0][1], ell[1][1], ell[1][0], ell[2]))
                                index.append(i)
                                cnts.append(w)
                                ratio.append(IO)
    LE = len(ellipse)
    if LE > 1:
        for i in range(LE):
            r1 = sum(ellipse[i][2:4])
            for j in range(i, LE):
                r2 = sum(ellipse[j][2:4])
                rm = max(r1, r2)
                cd = np.linalg.norm(np.array(ellipse[j][0:2]) - np.array(ellipse[i][0:2]))
                cr = 4 * cd / float(r1 + r2)
                if cr > 0.75 and cr < 1.5:
                    rb = rm / float(min(r1, r2))
                    if rb < 1.2:
                        pi, imat = fp_perimeter(contours[index[i]])
                        pj, jmat = fp_perimeter(contours[index[j]])
                        hull_i = ConvexHull(imat)
                        hull_j = ConvexHull(jmat)

                        S1 = cv2.contourArea(contours[index[i]])
                        S2 = cv2.contourArea(contours[index[j]])
                        if hull_i.volume / float(S1) < 1.1 and hull_j.volume / float(S2) < 1.1:
                            if pi / float(hull_i.area) < 1.1 and pj / float(hull_j.area) < 1.1:
                                var_j = fp_ellip_var(jmat, ellipse[j])
                                var_i = fp_ellip_var(imat, ellipse[i])
                                if var_j < 0.04 and var_i < 0.04:

                                    wi = abs(np.log2(ellipse[i][2] / float(ellipse[i][3])))
                                    wj = abs(np.log2(ellipse[j][2] / float(ellipse[j][3])))
                                    comp_i = 4 * S1 / float(np.pi * ellipse[i][2] * ellipse[i][3])
                                    comp_j = 4 * S2 / float(np.pi * ellipse[j][2] * ellipse[j][3])
                                    cvx_i = np.log2(pi / float(hull_i.area))
                                    cvx_j = np.log2(pj / float(hull_j.area))
                                    io_i = -np.log2(ratio[i])
                                    io_j = -np.log2(ratio[j])
                                    ct = np.array([0, 0])
                                    inner = contours[cnts[i]]
                                    for k in inner:
                                        ct = ct + k[0]
                                    ct = ct / float(len(inner))
                                    cd_i = np.linalg.norm(np.array(ellipse[i][0:2]) - ct)
                                    ct = np.array([0, 0])
                                    inner = contours[cnts[j]]
                                    for k in inner:
                                        ct = ct + k[0]
                                    ct = ct / float(len(inner))
                                    cd_j = np.linalg.norm(np.array(ellipse[j][0:2]) - ct)
                                    ell_pair.append(ellipse[i])
                                    ell_pair.append(ellipse[j])
                                    cur_ft = [cd_i / float(ellipse[i][2]), cd_j / float(ellipse[j][2]), np.log2(rb), cr,
                                              cd_i, cd_j, wi, wj, io_i, io_j, cvx_i, cvx_j, var_i, var_j,
                                              abs(np.log2(comp_i)), abs(np.log2(comp_j))]
                                    features.append(cur_ft)
    cv2.drawContours(color, contours, -1, (0, 255, 0), 2)
    features = np.array(features)
    return thresh, ell_pair, cnts, features


def fp_ellip_var(jmat, ell):
    ell_var = 0
    for m in jmat:
        # print jmat[m],ell
        var = fp_variance(m, ell)
        ell_var = ell_var + var
    ell_var_j = ell_var / float(len(jmat))
    return ell_var_j


def fb_localpoint(img, Ap, scale, enum):
    L = len(Ap)
    c = np.pi / float(180)
    S = img.shape
    ie = 6
    eps = 60
    threshold = 0.8
    Wp = []
    for i in range(L):
        rad = c * Ap[i][4]
        x = scale * max(np.abs(np.cos(rad) * Ap[i][2]), np.abs(np.sin(rad) * Ap[i][3]))
        y = scale * max(np.abs(np.cos(rad) * Ap[i][3]), np.abs(np.sin(rad) * Ap[i][2]))
        local_sz = max(x, y)
        xi = max(0, int(Ap[i][0] - local_sz / 2))
        xe = min(S[0], int(Ap[i][0] + local_sz / 2))
        yi = max(0, int(Ap[i][1] - local_sz / 2))
        ye = min(S[1], int(Ap[i][1] + local_sz / 2))
        if ye - yi < 1 or xe - xi < 1:
            Info = [Ap[i][0], Ap[i][1], Ap[i][2], Ap[i][3], Ap[i][4], xi, xe, yi, ye]
            Wp.append(Info)
            continue
        if local_sz > enum[1]:
            local_img = img[yi:ye, xi:xe]
            sz = max(int(local_sz / 20), 2)

            orb = cv2.ORB(nfeatures=80, edgeThreshold=sz, patchSize=sz)
            kp = orb.detect(local_img, None)
            Up = fp_reduction(sz, kp)
            rank = np.linalg.matrix_rank(Up)
            if len(Up) > 4 and rank > 1:
                hull = ConvexHull(Up)
                vertices = hull.vertices
                vl = len(vertices)
                if vl > enum[0]:
                    Info = [Ap[i][0], Ap[i][1], Ap[i][2], Ap[i][3], Ap[i][4], xi, xe, yi, ye]
                    Wp.append(Info)
        ##                                for k in range(vl):
        ##                                        cv2.circle(color_img,(int(Up[k][0]+xi),int(Up[k][1]+yi)),5,(255,0,0),-1)

        else:
            Info = [Ap[i][0], Ap[i][1], Ap[i][2], Ap[i][3], Ap[i][4], xi, xe, yi, ye]
            Wp.append(Info)

    return Wp


def fp_draw(Ap, img):
    for i in range(len(Ap)):
        a = int(Ap[i][0])
        b = int(Ap[i][1])
        cv2.circle(img, (a, b), 10, (0, 0, 255), -1)
    return img


def fp_decision(pt, fr, sque, df):
    ql = len(sque)
    print(fr)
    if ql == 0:
        # print sque
        return False
    else:
        for i in range(ql):
            Gp = sque[i][0]
            for j in range(len(Gp)):
                dist = np.sqrt((pt[0] - Gp[j][0]) ** 2 + (pt[1] - Gp[j][1]) ** 2)

                if dist / float(fr - sque[i][2]) < df and sque[i][1][j] > 0:
                    return True

    return False


lk_params = dict(winSize=(30, 30),
                 maxLevel=1,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

camera = cv2.VideoCapture(0)  #

ret, frame = camera.read()  #
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #
kernel = np.ones((5, 5), np.uint8)  #
fourcc = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')  #
out = cv2.VideoWriter('output_withsvm2.avi', cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), 15.0, (1280, 720))  #
ret = 1  #
fr = 0
sque = deque([])
Ft = []
avg = 0
Fts = np.load("features.npy")
labels = np.load("ft_labels.npy")
clf = svm.SVR(C=1000)
clf.fit(Fts, labels)
while (ret):
    start_q = time.clock()
    ret, frame = camera.read()
    end_q = time.clock()
    # frame = cv2.pyrDown(frame)
    start = time.clock()
    # print frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img, Gp, contours, features = fb_ellip(frame, gray, 7, 0.2)
    for i in features:
        Ft.append(i)
    fr = fr + 1
    Hp = []
    end = time.clock()
    avg = avg + end - start + end_q - start_q
    print(avg / float(fr))
    for j in range(len(features)):
        cls = clf.predict([features[j]])
        if cls > 0.8:
            if len(features) > 1:
                print
                cls
            cv2.ellipse(frame, (int(Gp[2 * j][0]), int(Gp[2 * j][1])), (int(Gp[2 * j][3]) / 2, int(Gp[2 * j][2]) / 2),
                        0, 0, 360, (0, 0, 255), 4)
            cv2.ellipse(frame, (int(Gp[2 * j + 1][0]), int(Gp[2 * j + 1][1])),
                        (int(Gp[2 * j + 1][3]) / 2, int(Gp[2 * j + 1][2]) / 2), 0, 0, 360, (0, 0, 255), 4)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 960, 540)
    cv2.imshow('frame', frame)
    out.write(frame)
    old_gray = gray.copy()
    k = cv2.waitKey(30) & 0xff
    if (k & 0xff == ord('q')):
        break
camera.release()
cv2.destroyAllWindows()

