"""
作者:30500
日期:2021年04月19日
简介:
"""
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import os
import paddlehub as hub

def Iou(bbox1,bbox2):
    #计算Iou
    #bbox1,bbox为xyxy数据
    area1 = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
    area2 = (bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])
    w = min(bbox1[3],bbox2[3])-max(bbox1[1],bbox2[1])
    h = min(bbox1[2],bbox2[2])-max(bbox1[0],bbox2[0])
    if w<=0 or h<=0:
        return 0
    area_mid = w*h
    return area_mid/(area1+area2-area_mid)

def detect_and_predect_mask(img, faceNet):
    # 人脸检测
    results = faceNet.face_detection(images=[img])
    data = results[0]['data']
    # 初始化人脸列表
    index = []
    faces = []
    # 在探测器上循环
    for i in range(0, len(data)):
        # 概率检测
        confidence = data[i]['confidence']
        if confidence > 0.9:
            left, right = int(data[i]['left']), int(data[i]['right'])
            top, bottom = int(data[i]['top']), int(data[i]['bottom'])
            bbox = (left, top, right, bottom)

            if right > 1600 and left < 1600:
                for k in range(len(faces)):
                    if Iou(bbox, faces[k]) > 0.1 and k not in index:
                        index.append(k)
                        break
                faces.append(bbox)
            face = img[left:right, top:bottom]
            cv2.imwrite("video/{}.jpg".format(i), face)

#  加载序列化人脸检测器模型
print("[INFO] loading face detector model...")
facenet = hub.Module(name="pyramidbox_lite_server")


files = os.listdir('img')
# 从视频流循环读取
for i in range(1):
    img = cv2.imread('img' + '/%d.jpg' % i)
    locs, preds = detect_and_predect_mask(img, facenet)
