"""
作者:30500
日期:2021年03月11日
简介:
"""
# 导包
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import tensorflow as tf

# 指定使用的本地GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def detect_and_predect_mask(frame, faceNet, maskNet):
    # 获取框架尺寸
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detection = faceNet.forward()
    # 初始化人脸列表
    faces = []
    locs = []
    preds = []
    # 在探测器上循环
    for i in range(0, detection.shape[2]):
        # 概率检测
        confidence = detection[0, 0, i, 2]
        if confidence > args["confidence"]:
            box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startx, starty, endx, endy) = box.astype("int")
            (startx, starty) = (max(0, startx), max(0, starty))
            (endx, endy) = (min(w - 1, endx), min(h - 1, endy))

            face = frame[starty:endy, startx:endx]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startx, starty, endx, endy))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    return locs, preds


# 构造解析器参数
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#  从磁盘加载序列化人脸检测器模型
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
facenet = cv2.dnn.readNet(prototxtPath, weightsPath)

#  加载口罩检测器模型
print("[INFO] loading face mask detector model...")
masknet = load_model(args["model"])

# 初始化视频流，调用摄像头
print("[INFO] starting video stream...")
vs = cv2.VideoCapture("examples/test.avi")
time.sleep(2.0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(vs.get(3)),int(vs.get(4))))

# 从视频流循环读取
while True:
    ret, frame = vs.read()
    if ret == False:
        break
    locs, preds = detect_and_predect_mask(frame, facenet, masknet)
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        lable = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if lable == "Mask" else (0, 0, 255)
        lable = "{}:{:.2f}%".format(lable, max(mask, withoutMask) * 100)
        cv2.putText(frame, lable, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    out.write(frame)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.release()
out.release()
cv2.destroyAllWindows()



