"""
作者:30500
日期:2021年03月10日
简介:
"""
# 导包
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
import tensorflow as tf

# 指定当前程序使用的GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

#  构造参数解析器并解析参数
ap = argparse.ArgumentParser()

#  输入图像的路径
ap.add_argument("-i", "--image", required=False, default="examples/people_mask.jpg",
                help="path to input image")

# 人脸检测器模型目录的路径
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")

# 训练口罩检测器模型的路径
ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="path to trained face mask detector model")

#  过滤弱检测的最小概率
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#  从磁盘加载序列化人脸检测器模型 opnencv自带的人脸检测模型
print("[INFO] loading face detector model...")
#  加载配置和框架
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

#  加载口罩检测器模型
print("[INFO] loading face mask detector model...")
model = load_model(args["model"])

#  从磁盘加载输入图像,复制并抓取图像的尺寸大小
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]

# 从图像中构造一个BLOB 进行图像的预处理 缩放输入 减均值
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                             (104.0, 177.0, 123.0))

#  通过网络传递BLOB到模型 前向获得人脸检测结果
print("[INFO] computing face detections...")
net.setInput(blob)
detections = net.forward()

#  在探测上循环
for i in range(0, detections.shape[2]):

    # 提取与检测相关的置信度(即概率）
    confidence = detections[0, 0, i, 2]

    # 通过确保置信度大于最小置信度来过滤弱检测 > 0.5
    if confidence > args["confidence"]:
        # 计算对象的包围框的(x，y)坐标
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # 确保边界框在框架的尺寸范围内
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        #  提取人脸ROI，将其从BGR转换为RGB通道排序，将其调整为224x224，并对其进行预处理 输入给口罩模型进行判断是否佩戴口罩
        face = image[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # 通过模型来判断人脸是否有口罩
        (mask, withoutMask) = model.predict(face)[0]

        #  确定类标签和颜色，将使用它来绘制边框和文本
        label = "Mask" if mask > 0.75 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # 包括标签中的概率
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # 在输出框上显示标签和边框矩形
        cv2.putText(image, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# 显示输出图像
cv2.imwrite("result/people_mask_result.jpg", image)
cv2.imshow("Output", image)
cv2.waitKey(0)
