"""
作者:30500
日期:2021年04月19日
简介:
"""
import cv2
import os

# 初始化视频流
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_yolo.avi', fourcc, 20.0, (1920, 1080))


def image2video(path='video'):
    files = os.listdir(path)
    for i in range(len(files)):
        img = cv2.imread(path + '/%d.jpg' % i)
        out.write(img)
    out.release()
