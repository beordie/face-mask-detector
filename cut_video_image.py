"""
作者:30500
日期:2021年04月19日
简介:
"""
import cv2

def CutVideo2Image(video_path, img_path):
    #  将视频输出为图像
    #  video_path为输入视频文件路径
    #  img_path为输出图像文件夹路径
    cap = cv2.VideoCapture(video_path)
    index = 0
    while(True):
        ret,frame = cap.read()
        if ret:
            cv2.imwrite(img_path+'/%d.jpg'%index, frame)
            index += 1
        else:
            break
    cap.release()

CutVideo2Image('examples/test.avi', 'img')