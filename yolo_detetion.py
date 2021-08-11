import cv2
import os
import numpy as np
import paddlehub as hub
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from cut_video_image import CutVideo2Image
from image_video import image2video
masknet = load_model('mask_detector.model')


def Iou(bbox1, bbox2):
    # 计算Iou
    # bbox1,bbox为xyxy数据
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    w = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])
    h = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])
    if w <= 0 or h <= 0:
        return 0
    area_mid = w * h
    return area_mid / (area1 + area2 - area_mid)


def GetFace(in_path, out_path, maskNet):
    # in_path为输入图像文件夹的路径
    # out_path为输出图像文件夹的路径
    files = os.listdir(in_path)
    face_detector = hub.Module(name="pyramidbox_lite_server")
    for i in range(len(files)):
        faces = []
        preds = []
        # 文件中的每张图片
        img = cv2.imread(in_path + '/%d.jpg' % i)
        result = face_detector.face_detection(images=[img])
        img = img_to_array(img)
        data = result[0]['data']
        bbox_upgrade = []
        index = []
        for j in range(len(data)):
            # 图片中的每个bbox
            left, right = int(data[j]['left']), int(data[j]['right'])
            top, bottom = int(data[j]['top']), int(data[j]['bottom'])
            bbox = (left, top, right, bottom)

            if right > 1600 and bottom > 1600:
                for k in range(len(bbox_buffer)):
                    if Iou(bbox, bbox_buffer[k]) > 0.1 and k not in index:
                        index.append(k)
                        break
                bbox_upgrade.append((left, top, right, bottom))
            else:
                preds.append([left, top, right, bottom])
                faces.append(img[top:bottom, left:right])
        bbox_buffer = bbox_upgrade.copy()

        if len(faces) > 0:
            count = 0
            for face in faces:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
                (mask, withoutMask) = maskNet.predict(face)[0]
                lable = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if lable == "Mask" else (0, 0, 255)
                lable = "{}:{:.2f}%".format(lable, max(mask, withoutMask) * 100)
                cv2.putText(img, lable, (preds[count][0], preds[count][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color,
                            2)
                cv2.rectangle(img, (preds[count][0], preds[count][1]), (preds[count][2], preds[count][3]), color, 2)
                count += 1
        cv2.imwrite(out_path + '/%d.jpg' % i, img)
        print('正在进行{}张图的处理'.format(i))

CutVideo2Image('examples/test.avi', 'img')
GetFace('img', 'video', masknet)
image2video('video')