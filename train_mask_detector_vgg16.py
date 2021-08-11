"""
作者:30500
日期:2021年02月26日
简介:
"""
"""
作者:30500
日期:2021年02月26日
简介:
"""

# 导包
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# 指定当前程序使用的GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 构造参数解析器并解析参数
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default="dataset", help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot_vgg16.png", help="path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="mask_detector_vgg16.model",
                help="path to output face mask detector model")
args = vars(ap.parse_args())

# 设置初始化初始学习速率，要训练的迭代数和批次读入图片的大小
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

# 获取数据集目录中的图像列表，然后初始化数据列表（即图像）和图像标签
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

#  在图像路径上循环 将训练数据和标签数据读入
for imagePath in imagePaths:
    # 利用路径分隔符 从文件名中提取类标签
    label = imagePath.split(os.path.sep)[-2]
    # 测试代码
    # label1 = imagePath.split(os.path.sep)[1]
    # print(label, label1)

    # 加载输入图像并对其进行预处理 读入大小为（224*224）为了适用MobileNetV2模型的输入
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    # 分别更新数据和标签列表 将图片数据和对应标签数据添加到列表中
    data.append(image)
    labels.append(label)

# 将数据和标签转换为NumPy数组格式 便于数据的处理
data = np.array(data, dtype="float32")
labels = np.array(labels)

# 将with_mask without_mask标签转换成二值模式并转换成 one hot 格式
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# 进行数据的划分
# 将数据划分为使用75%用于训练
# 剩余的25%用于测试
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# 构建用于数据增强的训练图像生成器
aug = ImageDataGenerator(
    rotation_range=20,  # 旋转范围
    zoom_range=0.15,  # 缩放范围
    width_shift_range=0.2,  # 水平平移范围
    height_shift_range=0.2,  # 垂直平移范围
    shear_range=0.15,  # 透视变换范围
    horizontal_flip=True,  # 水平翻转
    fill_mode="nearest")  # 填充模式

# 加载 MobileNetV2 网络，将数据的输出层关闭，只用于提取数据的特征用于构建自己的处理模型
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# 构造顶部的模型放置在模型头部
# 基本模型 平均池化层 全连接层 隐藏层 Dropout层 输出层
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="sigmoid")(headModel)

# 模型整合
model = Model(inputs=baseModel.input, outputs=headModel)

# 冻结VGG16网络中加载的模型层 不再进行相应的权重更新
for layer in baseModel.layers:
    layer.trainable = False

# 模型配置 损失函数二分类 优化参数加入学习率衰减
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# 训练网络
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),  # 数据打乱增强
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),  # 模型的数据验证
    validation_steps=len(testX) // BS,
    epochs=2)

# 对测试集进行预测
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# 对于测试集中的每个图像，需要找到具有相应最大预测概率的标签的索引
predIdxs = np.argmax(predIdxs, axis=1)

# 显示格式的分类报告
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# 将模型保存
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")

# plot the training loss and accuracy
# 绘制训练损失和准确性
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
