# 视频人流口罩识别

​    通过抓取视频接口的视频流数据作为输入送给算法进行处理、或者直接将图片输入给算法，输出标记完成的视频或图片，其中包括人脸的检测及口罩的识别标记。

# 一、项目背景

​    虽然国内的疫情形势比较稳定，但是不排除境外输入的可能，因此在公共场合应该佩戴口罩，但是存在某些人群抱着侥幸心理进入公共场合不佩戴口罩，如果在每个场所出入口都设置人员进行排查将会浪费大量的人力、物力，因此萌发了设计一个能够自动检测行人是否戴口罩算法模型的想法。  

> 最终效果展示

<center><img src="E:\file\object file\Baidu AI创造营\150.jpg"   width="70%"></center>


# 二、数据集简介

​    戴口罩和不戴口罩的图片总量为5000张，通过图片增强的预处理方式再次增大样本数量，测试和训练的比例为2:8，更好的让模型进行权重的更新学习，为了防止分类器出现过拟合的现象，再次用1000张没有训练过的图片进行模型测试。  

## 1.数据加载和预处理


```python
# 将数据和标签转换为NumPy数组格式 便于数据的处理
data = np.array(data, dtype="float32")
labels = np.array(labels)

# 将with_mask without_mask标签转换成二值模式并转换成 one hot 格式
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# 进行数据的划分
# 将数据划分为使用80%用于训练
# 剩余的20%用于测试
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
```


## 2.数据集查看


```python
#  在图像路径上循环 将训练数据和标签数据读入
for imagePath in imagePaths:
    # 利用路径分隔符 从文件名中提取类标签
    label = imagePath.split(os.path.sep)[-2]

    # 加载输入图像并对其进行预处理 读入大小为（224*224）为了适用MobileNetV2模型的输入
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    # 分别更新数据和标签列表 将图片数据和对应标签数据添加到列表中
    data.append(image)
    labels.append(label)
    
# 可视化展示
plt.figure()
plt.imshow(data, cmap=plt.cm.binary)
plt.show()
```


# 三、模型选择和开发

采用 `MobileNetV2` 来进行特征提取，`YOLO3` 人脸检测。

## 1.模型流程

![1628665998701](C:\Users\30500\AppData\Roaming\Typora\typora-user-images\1628665998701.png)


```python
# 模型网络结构搭建
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# 构造顶部的模型放置在模型头部
# 基本模型 平均池化层 全连接层 隐藏层 Dropout层 输出层
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="sigmoid")(headModel)
```

## 2.模型网络结构可视化


```python
# 模型整合
model = Model(inputs=baseModel.input, outputs=headModel)

# 冻结MobileNetV2网络中加载的模型层 不再进行相应的权重更新
for layer in baseModel.layers:
    layer.trainable = False
```

    _________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to             
    =========================================================================================
    input_1 (InputLayer)            [(None, 224, 224, 3) 0                                   
    _________________________________________________________________________________________
    Conv_1_bn (BatchNormalization)  (None, 7, 7, 1280)   5120        Conv_1[0][0]             
    _________________________________________________________________________________________
    out_relu (ReLU)                 (None, 7, 7, 1280)   0           Conv_1_bn[0][0
    _________________________________________________________________________________________
    average_pooling2d (AveragePooli (None, 1, 1, 1280)   0           out_relu[0][0]           
    _________________________________________________________________________________________
    flatten (Flatten)               (None, 1280)         0           average_pooling2d[0][0] 
    _________________________________________________________________________________________
    dense (Dense)                   (None, 128)          163968      flatten[0][0]           
    _________________________________________________________________________________________
    dropout (Dropout)               (None, 128)          0           dense[0][0]             
    _________________________________________________________________________________________
    dense_1 (Dense)                 (None, 2)            258         dropout[0][0]           
    =========================================================================================
    Total params: 2,422,210
    Trainable params: 164,226
    Non-trainable params: 2,257,984
    _________________________________________________________________________________________

## 3.模型训练


```python
# 模型配置 损失函数二分类 优化参数加入学习率衰减
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
model.summary()
# 训练网络
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),  # 数据打乱增强
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),  # 模型的数据验证
    validation_steps=len(testX) // BS,
    epochs=2)
```

    Epoch 1/2
     1/58 [..............................] - ETA: 5:50 - loss: 0.8232 - accuracy: 0.62502021
     2/58 [>.............................] - ETA: 52s - loss: 0.8053 - accuracy: 0.6172



## 4.模型评估测试


```python
# 对测试集进行预测
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# 对于测试集中的每个图像，需要找到具有相应最大预测概率的标签的索引
predIdxs = np.argmax(predIdxs, axis=1)

# 显示格式的分类报告
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))
```

```python
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
```

## 5.模型预测

​    通过输入具体的预测文件路径来对路径下的所有图片进行预测。


```python
GetFace('img', 'demo', masknet)
```

```
。。。。。。。。。。。
正在进行197张图的处理
正在进行198张图的处理
正在进行199张图的处理
。。。。。。。。。。。
```

# 四、效果展示

将需要进行预测视频放置在 `examples` 文件目录下。

```python
CutVideo2Image('examples/文件名.avi', 'img')
```

输出预测完的视频在根目录下。

```python
cv2.VideoWriter('output_yolo.avi', fourcc, 20.0, (1920, 1080))
```

将文件地址配置完成后，项目可直接在 `pytharm` 中进行启动，启动脚本为 `yolo_detetion.py`。



# 五、总结与升华

​    视频检测部分，利用OpenCV的API函数调取摄像头进行检测，将图片分帧处理完毕之后进行合并输出，但限于电脑算力问题，实时检测时遇到卡帧的现象，因此将摄像头的数据替换为一段视频进行检测，最终的检测效果也不错，但是对于信息复杂的视频，caffe不能很好的进行检测，因此将视频模块的模型替换为YOLO3，达到很好的检测效果。

写写你在做这个项目的过程中遇到的坑，以及你是如何去解决的。

最后一句话总结你的项目

# 个人简介

模型更新地址：https://aistudio.baidu.com/aistudio/personalcenter/thirdview/699500