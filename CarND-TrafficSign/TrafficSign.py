#! /usr/bin/env python  
# -*- coding:utf-8 -*-  
# Load pickled data
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random


# TODO: Fill this in based on where you saved the training and testing data

training_file = "data/train.p"
validation_file = "data/valid.p"
testing_file = "data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# 可视化观察数据形状
print("Train")
print("***************")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("")
print("Validation")
print("***************")
print("X_valid shape:", X_valid.shape)
print("y_valid shape:", y_valid.shape)
print("")
print("Test")
print("***************")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

####################################################

### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
# 抽取一个输入标本，这里使用 X_train[0]
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.


n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#########################################################
#
# # 随机选取训练集中的 10 张图片进行显示
# fig, axs = plt.subplots(2,5, figsize=(15,6))
# axs = axs.ravel() #返回一个连续的扁平数组,将二维的 axs 转化为一维的数组。
#
#
# for i in range(10):
#     index = random.randint(0, n_train)
#     image = X_train[index]
#     axs[i].imshow(image)
#     axs[i].axis("off") # 不显示坐标系
#     axs[i].set_title(y_train[index]) # 显示图像标签
#
#
# # 哈哈，对训练集图像类型有了一定的了解了。
# # 下面我们就需要对训练集的标签数据进行了解了。
#
# # 绘制标签数据的直方图
# hist, bins = np.histogram(y_train, bins=n_classes)
# width = (bins[1] - bins[0]) * 0.7
# center = (bins[:-1] + bins[1:]) / 2  # 取数组中前后两个数的平均构成新的数组
#
# plt.bar(center, hist, align="center", width=width)
# plt.show()

#######################################################################

# 在 Yann LeCun 的论文中提到，使用灰度图可以进步提高分类的准确度。那么我们首先尝试将图像转化为灰度图

# 保留一下原始格式的图像，万一后面有用呢？
X_train_rgb = X_train
X_train_gry = np.sum(X_train/3, axis=3, keepdims=True)
X_valid_rgb = X_valid
X_valid_gry = np.sum(X_valid/3, axis=3, keepdims=True)
X_test_rgb = X_test
X_test_gry = np.sum(X_test/3, axis=3, keepdims=True)

# 输出，转化后的图像尺寸
print("Train")
print("***********************")
print("X_train_rgb shape:", X_train_rgb.shape)
print("X_train_gry shape:", X_train_gry.shape)

# 实时进行可视化观察永远不是一个坏主意
# 将 原始图像 与 灰度图 进行可视化比较
# 选取 10 张随机图

# fig, axs = plt.subplots(2, 10, figsize=(20, 4))
# fig.subplots_adjust(hspace=0.1, wspace=0.01)
# axs = axs.ravel()
#
# for i in range(10):
#     index = random.randint(0, n_train)
#     rgb_image = X_train_rgb[index]
#     gry_image = X_train_gry[index].squeeze()
#
#     axs[i].axis("off")
#     axs[i].imshow(rgb_image)
#     axs[i + 10].axis("off")
#     axs[i + 10].imshow(gry_image, cmap="gray")

# 灰度化完成，下面就要进行归一化操作
X_train_norm = (X_train_gry - 128) / 128
X_valid_norm = (X_valid_gry - 128) / 128
X_test_norm = (X_test_gry - 128) / 128

print("Normalized X_train_gry shape:", X_train_norm.shape)
print("Normalized X_valid_gry shape:", X_valid_norm.shape)
print("Normalized X_test_gry shape:", X_test_norm.shape)




# 可视化归一化操作前后的图像

fig, axs = plt.subplots(2, 5, figsize=(15, 6))
fig.subplots_adjust(hspace=0.2, wspace=0.01)
axs = axs.ravel()

for i in range(5):
    index = random.randint(0, n_train)
    gry_image = X_train_gry[index].squeeze()  # 减少一个维度的数据，消除深度数据
    norm_gry_image = X_train_norm[index].squeeze()

    axs[i].axis("off")
    axs[i].imshow(gry_image, cmap="gray")
    axs[i + 5].axis("off")
    axs[i + 5].imshow(norm_gry_image, cmap="gray")

plt.show()

# 此处随机选取的数据集不太好，对于进行归一化前后的图像变化显示的不是比较明显。
# 其实，通过归一化使用可以有一定效果的提升图像显示质量。
# 大家可以手动选取一些数据集，进行观察。


