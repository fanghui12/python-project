import tensorflow as tf
import numpy as np
import pickle

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


n_classes = len(np.unique(y_train))
print("Number of classes =", n_classes)


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



# 灰度化完成，下面就要进行归一化操作
X_train_norm = (X_train_gry - 128) / 128
X_valid_norm = (X_valid_gry - 128) / 128
X_test_norm = (X_test_gry - 128) / 128

print("Normalized X_train_gry shape:", X_train_norm.shape)
print("Normalized X_valid_gry shape:", X_valid_norm.shape)
print("Normalized X_test_gry shape:", X_test_norm.shape)


test_labels = [3, 11, 1, 12, 38, 34, 18, 25]
save_file = 'lenet/train_model.ckpt'
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, save_file)