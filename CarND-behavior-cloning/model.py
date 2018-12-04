import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import pandas as pd
from PIL 
import csv
from sklearn.model_selection import *
from sklearn.utils import shuffle
from keras.layers import *
from keras.models import *
from keras.layers.normalization import *



IMG_path='data_collected'
width, height=320, 160
samples=[]
with open('data_collected\driving_log.csv', 'r') as f:
    reader=csv.reader(f)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=32):
    n_len=len(samples)
    while True:
        shuffle(samples)
        for i in range(0, n_len, batch_size):
            X=[]
            y=[]
            batches=samples[i:i+batch_size]
            for row in batches:
                steering_center=float(row[3])
                steering_left=steering_center+correction
                steering_right=steering_center-correction
                
                img_center=np.asarray(PIL.Image.open(row[0]),dtype=np.uint8)
                img_left=np.asarray(PIL.Image.open(row[1]),dtype=np.uint8)
                img_right=np.asarray(PIL.Image.open(row[2]),dtype=np.uint8)
                img_flip=np.fliplr(img_center)# data augmentation with flip the image
                
                X.extend([img_center, img_left, img_right,img_flip])
                y.extend([steering_center, steering_left, steering_right,-steering_center])
               
            X=np.array(X)
            y=np.array(y)
            yield shuffle(X, y)



model=Sequential()

model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(height, width ,3)))
model.add(Lambda(lambda x: x/255.0-0.5))

model.add(Conv2D(filters=16, kernel_size=3, strides=2, activation='relu',input_shape=(height, width, 1)))
model.add(Conv2D(filters=16, kernel_size=3, strides=1, activation=None))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
model.add(Dropout(0.5))

model.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu'))
model.add(Conv2D(filters=32, kernel_size=3, strides=1, activation=None))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
model.add(Dropout(0.5))


model.add(Conv2D(filters=32, kernel_size=3, strides=1, activation='relu'))
model.add(Conv2D(filters=32, kernel_size=3, strides=1, activation=None))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))
model.add(Dropout(0.5))


model.add(Flatten())


model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(100, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(1, activation=None))
model.compile(optimizer='adam', loss='mse')

train_generator=generator(train_samples, 32)
valid_generator=generator(validation_samples, 32)


model.fit_generator(train_generator, steps_per_epoch=1000, epochs=8,
                    validation_data=valid_generator, validation_steps=100)
