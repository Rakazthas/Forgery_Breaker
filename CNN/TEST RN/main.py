import keras.optimizer_v1
import numpy
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras import Model
import tensorflow.keras as tk
import tensorflow_datasets as tds
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)
import pywt


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout


from PIL import Image, ImageChops, ImageEnhance
import os
import random




'''
Programme pour tester le RN entrainée sur la db casia v1.
'''

X = []
Y = []

path = r"C:\\Users\\Yann\\Desktop\\dataset\\CMF-Dataset-master\\NB-CASIA"
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('tif'):
            full_path = os.path.join(dirname, filename)
            im=np.asarray(Image.open(full_path).convert('RGB').resize((227,227)))
            X.append(im)
            if filename.startswith('Au'):
                Y.append(1)
            else:
                if filename.startswith('Tp'):
                    Y.append(0)
            if len(Y) % 500 == 0:
                print(f'Processing {len(Y)} images')


print(len(X), len(Y))
X = np.array(X)
Y = to_categorical(Y, 2)


vgg16 = VGG16(weights="imagenet",include_top=False,input_shape=(227,227,3))


RN2TRAIN=Sequential()
RN2TRAIN.add(Conv2D(11 * 11, (4, 4), padding='same', activation='relu'))
RN2TRAIN.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
RN2TRAIN.add(Conv2D(5 * 5, (1, 1), padding='valid', activation='relu'))
RN2TRAIN.add(MaxPooling2D(pool_size=(3 * 3), strides=(2, 2), padding='same'))
RN2TRAIN.add(Conv2D(3 * 3, (1, 1), padding='valid', activation='relu'))
RN2TRAIN.add(Conv2D(3 * 3, (1, 1), padding='valid', activation='relu'))
RN2TRAIN.add(Conv2D(6 * 6, (1, 1), padding='same', activation='relu'))
RN2TRAIN.add(MaxPooling2D(pool_size=3 * 3, strides=(2, 2), padding='same'))

RN2TRAIN.add(Flatten()) #on transforme un tableau 3D en 1D.
RN2TRAIN.add(Dense(2, activation='softmax'))


RN2TRAIN.load_weights('model.h5')

myCNN=Sequential()
myCNN.add(vgg16)
myCNN.add(RN2TRAIN)


predictions = myCNN.predict(X)
print(np.argmax(predictions, axis=1))
print(Y)