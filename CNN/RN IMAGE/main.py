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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout


from PIL import Image, ImageChops, ImageEnhance
import os


#Troisième test de donnée, https://www.kaggle.com/shaft49/real-vs-fake-images-casia-dataset



X = []
Y = []  # 0 for fake, 1 for real



import random

path = r"C:\\Users\\Yann\\Desktop\\dataset\\CASIA2.0_revised\\Au"
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('tif'):
            full_path = os.path.join(dirname, filename)
            X.append(Image.open(full_path).convert('RGB'))
            Y.append(1)
            if len(Y) % 500 == 0:
                print(f'Processing {len(Y)} images')

random.shuffle(X)
X = X[:2100]
Y = Y[:2100]
print(len(X), len(Y))

path = 'C:\\Users\\Yann\\Desktop\\dataset\\CASIA2.0_revised\\Tp'
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('tif'):
            full_path = os.path.join(dirname, filename)
            X.append(Image.open(full_path).convert('RGB'))
            Y.append(0)
            if len(Y) % 500 == 0:
                print(f'Processing {len(Y)} images')

print(len(X), len(Y))

X = np.array(X)
Y = to_categorical(Y, 2)

for i in range(0,X.size):
    X[i].resize((227,227))
    X[i] = numpy.asarray(X[i])


X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=5)
X = X.reshape(-1,1,1,1)
print(len(X_train), len(Y_train))
print(len(X_val), len(Y_val))

#fin de l'inspiration




'''

(ds_train,ds_test),ds_info = tds.load('mnist',split=['train','test'],shuffle_files=True,as_supervised=True,with_info=True)

def normalize_img(image,label):
    return tf.cast(image,tf.float32)/255., label


ds_train = ds_train.map(normalize_img,num_parallel_calls=tf.data.AUTOTUNE)
ds_train=ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train=ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


'''




'''
(x_train, y_train),(x_test, y_test) =mnist

x_train = tk.utils.normalize(x_train,axis=1)
x_test = tk.utils.normalize(x_test, axis=1)

'''

'''
# Load MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data by flattening & scaling it
x_train = x_train.reshape(-1, 127).astype("float32") / 255
x_test = x_test.reshape(-1, 127).astype("float32") / 255

# Categorical (one hot) encoding of the labels
y_train = tk.utils.to_categorical(y_train)
y_test = tk.utils.to_categorical(y_test)

'''



#On importe l'entrainnement de VGG16 pour classifier 1000 types d'image
model = VGG16(weights="imagenet",include_top=False,input_shape=(227,227,3))

for layer in model.layers:
    layer.trainable = False




#On définit les calques de notre model CNN qui prend en entrée
#l'entrainnement de VGG16 et retourne 2 classes.

CNN=Sequential()
#CNN=Model(inputs=model.input)
CNN.add(model)
CNN.add(Conv2D(11*11,(4,4),padding='same',activation='relu'))
CNN.add(MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same'))
CNN.add(Conv2D(5*5,(1,1),padding='valid',activation='relu'))
CNN.add(MaxPooling2D(pool_size=(3*3),strides=(2,2),padding='same'))
CNN.add(Conv2D(3*3,(1,1),padding='valid',activation='relu'))
CNN.add(Conv2D(3*3,(1,1),padding='valid',activation='relu'))
CNN.add(Conv2D(6*6,(1,1),padding='same',activation='relu'))
CNN.add(MaxPooling2D(pool_size=3*3,strides=(2,2),padding='same'))

CNN.add(Flatten()) #on transforme un tableau 3D en 1D.
CNN.add(Dense(2,activation='softmax'))

#model.summary()
CNN.summary()
#on compile CNN


CNN.compile(optimizer='adam',loss='keras.losses.hinge',metrics='accuracy')


#on entraine le model.
#model_info = CNN.fit(train_generator,steps_per_epoch=8,epochs=15,verbose=1,validation_data=validation_generator,validation_steps=8)
#CNN.fit(x_train,y_train,epochs=3)
#CNN.fit(ds_train, epochs=6, validation_data=ds_test)
CNN.fit(X_train,Y_train, epochs=6, validation_data=(X_val, Y_val))



