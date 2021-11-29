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


#Troisième test de donnée, https://www.kaggle.com/shaft49/real-vs-fake-images-casia-dataset



X = []
Y = []  # 0 for fake, 1 for real



import random

path = r"C:\\Users\\Yann\\Desktop\\dataset\\CASIA2.0_revised\\Au"
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('tif'):
            full_path = os.path.join(dirname, filename)
            im=np.asarray(Image.open(full_path).convert('RGB').resize((227,227)))
            X.append(im)
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
            im = np.asarray(Image.open(full_path).convert('RGB').resize((227,227)))
            X.append(im)
            Y.append(0)
            if len(Y) % 500 == 0:
                print(f'Processing {len(Y)} images')

print(len(X), len(Y))

X = np.array(X)
Y = to_categorical(Y, 2)
print(f'still not bugging for now')





print(X)

# X = pywt.dwt2(X,'haar') Mon pc meurt, faut pas faire ça
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=5)
#X = X.reshape(-1,1,1,1)
print(len(X_train), len(Y_train))
print(len(X_val), len(Y_val))

#fin de l'inspiration





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


CNN.compile(optimizer='adam', loss=keras.losses.hinge, metrics='accuracy')



tf.convert_to_tensor(X_val.astype(np.float32), dtype=tf.float32)
tf.convert_to_tensor(Y_val.astype(np.float32), dtype=tf.float32)

history = CNN.fit(X_train,Y_train, batch_size=32, epochs=100, validation_data=(X_val, Y_val))

CNN.save_weights('CNN2.h5')
#model.load_weights('CNN.h5')


# Plot the loss and accuracy curves for training and validation  // CODE VOLE SUBTILEMENT
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)

'''
def plot_confusion_matrix(cm, classes,   #code volé subtilement aussi
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

'''

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors
Y_pred_classes = np.argmax(Y_pred, axis=1)
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val, axis=1)
# compute the confusion matrix
#confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
# plot the confusion matrix
#plot_confusion_matrix(confusion_mtx, classes=range(2))


#voir le guide https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/   pour sauvegarder le RN