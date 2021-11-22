import keras.optimizer_v1
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.preprocessing.image import load_img,img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras import Model
import tensorflow.keras as tk



mnist = tk.datasets.load('mnist',split=['train','test'],shuffle_files=True,as_supervised=True,with_info=True)

def normalize_im(image,label):
    return tk.cast(image,tk.float32)/255., label


tk.datasets_test=tk.datasets_test.map(normalize_img,num_parallel_calls=tf.data.AUTOTUNE)
tk.datasets_test =tk.datasets_test.batch(128)


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
CNN.fit(x_train, y_train, epochs=20, batch_size=128, validation_split=0.2)

CNN.summary()

