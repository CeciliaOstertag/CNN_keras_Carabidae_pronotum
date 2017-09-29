from __future__ import print_function

import sys
import os
import time
import re

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.preprocessing import image

def load_images():
	
	path="/home/fou/Desktop/Norm/train/"
	
	matrixs=[]
	y=[]
	for filename in os.listdir(path):
		label_search=re.search(r"p_\d{3}.TPS_(\d).jpg",filename)
		if label_search:
			label=label_search.group(1)
			label=int(label)
			y.append(label)
			img=image.load_img(path+filename)
			array=image.img_to_array(img)
			matrixs.append(array)
	x_train=np.array([matrix for matrix in matrixs])
	y_train=np.array([label for label in y])

	path="/home/fou/Desktop/Norm/val/"
	
	matrixs=[]
	y=[]
	for filename in os.listdir(path):
		label_search=re.search(r"p_\d{3}.TPS_(\d).jpg",filename)
		if label_search:
			label=label_search.group(1)
			label=int(label)
			y.append(label)
			img=image.load_img(path+filename)
			array=image.img_to_array(img)
			matrixs.append(array)

	x_val=np.array([matrix for matrix in matrixs])
	y_val=np.array([label for label in y])	


	path="/home/fou/Desktop/Norm/test/"
	
	matrixs=[]
	y=[]
	for filename in os.listdir(path):
		label_search=re.search(r"p_\d{3}.TPS_(\d).jpg",filename)
		if label_search:
			label=label_search.group(1)
			label=int(label)
			y.append(label)
			img=image.load_img(path+filename)
			array=image.img_to_array(img)
			matrixs.append(array)	

	x_test=np.array([matrix for matrix in matrixs])
	y_test=np.array([label for label in y])
		
		
	return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# main program


num_classes=8
rows,cols=63,63


(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_images()

input_shape=(3,rows,cols)

#normalization
#x_train /= 255
#x_val /= 255

#mean substraction
#avg=np.mean(x_train, axis=0)
#x_train -= avg
#x_val -= avg

print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'val samples')
print(x_test.shape[0], 'test samples')
print('x_train shape:', x_train.shape)
print('x_val shape:', x_val.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_val shape:', y_val.shape)
print('y_test shape:', y_test.shape)

y_test_refs=y_test.copy()

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()

# Convolutional layer with 32 kernels of size 5x5. 
model.add(Conv2D(filters=32, kernel_size=(5,5), strides=1, padding='same',data_format='channels_first', activation='relu', kernel_initializer= keras.initializers.RandomNormal(mean=0.0001, stddev=0, seed=None),bias_initializer=keras.initializers.Constant(value=0),kernel_regularizer=keras.regularizers.l2(0.0005),input_shape=input_shape)) #use data_format : channels_first in keras.json
print("shape conv1 : ",model.output_shape) 
    
#Pooling layer
model.add(MaxPooling2D(pool_size=2, strides=2, padding='valid'))
print("shape pool1 : ",model.output_shape)
    
#Conv layer with 64 kernels of size 5x5
model.add(Conv2D(filters=64, kernel_size=(5,5), strides=1, padding='same', activation='relu', kernel_initializer= keras.initializers.RandomNormal(mean=0.01, stddev=0, seed=None),bias_initializer=keras.initializers.Constant(value=0),kernel_regularizer=keras.regularizers.l2(0.0005))) 
print("shape conv2 : ",model.output_shape)
    
#Pooling layer
model.add(AveragePooling2D(pool_size=3, strides=2, padding='valid'))
print("shape pool2 : ",model.output_shape)
    
#Conv layer with 32 kernels of size 3x3
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='same', activation='relu', kernel_initializer= keras.initializers.RandomNormal(mean=0.01, stddev=0, seed=None),bias_initializer=keras.initializers.Constant(value=0),kernel_regularizer=keras.regularizers.l2(0.0005)))
print("shape conv3 : ",model.output_shape) 
    
#Pooling layer
model.add(AveragePooling2D(pool_size=3, strides=2, padding='valid'))
print("shape pool3 : ",model.output_shape)

#Flatten layer (not done by default like with Lasagne)
model.add(Flatten())
    
#Fully connected layer
model.add(Dense(units=32, activation='relu', kernel_initializer=keras.initializers.RandomNormal(mean=0.001, stddev=0, seed=None),kernel_regularizer=keras.regularizers.l2(0.0005)))
print("shape fc1 : ",model.output_shape)
    
#Dropout layer
model.add(Dropout(rate=0.3, noise_shape=None, seed=None))
print("shape dropout1 : ",model.output_shape)
    
#Fully connected layer
model.add(Dense(units=num_classes, activation='softmax', kernel_initializer=keras.initializers.RandomNormal(mean=0.1, stddev=0, seed=None),kernel_regularizer=keras.regularizers.l2(0.0005)))
print("shape fc2 : ",model.output_shape)

lr=0.000002
decay=0.00001
sgd=keras.optimizers.SGD(lr=lr,momentum=0.9,decay=decay, nesterov=False)
rms=keras.optimizers.RMSprop()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'])

#best parameters : lr=0.000002,momentum=0.9,decay=0.00001
start_time = time.time()
model.fit(x_train, y_train,
          batch_size=16,
          epochs=500,
          verbose=1,
          validation_data=(x_val, y_val))
score = model.evaluate(x_val, y_val, verbose=1)

print('Val loss:', score[0])
print('Val accuracy:', score[1])

score2 = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', score2[0])
print('Test accuracy:', score2[1])


model.save("modelKeras.h5") #open with load_model

stop_time = time.time()
elapsed_time=stop_time-start_time



