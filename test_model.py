from __future__ import print_function

import os
import re

import numpy as np

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.preprocessing import image

def load_images():	

	path="path/to/CNN_keras_Carabidae_pronotum/TestImg/"
	
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
		
		
	return (x_test, y_test)

num_classes=8

x_test, y_test = load_images()
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)
y_test_refs=y_test.copy() #true labels
y_test = keras.utils.to_categorical(y_test, num_classes)

model = load_model("modelKeras.h5")

score = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

prediction=model.predict_classes(x_test,verbose=1)

f=open("predictions.txt","w")
f.write("true label\tprediction\n")
for i in range(len(prediction)):
	f.write(str(y_test_refs[i]))
	f.write("\t"+str(prediction[i]))
	f.write("\n")
f.close()
