# CNN_keras_Carabidae_pronotum

Our dataset is composed of 293 RGB images of carabidae’s pronotums (see prono.jpg).
For each image, 8 landmarks were manually placed, and their coordinates are noted in a separate file (with the y axis inverted) (see landmark\_positions.png)
We aim to use a deep learning approach to locate automatically the landmarks in the images. To do this, we began by a simple classification task, to see if our landmarks could be easily separated into 8 different classes by a neural network.

The code providing the architecture of the CNN as well as the training, validation and test phases on the whole dataset is given in cnn\_keras.py. It was run using TensorFlow as backend and "channels first" as image data format, which must be specified in the keras.json file. 
The trained model is saved in the file modelKeras.h5 and can be loaded using the file test\_model.py. This script runs the model on the sample images (in TestImg directory) and outputs the result of the model predictions in a new text file.
