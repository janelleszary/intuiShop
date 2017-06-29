'''
Builds a keras model based on the VGG-16 architecture and loads in pretrained weights, trained on ImageNet.
Currently configured to use TensorFlow as backend, running on GPU (this might need modifications to run on CPU).

Requires:
	-Should be run from a directory which contains a subdirectory called "images2" with the following structure:
	/Image2
		/category1
			item1.jpg
			item2.jpg
		/category2
			item1.jpg
			item2.jpg
			...
			(up to 1k from each category)

Returns:
	-A .npy numpy array file for each image, with the same name and in the same location as the original images.

PENDING: Combine numpy arrays into master list.

'''
import h5py
import numpy as np
import os
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications


from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense, Input, Dropout
from keras.layers import Convolution2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
from keras.engine.topology import get_source_inputs


def get_base_model():
    """
    Returns the convolutional part of VGG net as a keras model 
    All layers have trainable set to False
    """
    img_width, img_height = 224, 224
    
    # Add convolutional layers 
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_width, img_height, 3), name='image_input'))
    model.add(Convolution2D(64, (3, 3), activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    
    # Add feed-forward layers
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dense(4096, activation='relu', name='fc2'))
    
    # Add output layer (softmax classifier)
    model.add(Dense(1000, activation='softmax', name='predictions'))
    
    # set trainable to false in all layers 
    for layer in model.layers:
        if hasattr(layer, 'trainable'):
            layer.trainable = False
    return model

# Function to load pretrained weights from the internet and apply them to the model
def load_weights_in_base_model(model):
	WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
	weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5', WEIGHTS_PATH, cache_subdir='models')
	model.load_weights(weights_path)
	return model

# Function to force an image into size of VGG input
def processanimage(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return(x)
 
# Instantiate model and load weights.
mymodel = get_base_model()
mymodel = load_weights_in_base_model(mymodel)

# Predicting a cat, just for fun!
#	img = processanimage('cat.jpg')
#	output = mymodel.predict(img)

# Remove the softmax classifier from the end and set up the new final layer to serve as output.
mymodel.layers.pop()
mymodel.outputs = [mymodel.layers[-1].output]
mymodel.layers[-1].outbound_nodes = []


homedir = (os.getcwd())
os.chdir('images2')
imdir = (os.getcwd())
dirs = [f for f in os.listdir('.') if os.path.isdir(f)]

# Cycle through images in all subdirectories and create npy files for each
for dir in dirs:
	print('Starting '+dir)
	os.chdir(dir)
	ims = [f for f in os.listdir('.') if os.path.isfile(f)]
	random.shuffle(ims)
	totaltaken=0
	for im in ims:
		if ((im[-4:]=='.jpg') and totaltaken<1000):
			totaltaken+=1
			mynpy = mymodel.predict(processanimage(im))
			np.save(open(im[0:-4]+'.npy', 'wb'), mynpy)
	os.chdir(imdir)
