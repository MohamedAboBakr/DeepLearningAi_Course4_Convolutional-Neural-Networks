import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


# load data and normalize it
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

X_train = X_train_orig / 255
X_test = X_test_orig / 255

Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


def HappyModel(input_shape):
  """
  Implementation of the HappyModel.

  Arguments:
  input_shape -- shape of the images of the dataset

  Returns:
  model -- a Model() instance in Keras
  """

  ### START CODE HERE ###
  # Feel free to use the suggested outline in the text above to get started, and run through the whole
  # exercise (including the later portions of this notebook) once. The come back also try out other
  # network architectures as well.
  # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
  X_input = Input(input_shape)

  # Zero-Padding: pads the border of X_input with zeroes
  X = ZeroPadding2D((3, 3))(X_input)

  # CONV -> BN -> RELU Block applied to X
  X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
  X = BatchNormalization(axis=3, name='bn0')(X)
  X = Activation('relu')(X)

  # MAXPOOL
  X = MaxPooling2D((2, 2), name='max_pool')(X)

  # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
  X = Flatten()(X)
  X = Dense(1, activation='sigmoid', name='fc')(X)

  # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
  model = Model(inputs=X_input, outputs=X, name='HappyModel')

  return model
  ### END CODE HERE ###


# Four steps to train and test models in keras
"""
1- create the model by calling the function above
2- compile the model with specific optimizer --> model.compile(optimizer='', loss='', metrics=['accuracy'])
3- train the model on train data --> model.fit(x=  , y= , epochs=  , batch_size = )
4- test the model on test data --> model.evaluate(x= , y=)
"""


# 1- create the model
happyModel = HappyModel(X_train.shape[1:])

# 2- compile the model
happyModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 3- train the model on train data
happyModel.fit(X_train, Y_train, epochs=40, batch_size=50)

# 4- test the model on test data
result = happyModel.evaluate(X_test, Y_test, batch_size=50, verbose=1)

loss = result[0]
accuracy = result[1]
print(loss, accuracy)


'''
img_path = 'images/my_image.jpg'
img = image.load_img(img_path, target_size=(64, 64))
imshow(img)

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(happyModel.predict(x))
'''