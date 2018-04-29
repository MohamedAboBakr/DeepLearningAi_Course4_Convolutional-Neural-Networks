import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, MaxPooling2D, AveragePooling2D, BatchNormalization, Conv2D, Flatten, ZeroPadding2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform

import pydot
from IPython.display import SVG
from resnets_utils import *
import scipy.misc
from matplotlib.pyplot import imshow

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


# The identity block
# when a[l] has the same dimension as a[l+2]
def identity_block(X, f, filters, stage, block):
  """
  Implementation of the identity block as defined in Figure 3

  Arguments:
  X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
  f -- integer, specifying the shape of the middle CONV's window for the main path
  filters -- python list of integers, defining the number of filters in the CONV layers of the main path
  stage -- integer, used to name the layers, depending on their position in the network
  block -- string/character, used to name the layers, depending on their position in the network

  Returns:
  X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
  """

  # defining name basis
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  # Retrieve Filters
  F1, F2, F3 = filters

  X_shortcut = X

  # First component of main path
  X = Conv2D(F1, (1, 1), strides=(1, 1), name=conv_name_base + '2a', padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
  X = Activation('relu')(X)

  # Second component of main path
  X = Conv2D(F2, (f, f), strides=(1, 1), name=conv_name_base + '2b', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
  X = Activation('relu')(X)

  # Third component of main path

  X = Conv2D(F3, (1, 1), strides=(1, 1), name=conv_name_base + '2c', padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
  # Add the input before activation function
  X = Add()([X, X_shortcut])
  X = Activation('relu')(X)

  return X


'''
# check point
tf.reset_default_graph()

with tf.Session() as test:
  np.random.seed(1)
  A_prev = tf.placeholder("float", [3, 4, 4, 6])
  X = np.random.randn(3, 4, 4, 6)
  A = identity_block(A_prev, f=2, filters=[2, 4, 6], stage=1, block='a')
  test.run(tf.global_variables_initializer())
  out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
  print("out = " + str(out[0][1][1][0]))
'''

######################################################################################################################

# The convolutional block
# when a[l] and a[l+2] dimensions don't match up
# The difference with the identity block is that there is a CONV2D layer in the shortcut path


def convolutional_block(X, f, filters, stage, block, s=2):
  """
  Implementation of the convolutional block as defined in Figure 4

  Arguments:
  X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
  f -- integer, specifying the shape of the middle CONV's window for the main path
  filters -- python list of integers, defining the number of filters in the CONV layers of the main path
  stage -- integer, used to name the layers, depending on their position in the network
  block -- string/character, used to name the layers, depending on their position in the network
  s -- Integer, specifying the stride to be used

  Returns:
  X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
  """

  # defining name basis
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  # Retrieve Filters
  F1, F2, F3 = filters

  # Save the input value
  X_shortcut = X

  # First component in main path
  X = Conv2D(F1, (1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
  X = Activation('relu')(X)

  # Second component in main path
  X = Conv2D(F2, (f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
  X = Activation('relu')(X)

  # Third component in main path
  X = Conv2D(F3, (1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

  # Add CONV->BN component to shortcut path
  X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
  X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

  # Add two branches before the last RELU
  X = Add()([X, X_shortcut])
  X = Activation('relu')(X)

  return X


'''
# check point
tf.reset_default_graph()

with tf.Session() as test:
  np.random.seed(1)
  A_prev = tf.placeholder("float", [3, 4, 4, 6])
  X = np.random.randn(3, 4, 4, 6)
  A = convolutional_block(A_prev, f=2, filters=[2, 4, 6], stage=1, block='a')
  test.run(tf.global_variables_initializer())
  out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
  print("out = " + str(out[0][1][1][0]))
'''

################################################################################################################################


# Building ResNet model (50 layers)
def ResNet50(input_shape=(64, 64, 3), classes=6):
  """
  Implementation of the popular ResNet50 the following architecture:
  CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
  -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

  Arguments:
  input_shape -- shape of the images of the dataset
  classes -- integer, number of classes

  Returns:
  model -- a Model() instance in Keras
  """
  X_input = Input(input_shape)
  X = ZeroPadding2D((3, 3))(X_input)

  # Stage 1
  X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis=3, name='bn1')(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((3, 3), strides=(2, 2), name='MP1')(X)

  # Stage 2
  X = convolutional_block(X, 3, [64, 64, 256], 2, 'a', 1)
  X = identity_block(X, 3, [64, 64, 256], 2, 'b')
  X = identity_block(X, 3, [64, 64, 256], 2, 'c')

  # Stage 3
  X = convolutional_block(X, 3, [128, 128, 512], 3, 'a', 2)
  X = identity_block(X, 3, [128, 128, 512], 3, 'b')
  X = identity_block(X, 3, [128, 128, 512], 3, 'c')
  X = identity_block(X, 3, [128, 128, 512], 3, 'd')

  # Stage 4
  X = convolutional_block(X, 3, [256, 256, 1024], 4, 'a', 2)
  X = identity_block(X, 3, [256, 256, 1024], 4, 'b')
  X = identity_block(X, 3, [256, 256, 1024], 4, 'c')
  X = identity_block(X, 3, [256, 256, 1024], 4, 'd')
  X = identity_block(X, 3, [256, 256, 1024], 4, 'e')
  X = identity_block(X, 3, [256, 256, 1024], 4, 'f')

  # Stage 5
  X = convolutional_block(X, 3, [512, 512, 2048], 5, 'a', 2)
  X = identity_block(X, 3, [256, 256, 2048], 5, 'b')
  X = identity_block(X, 3, [256, 256, 2048], 5, 'c')

  # AvgPool
  X = AveragePooling2D((2, 2), name='avg_pool', padding='same')(X)
  X = Flatten()(X)
  X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

  # Create model
  model = Model(inputs=X_input, outputs=X, name='ResNet50')

  return model


################################################################################################################################

# load dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# Normalize train & test sets
X_train = X_train_orig / 255
X_test = X_test_orig / 255

# one-hot encoding on labels
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

#  4 steps
"""
1- create the model by calling the function above
2- compile the model with specific optimizer --> model.compile(optimizer='', loss='', metrics=['accuracy'])
3- train the model on train data --> model.fit(x=  , y= , epochs=  , batch_size = )
4- test the model on test data --> model.evaluate(x= , y=)
"""

model = ResNet50(input_shape=(64, 64, 3), classes=6)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=2, batch_size=32)
result = model.evaluate(X_test, Y_test)
print("Loss = " + str(result[0]))
print("Test Accuracy = " + str(result[1]))
