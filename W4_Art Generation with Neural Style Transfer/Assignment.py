import os
import sys
import scipy
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf

'''
content_image = scipy.misc.imread("images/louvre.jpg")
imshow(content_image)
plt.show()
'''


def compute_content_cost(n_C, n_G):
  """
  Computes the content cost

  Arguments:
  a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
  a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

  Returns:
  J_content -- scalar that you compute using equation 1 above.
  """
  m, n_h, n_w, n_c = n_G.get_shape().as_list()
  n_C = tf.reshape(n_C, [m, n_h * n_w, n_c])
  n_G = tf.reshape(n_G, [m, n_h * n_w, n_c])
  J_content = tf.reduce_sum(tf.squared_difference(n_C, n_G)) / (4 * n_h * n_w * n_c)
  return J_content


# check
'''
tf.reset_default_graph()
with tf.Session() as test:
  tf.set_random_seed(1)
  a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
  a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
  J_content = compute_content_cost(a_C, a_G)
  print("J_content = " + str(J_content.eval()))
'''

'''
style_image = scipy.misc.imread("images/monet_800600.jpg")
imshow(style_image)
plt.show()
'''


def gram_matrix(A):
  """
  Argument:
  A -- matrix of shape (n_C, n_H*n_W)

  Returns:
  GA -- Gram matrix of A, of shape (n_C, n_C)
  """

  A = tf.matmul(A, A, transpose_b=True)
  return A


# check
'''
tf.reset_default_graph()
with tf.Session() as test:
  tf.set_random_seed(1)
  A = tf.random_normal([3, 2 * 1], mean=1, stddev=4)
  GA = gram_matrix(A)

  print("GA = " + str(GA.eval()))
'''


def compute_layer_style_cost(a_S, a_G):
  """
  Arguments:
  a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
  a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

  Returns:
  J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
  """
  m, n_h, n_w, n_c = a_G.get_shape().as_list()
  a_S = tf.transpose(tf.reshape(a_S, [m, n_h * n_w, n_c]), perm=[0, 2, 1])
  a_G = tf.transpose(tf.reshape(a_G, [m, n_h * n_w, n_c]), perm=[0, 2, 1])

  S_gmat = gram_matrix(a_S)
  G_gmat = gram_matrix(a_G)

  J_style_layer = tf.reduce_sum(tf.squared_difference(S_gmat, G_gmat)) / (2 * n_h * n_w * n_c)**2
  return J_style_layer



# check
'''
tf.reset_default_graph()
with tf.Session() as test:
  tf.set_random_seed(1)
  a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
  a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
  J_style_layer = compute_layer_style_cost(a_S, a_G)

  print("J_style_layer = " + str(J_style_layer.eval()))
'''

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


def compute_style_cost(model, STYLE_LAYERS):
  """
  Computes the overall style cost from several chosen layers

  Arguments:
  model -- our tensorflow model
  STYLE_LAYERS -- A python list containing:
                      - the names of the layers we would like to extract style from
                      - a coefficient for each of them

  Returns:
  J_style -- tensor representing a scalar value, style cost defined above by equation (2)
  """
  J_style = 0.0
  for layer, lambdaa in STYLE_LAYERS:
    layer_weights = model[layer]
    a_S = sess.run(layer_weights)
    a_G = layer_weights     # a_G is a tensor and hasn't been evaluated yet

    J_style += lambdaa * compute_layer_style_cost(a_S, a_G)

  return J_style


def total_cost(J_content, J_style, alpha=10, beta=40):
  """
  Computes the total cost function

  Arguments:
  J_content -- content cost coded above
  J_style -- style cost coded above
  alpha -- hyperparameter weighting the importance of the content cost
  beta -- hyperparameter weighting the importance of the style cost

  Returns:
  J -- total cost as defined by the formula above.
  """
  J = alpha * J_content + beta * J_style
  return J


# check
'''
tf.reset_default_graph()
with tf.Session() as test:
    np.random.seed(3)
    J_content = np.random.randn()
    J_style = np.random.randn()
    J = total_cost(J_content, J_style)
    print("J = " + str(J))
'''

###################################################################################################################################

# Solving the optimization problem
# steps:
'''


    1- Create an Interactive Session
    2- Load the content image
    3- Load the style image
    4- Randomly initialize the image to be generated
    5- Load the VGG16 model
    6- Build the TensorFlow graph:
        * Run the content image through the VGG16 model and compute the content cost
        * Run the style image through the VGG16 model and compute the style cost
        * Compute the total cost
        * Define the optimizer and the learning rate
    7- Initialize the TensorFlow graph and run it for a large number of iterations, updating the generated image at every step.


'''


# 1- Create an Interactive Session
tf.reset_default_graph()
sess = tf.InteractiveSession()

# 2- load content image
content_image = scipy.misc.imread("images/louvre_small.jpg")
content_image = reshape_and_normalize_image(content_image)

# 3- load style image
style_image = scipy.misc.imread("images/monet.jpg")
style_image = reshape_and_normalize_image(style_image)

# 4- Randomly initialize the image to be generated
generated_image = generate_noise_image(content_image)


# 5- Load the VGG16 model
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")


# 6a - Run the content image through the VGG16 model and compute the content cost
sess.run(model['input'].assign(content_image))
layer_activations = model['conv4_2']
a_C = sess.run(layer_activations)

a_G = layer_activations    # tensor of activations , hasn't been evaluated yet
J_content = compute_content_cost(a_C, a_G)


# 6b- Run the style image through the VGG16 model and compute the style cost
sess.run(model['input'].assign(style_image))
J_style = compute_style_cost(model, STYLE_LAYERS)

# 6c - Compute the total cost
J = total_cost(J_content, J_style, alpha=10, beta=40)

# 6d - Define the optimizer and the learning rate
optimizer = tf.train.AdamOptimizer(2.0)
train_step = optimizer.minimize(J)


# 7- Initialize the TensorFlow graph and run it
def model_nn(sess, input_image, num_itr=200):
  sess.run(tf.global_variables_initializer())
  sess.run(model['input'].assign(input_image))
  for i in range(num_itr):
    sess.run(train_step)
    generated_image = sess.run(model['input'])
    if i % 20 == 0:
      Jt, Jc, Js = sess.run([J, J_content, J_style])
      print("Iteration " + str(i) + " :")
      print("total cost = " + str(Jt))
      print("content cost = " + str(Jc))
      print("style cost = " + str(Js))
      save_image("output/" + str(i) + ".png", generated_image)
  save_image('output/generated_image.jpg', generated_image)
  return generated_image


model_nn(sess, generated_image)
