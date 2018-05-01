import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body


# Filtering with a threshold on class scores

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
  """Filters YOLO boxes by thresholding on object and class confidence.

  Arguments:
  box_confidence -- tensor of shape (19, 19, 5, 1)
  boxes -- tensor of shape (19, 19, 5, 4)
  box_class_probs -- tensor of shape (19, 19, 5, 80)
  threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box

  Returns:
  scores -- tensor of shape (None,), containing the class probability score for selected boxes
  boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
  classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

  Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold.
  For example, the actual output size of scores would be (10,) if there are 10 boxes.
  """

  scores = box_confidence * box_class_probs
  classes = K.argmax(scores, axis=-1)
  mx_scores = K.max(scores, axis=-1)
  mask = (mx_scores > threshold)

  scores = tf.boolean_mask(mx_scores, mask)
  boxes = tf.boolean_mask(boxes, mask)
  classes = tf.boolean_mask(classes, mask)
  return scores, boxes, classes



# check point
with tf.Session() as test_a:
  box_confidence = tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed=1)
  boxes = tf.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed=1)
  box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed=1)
  scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.5)
  print("scores[2] = " + str(scores[2].eval()))
  print("boxes[2] = " + str(boxes[2].eval()))
  print("classes[2] = " + str(classes[2].eval()))
  print("scores.shape = " + str(scores.shape))
  print("boxes.shape = " + str(boxes.shape))
  print("classes.shape = " + str(classes.shape))


###############################################################################################################################


# Non-max suppression

def iou(box1, box2):
  """Implement the intersection over union (IoU) between box1 and box2

  Arguments:
  box1 -- first box, list object with coordinates (x1, y1, x2, y2)
  box2 -- second box, list object with coordinates (x1, y1, x2, y2)
  """
  xi1 = max(box1[0], box2[0])
  yi1 = max(box1[1], box2[1])
  xi2 = min(box1[2], box2[2])
  yi2 = min(box1[3], box2[3])

  intersection = (xi2 - xi1) * (yi2 - yi1)
  union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection

  return 1. * intersection / union


'''
# check point
box1 = (2, 1, 4, 3)
box2 = (1, 2, 3, 4)
print("iou = " + str(iou(box1, box2)))
'''


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
  """
  Applies Non-max suppression (NMS) to set of boxes

  Arguments:
  scores -- tensor of shape (None,), output of yolo_filter_boxes()
  boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
  classes -- tensor of shape (None,), output of yolo_filter_boxes()
  max_boxes -- integer, maximum number of predicted boxes you'd like
  iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

  Returns:
  scores -- tensor of shape (, None), predicted score for each box
  boxes -- tensor of shape (4, None), predicted box coordinates
  classes -- tensor of shape (, None), predicted class for each box

  Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
  function will transpose the shapes of scores, boxes, classes. This is made for convenience.
  """

  max_boxes_tensor = K.variable(max_boxes, dtype='int32')
  K.get_session().run(tf.variables_initializer([max_boxes_tensor]))

  indices = tf.image.non_max_suppression(scores=scores, boxes=boxes, max_output_size=max_boxes_tensor, iou_threshold=iou_threshold)

  scores = K.gather(scores, indices)
  boxes = K.gather(boxes, indices)
  classes = K.gather(classes, indices)

  return scores, boxes, classes



# check point
with tf.Session() as test_b:
  scores = tf.random_normal([54, ], mean=1, stddev=4, seed=1)
  boxes = tf.random_normal([54, 4], mean=1, stddev=4, seed=1)
  classes = tf.random_normal([54, ], mean=1, stddev=4, seed=1)
  scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
  print("scores[2] = " + str(scores[2].eval()))
  print("boxes[2] = " + str(boxes[2].eval()))
  print("classes[2] = " + str(classes[2].eval()))
  print("scores.shape = " + str(scores.eval().shape))
  print("boxes.shape = " + str(boxes.eval().shape))
  print("classes.shape = " + str(classes.eval().shape))


def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
  """
  Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.

  Arguments:
  yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                  box_confidence: tensor of shape (None, 19, 19, 5, 1)
                  box_xy: tensor of shape (None, 19, 19, 5, 2)
                  box_wh: tensor of shape (None, 19, 19, 5, 2)
                  box_class_probs: tensor of shape (None, 19, 19, 5, 80)
  image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
  max_boxes -- integer, maximum number of predicted boxes you'd like
  score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
  iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

  Returns:
  scores -- tensor of shape (None, ), predicted score for each box
  boxes -- tensor of shape (None, 4), predicted box coordinates
  classes -- tensor of shape (None,), predicted class for each box
  """

  box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
  boxes = yolo_boxes_to_corners(box_xy, box_wh)
  scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)
  boxes = scale_boxes(boxes, image_shape)
  scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
  return scores, boxes, classes



# check point
with tf.Session() as test_b:
  yolo_outputs = (tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed=1),
                  tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
                  tf.random_normal([19, 19, 5, 2], mean=1, stddev=4, seed=1),
                  tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed=1))
  scores, boxes, classes = yolo_eval(yolo_outputs)
  print("scores[2] = " + str(scores[2].eval()))
  print("boxes[2] = " + str(boxes[2].eval()))
  print("classes[2] = " + str(classes[2].eval()))
  print("scores.shape = " + str(scores.eval().shape))
  print("boxes.shape = " + str(boxes.eval().shape))
  print("classes.shape = " + str(classes.eval().shape))
