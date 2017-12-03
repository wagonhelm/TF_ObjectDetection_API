#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 15:39:18 2017

@author: atabak
"""


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt

from object_detection.utils import label_map_util, visualization_utils

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('images_path', '', 'path to resized test images folder')
flags.DEFINE_string('save_path', '', 'path to save infered bounding boxes')


FLAGS = flags.FLAGS

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def main(_):
#    for i in ['test', 'train']:
  path_to_ckpt = 'object_detection_graph/frozen_inference_graph.pb'
  path_to_labels = 'data/label_map.pbtxt'
  num_classes = 3
  path_to_test_images_dir= FLAGS.images_path
  test_images_path = os.listdir(path_to_test_images_dir)
  img_size = (12,12)
  
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
  
  label_map = label_map_util.load_labelmap(path_to_labels)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)
  with tf.Session(graph=detection_graph) as sess:
    for image_path in test_images_path:
      image = Image.open(os.path.join(path_to_test_images_dir,image_path))
      image_np = load_image_into_numpy_array(image)
      image_np_expanded = np.expand_dims(image_np, axis=0)
      (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
      visualization_utils.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=1)
      plt.figure(figsize=img_size)
      plt.imshow(image_np)
      plt.savefig(os.path.join(FLAGS.save_path,image_path), format='png', bbox_inches='tight')


if __name__ == '__main__':
  tf.app.run()
