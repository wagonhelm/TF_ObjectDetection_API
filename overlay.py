#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 15:39:18 2017

@author: atabak
"""


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import cv2  
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

      width=1920
      height=1080
      #NEW

      font = cv2.FONT_HERSHEY_PLAIN
      fontsize=2
      fontweight=2
      #img = cv2.imread(full_path,cv2.IMREAD_COLOR)
      classes_text= np.squeeze(classes).astype(np.int32)
      #print(classes_text)
      for i in xrange(0,len(np.squeeze(boxes))):
	bbox=np.squeeze(boxes)[i]
        xmin=bbox[1]   
        ymin=bbox[0]
        xmax=bbox[3]
        ymax=bbox[2]   
        score=str("%.2f"%np.squeeze(scores)[i])
        if(np.squeeze(scores)[i]<=0.75) :
          continue

        if classes_text[i]==1:
          class_='cylinder'
          color=(0,255,0)
        elif classes_text[i]==2:
          class_='sphere'
          color=(255,0,0)
        elif classes_text[i]==3:
          class_='box'
          color=(0,0,255)
        cv2.rectangle(image_np,(int(xmin*width),int(ymin*height)),(int(xmax*width),int(ymax*height)),color,2)
        cv2.putText(image_np, class_+"("+score+")", (int(xmin*width), int(ymin*height-4)), font, fontsize, color, fontweight, cv2.LINE_AA)
      #cv2.imshow('image',image_np)
      #cv2.waitKey(0)
      #cv2.destroyAllWindows()
      cv2.imwrite(os.path.join(FLAGS.save_path,image_path).replace('test','new_test'),image_np)


      #ENDNEW


      #plt.savefig(os.path.join(FLAGS.save_path,image_path), format='png', bbox_inches='tight')


if __name__ == '__main__':
  tf.app.run()
