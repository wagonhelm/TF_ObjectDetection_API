# Adapted from
# https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('data_path', '', 'where to save train.record')
flags.DEFINE_string('images_path', '', 'path to resized images folder')
flags.DEFINE_string('csv_path', '', 'path to csv bounding boxes')


FLAGS = flags.FLAGS

# Add more class labels as needed, make sure to start at 1
def class_text_to_int(row_label):
  if row_label == 'rat':
    return 1
#  if row_label == 'sphere':
#    return 2 
#  if row_label == 'box':
#    return 3 
  else:
    None

def split(df, group):
  data = namedtuple('data', ['filename', 'object'])
  gb = df.groupby(group)
  return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
      encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
      xmins.append(row['xmin'] / width)
      xmaxs.append(row['xmax'] / width)
      ymins.append(row['ymin'] / height)
      ymaxs.append(row['ymax'] / height)
      classes_text.append(row['class'].encode('utf8'))
      classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
#    for i in ['test', 'train']:
  writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_path,'rat_train.record'))
  path = FLAGS.images_path
  examples = pd.read_csv(FLAGS.csv_path)
  grouped = split(examples, 'filename')
  for group in grouped:
    tf_example = create_tf_example(group, path)
    writer.write(tf_example.SerializeToString())
  writer.close()
  print('Successfully created the train TFRecords')


if __name__ == '__main__':
  tf.app.run()
