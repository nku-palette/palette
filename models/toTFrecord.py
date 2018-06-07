# coding: utf-8
"""
Convert sample images to tfrecord for tensorflow training. 
TFrecord 依旧使用feed_dict的方式来训练的吗？还需要看一下Dataset类。
"""

import tensorflow as tf
import numpy as np
import cv2
import os
import os.path
from PIL import Image

# param
train_file = 'train.txt' 
name='train'
output_directory='./tfrecords'  
resize_height=224
resize_width=224

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_file(samples_dir):
	with open(samples_dir, 'r') as f:

	return samples, labels, boxes, name, nums

def extract_image(imagefile, boxfile, resize_height, resize_width):
    image = cv2.imread(imagefile)
    image = cv2.resize(image, (resize_height, resize_width))
    b,g,r = cv2.split(image)
    rgb_image = cv2.merge([r,g,b])
    # TODO!!! Resize 标注框
    # TODO!!! 减均值
    return rgb_image

def transform2tfrecord(train_file, name, output_directory, resize_height, resize_width):
	"""
	Transform image to tfrecord.
	"""
    if not os.path.exists(output_directory) or os.path.isfile(output_directory):
        os.makedirs(output_directory)
    _examples, _labels, _boxes, _name, examples_num = load_file(train_file)
    filename = output_directory + "/" + name + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    for i, [example, label] in enumerate(zip(_examples, _labels)):
        print('No.%d' % (i))
        image, box = extract_image(example, _boxes[i], resize_height, resize_width)
        print('shape: %d, %d, %d, label: %d' % (image.shape[0], image.shape[1], image.shape[2], label))
        image_raw = image.tostring()
        box_raw = bos.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'depth': _int64_feature(image.shape[2]),
            'name': _bytes_feature(_name[i]),
            'bounding_box_raw': _bytes_feature(box_raw),
            'label': _int64_feature(label)
        }))  
        writer.write(example.SerializeToString())
    writer.close()
    return filename 

def disp_tfrecords(tfrecord_list_file):
	"""
    Display tfrecord.
	"""
    filename_queue = tf.train.string_input_producer([tfrecord_list_file]) # func includes random shuffle
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
 features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'depth': tf.FixedLenFeature([], tf.int64),
          'name': tf.FixedLenFeature([], tf.string),
          'bounding_box_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64)
      }
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    #print(repr(image))  
    height = features['height']
    width = features['width']
    depth = features['depth']
    label = tf.cast(features['label'], tf.int32)
    init_op = tf.initialize_all_variables()
    resultImg=[]
    resultLabel=[]
    with tf.Session() as sess:
        sess.run(init_op)
        # 多线程管理器
        coord = tf.train.Coordinator()
        # 启动tensor的入队线程
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(21):
	        # tensor.eval() = tf.get_default_session().run(t)
            image_eval = image.eval()
            resultLabel.append(label.eval())
            image_eval_reshape = image_eval.reshape([height.eval(), width.eval(), depth.eval()])
            resultImg.append(image_eval_reshape)
            pilimg = Image.fromarray(np.asarray(image_eval_reshape))
            pilimg.show()
        coord.request_stop()
        coord.join(threads)
        sess.close()
    return resultImg, resultLabel

def read_tfrecord(filename_queuetemp):
	"""
	Read tfrecord.
	"""
    filename_queue = tf.train.string_input_producer([filename_queuetemp])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'width': tf.FixedLenFeature([], tf.int64),
          'depth': tf.FixedLenFeature([], tf.int64),
          'label': tf.FixedLenFeature([], tf.int64)
      }
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    # image  
    tf.reshape(image, [224, 224, 3])
    # normalize
    # image = tf.cast(image, tf.float32) * (1. /255) - 0.5
    # label  
    label = tf.cast(features['label'], tf.int32)
    return image, label

def test():
    filename = transform2tfrecord(train_file, name , output_directory,  resize_height, resize_width)
    img,label=disp_tfrecords(filename)
    img,label=read_tfrecord(filename)
    print label

if __name__ == '__main__':
    test()