# !/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Copyright (c) luoadore, Inc. All Rights Reserved
#
"""
This module provide
    Authors: xiaobing
Date:	2018/4/30
"""
import tensorflow as tf
import numpy as np
from ..tf_dependency import label_map_util
from ..tf_dependency import visualization_utils as vis_util


class ObjectDetectionWorker(object):
    """
    封装基于TensorFlow的物体检测模型检测逻辑
    """

    def __init__(self, PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES):
        self.detection_graph = tf.Graph()
        self.PATH_TO_CKPT = PATH_TO_CKPT
        self.PATH_TO_LABELS = PATH_TO_LABELS
        self.NUM_CLASSES = NUM_CLASSES
        self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=NUM_CLASSES,
                                                                         use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

    def detect_objects(self, image_np, sess, detection_graph):
        """
        物体检测的逻辑
        :param self
        :param image_np:
        :param sess:
        :param detection_graph:
        :return:
        """
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=3)
        return image_np, classes[0], scores[0]

    def detect_worker(self, input_q, output_q):
        with self.detection_graph.as_default():  # 加载模型
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=self.detection_graph)

        while True:  # 全局变量input_q与output_q定义，请看下文
            to_check = input_q.get()  # 从多进程输入队列，取值
            # detect_objects函数 返回一张图片，标记所有被发现的物品
            output_q.put(self.detect_objects(to_check, sess, self.detection_graph))
        sess.close()
