# coding: utf-8
"""
Train model with tfrecords. Using feed_dict.
"""

import tensorflow as tf
import toTFrecord

def train():
	with tf.Session() as sess:  
		# next batch?
		filename = ''
	    img_batch, path_batch = read_tfRecord(filename)
	    # batch for training!
	    image_batches, label_batches = tf.train.batch([img_batch, path_batch], batch_size=batch, capacity=4096)
	    tf.local_variables_initializer().run()
	    coord = tf.train.Coordinator()
	    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
	    """
	    # 定义一个模型
	    model=ATDA(sess=sess)  
	    model.create_model()  
	    # 训练模型：（image_batches，label_batches）是训练集，（image_batches2，label_batches2）是测试集，  
	    model.fit_ATDA(source_train=image_batches, y_train=label_batches,  
	                   target_val=image_batches2, y_val=label_batches2,  
	                   # n是训练集总数，my_number是测试集总数，my_catelogy是标签种类，batch是迭代次数  
	                   nb_epoch=epochs, n = 86524, my_number = 25596, my_catelogy = 2,batch = 16)
	    """
	    coord.request_stop()  # 请求线程结束
	    coord.join()  # 等待线程结束