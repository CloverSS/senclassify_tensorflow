#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
import os
import traceback
import random
import logging,gensim
import data_handler

file_pos="D:/python/data/data_use/douban_data_p_ss.txt"
file_neg="D:/python/data/data_use/douban_data_n_ss.txt"
file_mid="D:/python/data/data_use/douban_data_m_ss.txt"
file_stopwd="D:/python/data/stopwd.txt"
#file_aim="D:/python/data/data_tan_test_2.txt"
#file_res="D:/python/data/res_tan_cnn_1.txt"
file_tensor_model="D:/python/model/tensorflow/model_data_cnn_cl3.ckpt"
sentence_length=200

stopwdlist=data_handler.stopwordslist(file_stopwd)
#sentence_length=data_handler.max_sentence(file_pos,sentence_length,stopwdlist)
#sentence_length=data_handler.max_sentence(file_neg,sentence_length,stopwdlist)
data_x_pos,data_y_pos=data_handler.data_tovec_w2v(file_pos,[1,0,0],sentence_length,stopwdlist)
data_x_neg,data_y_neg=data_handler.data_tovec_w2v(file_neg,[0,1,0],sentence_length,stopwdlist)
data_x_mid,data_y_mid=data_handler.data_tovec_w2v(file_mid,[0,0,1],sentence_length,stopwdlist)
data_raw_x=data_x_pos+data_x_neg+data_x_mid
data_raw_y=data_y_pos+data_y_neg+data_y_mid
del data_x_neg,data_y_neg,data_x_pos,data_y_pos,data_x_mid,data_y_mid
#print(data)
print(len(data_raw_x))
data_raw_x = np.array(data_raw_x)
data_raw_y = np.array(data_raw_y)
shuffle_indices = np.random.permutation(np.arange(len(data_raw_x)))
train_data_x = data_raw_x[shuffle_indices]
del data_raw_x
train_data_y = data_raw_y[shuffle_indices]
del data_raw_y

#test_data_x,test_data_y=data_handler.data_tovec_w2v(file_aim,[0,0],sentence_length,stopwdlist)
#test_data_x=np.array(test_data_x)
#test_size = int(len(data_x) * 0.2)  #取20%数据为测试数据
'''train_data_x = data_x[:-test_size]
train_data_y = data_y[:-test_size]
test_data_x = data_x[-test_size:]
test_data_y = data_y[-test_size:]'''

print(train_data_x[0:3])
print(train_data_y[0:3])

filter_sizes="3,4,5"

# Training
# ==================================================
with tf.Session() as sess:
	sequence_length=sentence_length
	num_classes=3
	embedding_size=400
	filter_sizes={3,4,5}
	num_filters=128
	l2_reg_lambda=0.0
	
	input_x = tf.placeholder(tf.float32, [None, sequence_length,embedding_size], name="input_x")
	input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
	dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

	l2_loss = tf.constant(0.0)

	embedded_chars_expanded = tf.expand_dims(input_x, -1)

	pooled_outputs = []
	for i, filter_size in enumerate(filter_sizes):
		with tf.name_scope("conv-maxpool-%s" % filter_size):
			# 卷积
			filter_shape = [filter_size, embedding_size, 1, num_filters]
			W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
			b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
			conv = tf.nn.conv2d(
				embedded_chars_expanded,
				W,
				strides=[1, 1, 1, 1],
				padding="VALID",
				name="conv")
			h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
			# Maxpooling
			pooled = tf.nn.max_pool(
				h,
				ksize=[1, sequence_length - filter_size + 1, 1, 1],
				strides=[1, 1, 1, 1],
				padding='VALID',
				name="pool")
			pooled_outputs.append(pooled)

	num_filters_total = num_filters * len(filter_sizes)
	h_pool = tf.concat(pooled_outputs, 3)
	h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

	h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

	W = tf.get_variable(
		"W",
		shape=[num_filters_total, num_classes],
		initializer=tf.contrib.layers.xavier_initializer())
	b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
	l2_loss += tf.nn.l2_loss(W)
	l2_loss += tf.nn.l2_loss(b)
	scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
	predictions = tf.argmax(scores, 1, name="predictions")

	losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y)
	loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
	
	correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

	global_step = tf.Variable(0, name="global_step", trainable=False)
	optimizer = tf.train.AdamOptimizer(1e-3)
	grads_and_vars = optimizer.compute_gradients(loss)
	train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

	sess.run(tf.global_variables_initializer())
	epochs=20
	batch_size=1000
	for epoch in range(epochs):
		i = 0
		while i < len(train_data_x):
			start = i
			end = i + batch_size
			batch_x = train_data_x[start:end]
			batch_y = train_data_y[start:end]
			#run的第一个参数fetches可以是单个,也可以是多个。 返回值是fetches的返回值。
			#此处因为要打印cost,所以cost_func也在fetches中
			#train_step(batch_x, batch_y)
			feed_dict = {
			  input_x: batch_x,
			  input_y: batch_y,
			  dropout_keep_prob:0.5
			}
			_, step_r, loss_r, accuracy_r = sess.run([train_op, global_step, loss, accuracy],feed_dict)
			time_str = datetime.datetime.now().isoformat()
			print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step_r, loss_r, accuracy_r))
			i = end
		print("epoch: {}",epoch)
	
	tf.add_to_collection('predictions', predictions)
	saver = tf.train.Saver(tf.all_variables())			
	saver_path = saver.save(sess, file_tensor_model,global_step=epochs)
	print("saveer path:",saver_path)
	