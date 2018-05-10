#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from text_cnn_wordvec import TextCNN
from tensorflow.contrib import learn
import os
import traceback
import random
import logging,gensim
import data_handler

class CNNClassify(object):
	def __init__ (
	  self,file_pos,file_neg,file_aim,file_res,num_classes,file_mid=""):
	  
		#file_pos="D:/python/data/data_tan_pos_s.txt"
		#file_neg="D:/python/data/data_tan_neg_s.txt"
		file_stopwd="D:/python/data/stopwd.txt"
		#file_aim="D:/python/data/data_tan_test_2.txt"
		#file_res="D:/python/data/res_tan_cnn_1.txt"
		sentence_length=0

		stopwdlist=data_handler.stopwordslist(file_stopwd)
		sentence_length=data_handler.max_sentence(file_pos,sentence_length,stopwdlist)
		sentence_length=data_handler.max_sentence(file_neg,sentence_length,stopwdlist)
		if num_classes==2:
			data_x_pos,data_y_pos=data_handler.data_tovec_w2v(file_pos,[1,0],sentence_length,stopwdlist)
			data_x_neg,data_y_neg=data_handler.data_tovec_w2v(file_neg,[0,1],sentence_length,stopwdlist)
			data_raw_x=data_x_pos+data_x_neg
			data_raw_y=data_y_pos+data_y_neg
			del data_x_neg,data_y_neg,data_x_pos,data_y_pos
		else:
			sentence_length=data_handler.max_sentence(file_mid,sentence_length,stopwdlist)
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

		test_data_x,test_data_y=data_handler.data_tovec_w2v(file_aim,[0,0],sentence_length,stopwdlist)
		test_data_x=np.array(test_data_x)

		# Training
		# ==================================================
		with tf.Session() as sess:
			sequence_length=sentence_length
			num_classes=num_classes
			embedding_size=400
			#filter_sizes=list(map(int, filter_sizes.split(",")))
			filter_sizes={3,4,5}
			num_filters=128
			l2_reg_lambda=0.0
			
			input_x = tf.placeholder(tf.float32, [None, sequence_length,embedding_size], name="input_x")
			input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
			dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

			# Keeping track of l2 regularization loss (optional)
			l2_loss = tf.constant(0.0)

			# Embedding layer
			embedded_chars_expanded = tf.expand_dims(input_x, -1)

			pooled_outputs = []
			for i, filter_size in enumerate(filter_sizes):
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
			
			res=sess.run(predictions, feed_dict={input_x:test_data_x, input_y:test_data_y,dropout_keep_prob:1.0})
			with open(file_res,"a+",encoding='UTF-8') as res_f:
				with open(file_aim,"r+",encoding='UTF-8') as aim_f:
					lines=aim_f.readlines()
					for num,line in enumerate(lines):
						if res[num]==0:
							res_f.write("正向  "+line+"\n")
						if res[num]==1:
							res_f.write("负向  "+line+"\n")       
			print("结果写入：%s"%file_res)
