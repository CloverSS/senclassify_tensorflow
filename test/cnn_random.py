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

from gensim.models import Word2Vec
import jieba
import jieba.posseg as pseg
import jieba.analyse
from sklearn.feature_extraction.text import CountVectorizer

jieba.load_userdict("D:/python/data/jieba_dict.txt")
#file_pos="D:/python/data/douban_data_p.txt"
#file_neg="D:/python/data/douban_data_n.txt"
#file_mid="D:/python/data/douban_data_m.txt"
file_pos="D:/python/data/data_tan_pos_s.txt"
file_neg="D:/python/data/data_tan_neg_s.txt"
file_stopwd="D:/python/data/stopwd.txt"
sentenct_length=0

def stopwordslist(file_stop):    #加载停用词词典
	stopwords = [line.strip() for line in open(file_stop, 'r', encoding='utf-8').readlines()]  
	return stopwords  
stopwdlist=stopwordslist(file_stopwd)

def line_cutstop_str(line):     #返回分词后句子(str)
	global sentenct_length
	result=jieba.cut(line.strip())   #结巴分词
	outstr=''
	i=0
	for word in result:  
		if word not in stopwdlist:  
			if word != '\t':
				i+=1
				outstr += word  
				outstr += " "
	if i>sentenct_length:
		sentenct_length=i
	return outstr
		
def line_cutstop_list(line):   #返回分词后句子（list)
	result=jieba.cut(line.strip())
	outstr=[]
	for word in result:  
		if word not in stopwdlist:  
			if word != '\t':  
				outstr.append(word)
	return outstr
	
def data_prevocab(file_data):   #返回分词后的句子list （句子与句子构成list，每个句子的分词以空格隔开）
	with open(file_data,"r+",encoding='UTF-8') as raw_dt:
		lines=raw_dt.readlines()
		data_aftcut=[]
		i=0
		for line in lines:
			data_aftcut.append(line_cutstop_str(line))
			#print(line_cutstop(line),i)
			i+=1
			#if i>5:
			#	break
		return data_aftcut
	
def build_vocab(data_befvec):     #特征提取，返回字典list
	vectorize_tf = CountVectorizer(max_df=0.9, min_df=10)  #tf-idf,至少出现10次
	vectorize = vectorize_tf.fit_transform(data_befvec)
	return list(vectorize_tf.vocabulary_.keys())  #返回字典
	
data_befvec=data_prevocab(file_pos)
data_befvec+=data_prevocab(file_neg)
#data_befvec+=data_prevocab(file_mid)
print("sentenct_length: %s",format(sentenct_length))
dict=build_vocab(data_befvec)
print(len(dict))
	
def data_tovec(file_data,flag):       #特征向量表示，返回向量list
	data=[]
	with open(file_data,"r",encoding='UTF-8') as raw_dt:
		lines=raw_dt.readlines()
		i=0
		for line in lines:
			word_lists=line_cutstop_list(line)
			line_vec=np.zeros(sentenct_length)
			i=i+1
			#if i>3:
			#	break
			for num,word in enumerate(word_lists):
				if word in dict:
					line_vec[num]=dict.index(word)
			data.append([line_vec,flag])
	return data		

data=[]
data.extend(data_tovec(file_pos,[1,0]));
#data.extend(data_tovec(file_mid,[0,1,0]));
data.extend(data_tovec(file_neg,[0,1]));
#print(data)
print(len(data))
random.shuffle(data)  #打乱顺序

test_size = int(len(data) * 0.2)  #取20%数据为测试数据
data = np.array(data)
data_x=list(data[:,0])
data_y=list(data[:,1])

train_data_x = data_x[:-test_size]
train_data_y = data_y[:-test_size]
test_data_x = data_x[-test_size:]
test_data_y = data_y[-test_size:]
print(train_data_x[0:1])

# Training
# ==================================================
with tf.Session() as sess:
	sequence_length=sentenct_length
	num_classes=2
	vocab_size=len(dict)
	embedding_size=400
	filter_sizes={3,4,5}
	num_filters=128
	l2_reg_lambda=0.0
	
	input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
	input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
	dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

	# Keeping track of l2 regularization loss (optional)
	l2_loss = tf.constant(0.0)

	W = tf.Variable(
		tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
		name="W")
	embedded_chars = tf.nn.embedding_lookup(W, input_x)
	embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

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
	
	print(sess.run(cnn.accuracy, {cnn.input_x: test_data_x, cnn.input_y:test_data_y,cnn.dropout_keep_prob: 1.0}))
	