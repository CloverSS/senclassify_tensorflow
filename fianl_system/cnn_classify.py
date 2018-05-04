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

#file_pos="D:/python/data/douban_data_p.txt"
#file_neg="D:/python/data/douban_data_n.txt"
#file_mid="D:/python/data/douban_data_m.txt"
file_pos="D:/python/data/data_pos.txt"
file_neg="D:/python/data/data_neg.txt"
file_stopwd="D:/python/data/stopwd.txt"
sentence_length=0

'''def stopwordslist(file_stop):    #加载停用词词典
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
	return list(vectorize_tf.vocabulary_.keys())  #返回字典  '''

stopwdlist=data_handler.stopwordslist(file_stopwd)
sentence_length=data_handler.max_sentence(file_pos,sentence_length,stopwdlist)
sentence_length=data_handler.max_sentence(file_neg,sentence_length,stopwdlist)
data_x_pos,data_y_pos=data_handler.data_tovec_w2v(file_pos,[1,0],sentence_length,stopwdlist)
data_x_neg,data_y_neg=data_handler.data_tovec_w2v(file_neg,[0,1],sentence_length,stopwdlist)
data_raw_x=data_x_pos+data_x_neg
data_raw_y=data_y_pos+data_y_neg
del data_x_neg,data_y_neg,data_x_pos,data_y_pos
#print(data)
print(len(data_raw_x))
data_raw_x = np.array(data_raw_x)
data_raw_y = np.array(data_raw_y)

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(data_raw_x)))
data_x = data_raw_x[shuffle_indices]
del data_raw_x
data_y = data_raw_y[shuffle_indices]
del data_raw_y

test_size = int(len(data_x) * 0.2)  #取20%数据为测试数据

train_data_x = data_x[:-test_size]
train_data_y = data_y[:-test_size]
test_data_x = data_x[-test_size:]
test_data_y = data_y[-test_size:]
print(train_data_x[0:3])
print(train_data_y[0:3])

# Model Hyperparameters
#tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
embedding_dim=400
#tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
filter_sizes="3,4,5"
#tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
num_filters=400
#tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
dropout_keep_prob=0.5
#tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
l2_reg_lambda=0.0

# Training
# ==================================================


with tf.Session() as sess:
	cnn = TextCNN(
		sequence_length=sentence_length,
		num_classes=2,
		embedding_size=embedding_dim,
		filter_sizes=list(map(int, filter_sizes.split(","))),
		num_filters=128,
		l2_reg_lambda=0.0)

	# Define Training procedure
	global_step = tf.Variable(0, name="global_step", trainable=False)
	optimizer = tf.train.AdamOptimizer(1e-3)
	grads_and_vars = optimizer.compute_gradients(cnn.loss)
	train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

	sess.run(tf.global_variables_initializer())
	def train_step(x_batch, y_batch):
		"""
		A single training step
		"""
		feed_dict = {
		  cnn.input_x: x_batch,
		  cnn.input_y: y_batch,
		  cnn.dropout_keep_prob: dropout_keep_prob
		}
		_, step, loss, accuracy = sess.run(
			[train_op, global_step, cnn.loss, cnn.accuracy],
			feed_dict)
		time_str = datetime.datetime.now().isoformat()
		print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
		#train_summary_writer.add_summary(summaries, step)

	epochs=20
	batch_size=100
	for epoch in range(epochs):
		i = 0
		while i < len(train_data_x):
			start = i
			end = i + batch_size
			batch_x = train_data_x[start:end]
			batch_y = train_data_y[start:end]
			#run的第一个参数fetches可以是单个,也可以是多个。 返回值是fetches的返回值。
			#此处因为要打印cost,所以cost_func也在fetches中
			train_step(batch_x, batch_y)
			i = end
		print("epoch: {}",epoch)
	'''all_predictions = []
	batch_predictions = sess.run(cnn.predictions, {cnn.input_x: test_data_x, cnn.dropout_keep_prob: 1.0})
	all_predictions = np.concatenate([all_predictions, batch_predictions])
	correct_predictions = float(sum(all_predictions == test_data_y))
	print("Accuracy: {}".format(correct_predictions/float(len(test_data_y))))'''
	print(sess.run(cnn.accuracy, {cnn.input_x: test_data_x, cnn.input_y:test_data_y,cnn.dropout_keep_prob: 1.0}))