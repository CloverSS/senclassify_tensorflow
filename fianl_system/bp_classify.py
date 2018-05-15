# -*- coding: utf-8 -*-
import jieba
import jieba.posseg as pseg
import jieba.analyse
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import tensorflow as tf
import os
import traceback
import random
import pickle
import data_handler

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif

class BpClassify(object):
	def __init__ (
	  self,file_pos,file_neg,file_aim,file_res,num_classes,file_mid=""):

		#file_pos="D:/python/data/data_tan_pos.txt"
		#file_neg="D:/python/data/data_tan_neg.txt"
		#file_aim="D:/python/data/data_tan_test_2.txt"
		#file_res="D:/python/data/res_tan_2.txt"
		#file_mid="D:/python/data/data_mid.txt"
		file_stopwd="D:/python/data/stopwd.txt"

		stopwdlist=data_handler.stopwordslist(file_stopwd)
		data_befvec=data_handler.data_prevocab(file_pos,stopwdlist)
		data_befvec+=data_handler.data_prevocab(file_neg,stopwdlist)
		data_befvec+=data_handler.data_prevocab(file_aim,stopwdlist)
		if num_classes==3:
			data_befvec+=data_handler.data_prevocab(file_mid,stopwdlist)
		len_aim=data_handler.count_lines(file_aim)
		dict=data_handler.build_vocab(data_befvec,5)
		print(len(dict))

		data=[]
		data_test=[]
		if num_classes==2:
			data.extend(data_handler.data_tovec(file_pos,[1,0],dict,stopwdlist))
			data.extend(data_handler.data_tovec(file_neg,[0,1],dict,stopwdlist))
			random.shuffle(data)
			data_test.extend(data_handler.data_tovec(file_aim,[0,0],dict,stopwdlist))
		else:
			data.extend(data_handler.data_tovec(file_pos,[1,0,0],dict,stopwdlist))
			data.extend(data_handler.data_tovec(file_neg,[0,1,0],dict,stopwdlist))
			data.extend(data_handler.data_tovec(file_mid,[0,0,1],dict,stopwdlist))
			random.shuffle(data)
			data_test.extend(data_handler.data_tovec(file_aim,[0,0,0],dict,stopwdlist))
		print(len(data))

		data = np.array(data)
		data_test=np.array(data_test)
		model1 = SelectKBest(chi2, k=400)
		train_data_x=model1.fit_transform(list(data[:,0]), list(data[:,1]))
		train_data_y=list(data[:,1])
		test_data_x =model1.transform(list(data_test[:,0]))
		test_data_y = data_test[:,1]

		#train_data_x = data_x[:-test_size]
		#train_data_y = data_y[:-test_size]
		#test_data_x = data_x[-test_size:]
		#test_data_y = data_y[-test_size:]

		#神经网络定义及训练（双隐层网络）

		n_input_layer = 400  #输入向量维度
		n_layer_1 = 400  
		n_output_layer=num_classes

		def define_layer(input,input_n,output_n):  #添加一个神经网络层	
			weight=tf.Variable(tf.random_normal([input_n, output_n]))
			baise=tf.Variable(tf.random_normal([output_n]))
			layer=tf.matmul(input,weight)+baise
			return layer
		#定义待训练的神经网络

		def define_network(data):
			layer_1=define_layer(data,n_input_layer,n_layer_1)
			layer_1 = tf.nn.relu(layer_1)
			layer_output=define_layer(layer_1,n_layer_1,n_output_layer)
			return layer_output
			
		batch_size = 20
		X = tf.placeholder('float', [None, n_input_layer],name='X')  #占位符
		Y = tf.placeholder('float',name='Y')

		#使用数据训练神经网络
		def train_neural_network(X, Y):
			predict = define_network(X)    #定义神经网络
			reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())
			cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))+reg   #定义代价函数，这里用交叉熵损失实现
			optimizer = tf.train.AdamOptimizer().minimize(cost_func)   #adam优化器
			#optimizer=tf.train.GradientDescentOptimizer(0.1).minimize(cost_func)
			
			epochs = 20  #迭代周期
			with tf.Session() as session:
				session.run(tf.global_variables_initializer()) #tensorflow初始化
				
				print('训练集数据量 {}'.format(len(train_data_x)))
				for epoch in range(epochs):
					epoch_loss = 0 #每个周期的loss
					i = 0
					while i < len(train_data_x):
						start = i
						end = i + batch_size
						batch_x = train_data_x[start:end]
						batch_y = train_data_y[start:end]
						_, c = session.run([optimizer, cost_func], feed_dict={X:batch_x, Y:batch_y})
						epoch_loss += c
						i = end
					print('迭代次数',epoch, ' : 损失函数', epoch_loss)

				res=session.run(tf.argmax(predict,1), feed_dict={X:list(test_data_x), Y:list(test_data_y)})

				with open(file_res,"a+",encoding='UTF-8') as res_f:
					with open(file_aim,"r+",encoding='UTF-8') as aim_f:
						lines=aim_f.readlines()
						for num,line in enumerate(lines):
							if res[num]==0:
								res_f.write("正向  "+line+"\n")
							if res[num]==1:
								res_f.write("负向  "+line+"\n")
							if res[num]==2:
								res_f.write("中性  "+line+"\n")
				print("结果写入：%s"%file_res)

		train_neural_network(X, Y)