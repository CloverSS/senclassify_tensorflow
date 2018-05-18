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

file_pos="D:/python/data/data_tan_pos.txt"
file_neg="D:/python/data/data_tan_neg.txt"
#file_mid="D:/python/data/douban_data_m.txt"
file_stopwd="D:/python/data/stopwd.txt"
file_dict="D:/python/data/dict_class_2.pkl"
file_tensor_model="D:/python/model/tensorflow/model_bp_class2.ckpt"
file_chi2_model="D:/python/data/model_chi2_bp2.pkl"
stopwdlist=data_handler.stopwordslist(file_stopwd)

def save_target(target,file_path):
	output = open(file_path, 'wb')
	pickle.dump(target, output, -1)
	output.close()
	
data_befvec=data_handler.data_prevocab(file_pos,stopwdlist)
data_befvec+=data_handler.data_prevocab(file_neg,stopwdlist)
#data_befvec+=data_handler.data_prevocab(file_mid,stopwdlist)
dict=data_handler.build_vocab(data_befvec,5)
save_target(dict,file_dict)
print(len(dict))

data=[]
data.extend(data_handler.data_tovec(file_pos,[1,0],dict,stopwdlist))
data.extend(data_handler.data_tovec(file_neg,[0,1],dict,stopwdlist))
#data.extend(data_handler.data_tovec(file_mid,[0,0,1],dict,stopwdlist))
random.shuffle(data)
print(len(data))
data = np.array(data)
model1 = SelectKBest(chi2, k=400)
data_x=model1.fit_transform(list(data[:,0]), list(data[:,1]))
save_target(model1,file_chi2_model)
#data_x=list(data[:,0])
data_y=list(data[:,1])  #卡方过滤
#print 'size of train_dataset is {}'.format(train_dataset)

#神经网络定义及训练（双隐层网络）

n_input_layer = 400  #输入向量维度
n_layer_1 = 400   
n_output_layer=2

def define_layer(input,input_n,output_n):  #添加一个神经网络层	
	weight=tf.Variable(tf.random_normal([input_n, output_n]))
	baise=tf.Variable(tf.random_normal([output_n]))
	layer=tf.matmul(input,weight)+baise
	return layer
#定义待训练的神经网络

def define_network(data):
	layer_1=define_layer(data,n_input_layer,n_layer_1)
	layer_1 = tf.nn.relu(layer_1)
	#layer_2=define_layer(layer_1,n_layer_1,n_layer_2)
	#layer_2 = tf.nn.relu(layer_2)
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
        
        print('训练集数据量 {}'.format(len(data_x)))
        for epoch in range(epochs):
            epoch_loss = 0 #每个周期的loss
            i = 0
            while i < len(data_x):
                start = i
                end = i + batch_size
                batch_x = data_x[start:end]
                batch_y = data_y[start:end]
                _, c = session.run([optimizer, cost_func], feed_dict={X:batch_x, Y:batch_y})
                epoch_loss += c
                i = end
            print('迭代次数',epoch, ' : 损失函数', epoch_loss)
        tf.add_to_collection('predict', predict)
        saver = tf.train.Saver(tf.all_variables())		
        saver_path = saver.save(session, file_tensor_model,global_step=epochs)
        print("saveer path:",saver_path)

train_neural_network(X, Y)