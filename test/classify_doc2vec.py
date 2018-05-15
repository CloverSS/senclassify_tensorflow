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
import sys
from gensim.models import Word2Vec
import logging,gensim
from gensim.models import Doc2Vec

file_pos="D:/python/data/data_tan_pos.txt"
file_neg="D:/python/data/data_tan_neg.txt"
file_mid="D:/python/data/data_mid.txt"
file_stopwd="D:/python/data/stopwd.txt"

def stopwordslist(file_stop):  
	stopwords = [line.strip() for line in open(file_stop, 'r', encoding='utf-8').readlines()]  
	return stopwords  
stopwdlist=stopwordslist(file_stopwd)

def line_cutstop_str(line):
	result=jieba.cut(line.strip())
	outstr=''
	for word in result:  
		if word not in stopwdlist:  
			if word != '\t':  
				outstr += word  
				outstr += " "
	return outstr
		
def line_cutstop_list(line):
	result=jieba.cut(line.strip())
	outstr=[]
	for word in result:  
		if word not in stopwdlist:  
			if word != '\t':  
				outstr.append(word)
	return outstr
	
def data_prevocab(file_data):   #返回分词list
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
	
def build_vocab(data_befvec):     #返回字典list
	vectorize_tf = CountVectorizer(max_df=0.9, min_df=5)
	vectorize = vectorize_tf.fit_transform(data_befvec)
	return list(vectorize_tf.vocabulary_.keys())
	
data_befvec=data_prevocab(file_pos)
data_befvec+=data_prevocab(file_neg)
dict=build_vocab(data_befvec)
print(len(dict))

def data_tovec(file_data,flag):       #返回向量list
	data=[]
	model = Doc2Vec.load('D:/python/data/doc2vec_tan_1.model')
	with open(file_data,"r",encoding='UTF-8') as raw_dt:
		lines=raw_dt.readlines()
		i=0
		for line in lines:
			i=i+1
			word_lists=line_cutstop_list(line)
			line_vec=model.infer_vector(word_lists)
			data.append([line_vec,flag])
	return data		

data=[]
data.extend(data_tovec(file_pos,[1,0]));
#data.extend(data_tovec(file_mid,[0,1,0]));
data.extend(data_tovec(file_neg,[0,1]));
print(len(data))
print(data[0:3])
random.shuffle(data)  #打乱顺序

#取样本的10%作为测试数据
test_size = int(len(data) * 0.1)
data = np.array(data)
train_data = data[:-test_size]
test_data = data[-test_size:]
print('test_size = {}'.format(test_size))
#print 'size of train_dataset is {}'.format(train_dataset)

#Feed-forward nueral network
#定义每个层有多少个神经元
n_input_layer = 200   #输入层每个神经元代表一个term

n_layer_1 = 100  #hiden layer
n_layer_2 = 100 # hiden layer
n_layer_3 = 100
n_output_layer = 2

#定义待训练的神经网络
def neural_netword(data):
    #定义第一层神经元的w和b, random_normal定义服从正态分布的随机变量
    layer_1_w_b = {'w_':tf.Variable(tf.random_normal([n_input_layer, n_layer_1])), 'b_':tf.Variable(tf.random_normal([n_layer_1]))}
    layer_2_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_1, n_layer_2])), 'b_':tf.Variable(tf.random_normal([n_layer_2]))}
   # layer_3_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_2, n_layer_3])), 'b_':tf.Variable(tf.random_normal([n_layer_3]))}
    layer_output_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_2, n_output_layer])), 'b_':tf.Variable(tf.random_normal([n_output_layer]))}

    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
    layer_1 = tf.nn.relu(layer_1) #relu做激活函数
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2)
    #layer_3 = tf.add(tf.matmul(layer_2, layer_3_w_b['w_']), layer_3_w_b['b_'])
    #layer_3 = tf.nn.relu(layer_3)
    layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w_']), layer_output_w_b['b_'])
    return layer_output

batch_size = 20
X = tf.placeholder('float', [None, n_input_layer])  #None表示样本数量任意; 每个样本纬度是term数量
Y = tf.placeholder('float')

#使用数据训练神经网络
def train_neural_network(X, Y):
    predict = neural_netword(X)
    #cost func是输出层softmax的cross entropy的平均值。 将softmax 放在此处而非nn中是为了效率.
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))
    #设置优化器
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)
    #optimizer=tf.train.GradientDescentOptimizer(0.1).minimize(cost_func)
	
    epochs = 20  #epoch本意是时代、纪, 这里是迭代周期
    with tf.Session() as session:
        session.run(tf.initialize_all_variables()) #初始化所有变量,包括w,b

        random.shuffle(train_data)
        train_x = train_data[:, 0] #每一行的features;
        train_y = train_data[:, 1] #每一行的label
        print('size of train_x is {}'.format(len(train_x)))
        for epoch in range(epochs):
            epoch_loss = 0 #每个周期的loss
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = train_x[start:end]
                batch_y = train_y[start:end]
                #run的第一个参数fetches可以是单个,也可以是多个。 返回值是fetches的返回值。
                #此处因为要打印cost,所以cost_func也在fetches中
                _, c = session.run([optimizer, cost_func], feed_dict={X:list(batch_x), Y:list(batch_y)})
                epoch_loss += c
                i = end
            print(epoch, ' : ', epoch_loss)

        #评估模型
        test_x = test_data[:, 0]
        test_y = test_data[:, 1]
        #argmax能给出某个tensor对象在某一维上的其数据最大值所在的索引值, 这里是索引值的list。tf.equal用于检测匹配,返回bool型的list
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        #tf.cast 可以将[True, False, True] 转化为[1, 0, 1]
        #reduce_mean用于在某一维上计算平均值, 未指定纬度则计算所有元素
        accurqcy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('doc2vec 准确率: {}'.format(accurqcy.eval({X:list(test_x), Y:list(test_y)})))
        #等价: print session.run(accuracy, feed_dict={X:list(test_x), Y:list(test_y)})

train_neural_network(X, Y)