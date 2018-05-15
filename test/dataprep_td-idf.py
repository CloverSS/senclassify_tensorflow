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

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
#from minepy import MINE  
jieba.load_userdict("D:/python/data/jieba_dict.txt")
file_pos="D:/python/data/data_tan_pos.txt"
file_neg="D:/python/data/data_tan_neg.txt"
#file_mid="D:/python/data/douban_data_m.txt"
file_stopwd="D:/python/data/stopwd.txt"

def stopwordslist(file_stop):    #加载停用词词典
	stopwords = [line.strip() for line in open(file_stop, 'r', encoding='utf-8').readlines()]  
	return stopwords  
stopwdlist=stopwordslist(file_stopwd)

def line_cutstop_str(line):     #返回分词后句子(str)
	result=jieba.cut(line.strip())   #结巴分词
	outstr=''
	for word in result:  
		if word not in stopwdlist:  
			if word != '\t':  
				outstr += word  
				outstr += " "
	return outstr
		
def line_cutstop_list(line):   #返回分词后句子（list)
	result=jieba.cut(line.strip())
	outstr=[]
	for word in result:  
		if word not in stopwdlist:  
			if word != '\t':  
				outstr.append(word)
	return outstr
	
def data_prevocab(file_data,flag):   #返回分词后的句子list （句子与句子构成list，每个句子的分词以空格隔开）
	with open(file_data,"r+",encoding='UTF-8') as raw_dt:
		lines=raw_dt.readlines()
		data_aftcut=[]
		data_y=[]
		for line in lines:
			data_aftcut.append(line_cutstop_str(line))
			data_y.append(flag)
		return data_aftcut,data_y
	
def build_vocab(data_befvec):     #特征提取，返回字典list
	vectorize_tf = CountVectorizer(max_df=0.9,min_df=5)  #tf-idf,至少出现10次
	vectorize = vectorize_tf.fit_transform(data_befvec)
	return vectorize.toarray()  #返回字典

data_befvec_pos,data_y_pos=data_prevocab(file_pos,[1,0])
data_befvec_neg,data_y_neg=data_prevocab(file_neg,[0,1])
data_befvec=data_befvec_pos+data_befvec_neg
data_raw_x=build_vocab(data_befvec)
data_raw_y=data_y_pos+data_y_neg
del data_befvec_pos,data_befvec_neg,data_befvec,data_y_neg,data_y_pos
model1 = SelectKBest(chi2, k=200)
#SelectKBest(lambda X, Y: array(map(lambda x:mic(x, Y), X.T)).T, k=200)
data_raw_x=model1.fit_transform(list(data_raw_x), data_raw_y)
data_raw_x=np.array(data_raw_x)
data_raw_y=np.array(data_raw_y)
print("data_x size: %s   data_y size:%s"%(len(data_raw_x),len(data_raw_y)))

'''def data_tovec(file_data,flag):       #特征向量表示，返回向量list
	data=[]
	with open(file_data,"r",encoding='UTF-8') as raw_dt:
		lines=raw_dt.readlines()
		i=0
		for line in lines:
			word_lists=line_cutstop_list(line)
			line_vec=np.zeros(len(dict))
			i=i+1
			#if i>3:
			#	break
			for word in word_lists:
				if word in dict:
					line_vec[dict.index(word)]=1
			data.append([line_vec,flag])
	return data		

data=[]
data.extend(data_tovec(file_pos,[1,0]));
#data.extend(data_tovec(file_mid,[0,1,0]));
data.extend(data_tovec(file_neg,[0,1]));
#print(data)
print(len(data))
random.shuffle(data)  #打乱顺序 '''

shuffle_indices = np.random.permutation(np.arange(len(data_raw_x)))
data_x = data_raw_x[shuffle_indices]
data_y = data_raw_y[shuffle_indices]
del data_raw_x,data_raw_y
test_size = int(len(data_y) * 0.1)  #取20%数据为测试数据
train_data_x = data_x[:-test_size]
train_data_y = data_y[:-test_size]
test_data_x = data_x[-test_size:]
test_data_y = data_y[-test_size:]
print(train_data_x[0:3])
print(train_data_y[0:3])

print('test_size = {}'.format(test_size))
#print 'size of train_dataset is {}'.format(train_dataset)

#神经网络定义及训练（双隐层网络）

n_input_layer = 200   #输入向量维度
n_layer_1 = 50  
n_layer_2 = 50
n_layer_3 = 100 
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
	layer_2=define_layer(layer_1,n_layer_1,n_layer_2)
	layer_2 = tf.nn.relu(layer_2)
	#layer_3=define_layer(layer_2,n_layer_2,n_layer_3)
	#layer_3 = tf.nn.relu(layer_3)
	layer_output=define_layer(layer_2,n_layer_2,n_output_layer)
	return layer_output
	
batch_size = 20
X = tf.placeholder('float', [None, n_input_layer])  #占位符
Y = tf.placeholder('float')

#使用数据训练神经网络
def train_neural_network(X, Y):
    predict = define_network(X)    #定义神经网络
    reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), tf.trainable_variables())
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))+reg   #定义代价函数，这里用交叉熵损失实现
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)   #adam优化器
    #optimizer=tf.train.GradientDescentOptimizer(0.1).minimize(cost_func)
	
    epochs = 20  #迭代周期
    with tf.Session() as session:
        session.run(tf.initialize_all_variables()) #tensorflow初始化

        #random.shuffle(train_data)
        #train_x = train_data[:, 0] #每一行的features;
        #train_y = train_data[:, 1] #每一行的label
        print('训练集数据量 {}'.format(len(train_data_x)))
        for epoch in range(epochs):
            epoch_loss = 0 #每个周期的loss
            i = 0
            while i < len(train_data_x):
                start = i
                end = i + batch_size
                batch_x = train_data_x[start:end]
                batch_y = train_data_y[start:end]
                #run的第一个参数fetches可以是单个,也可以是多个。 返回值是fetches的返回值。
                #此处因为要打印cost,所以cost_func也在fetches中
                _, c = session.run([optimizer, cost_func], feed_dict={X:batch_x, Y:batch_y})
                epoch_loss += c
                i = end
            #end=len(train_x)-1
            #batch_x=train_x[0:end]
            #batch_y=train_y[0:end]
            #_, epoch_loss= session.run([optimizer, cost_func], feed_dict={X:list(batch_x), Y:list(batch_y)})
            #if epoch%20==0:
            print('迭代次数',epoch, ' : 损失函数', epoch_loss)

        #评估模型
        #test_x = test_data[:, 0]
        #test_y = test_data[:, 1]
        #argmax能给出某个tensor对象在某一维上的其数据最大值所在的索引值, 这里是索引值的list。tf.equal用于检测匹配,返回bool型的list
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        #tf.cast 可以将[True, False, True] 转化为[1, 0, 1]
        #reduce_mean用于在某一维上计算平均值, 未指定纬度则计算所有元素
        accurqcy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('tf-idf 准确率: {}'.format(accurqcy.eval({X:list(test_data_x), Y:list(test_data_y)})))
        #等价: print session.run(accuracy, feed_dict={X:list(test_x), Y:list(test_y)})

train_neural_network(X, Y)