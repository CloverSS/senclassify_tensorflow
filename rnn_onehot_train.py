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

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif

file_pos="D:/python/data/data_tan_pos.txt"
file_neg="D:/python/data/data_tan_neg.txt"
file_aim="D:/python/data/data_neg.txt"
#file_mid="D:/python/data/data_mid.txt"
file_stopwd="D:/python/data/stopwd.txt"
file_dict="D:/python/data/dict.pkl"

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
	
def data_prevocab(file_data):   #返回分词后的句子list （句子与句子构成list，每个句子的分词以空格隔开）
	with open(file_data,"r+",encoding='UTF-8') as raw_dt:
		lines=raw_dt.readlines()
		data_aftcut=[]
		for line in lines:
			data_aftcut.append(line_cutstop_str(line))
	return data_aftcut

def count_lines(file_data):   #返回分词后的句子list （句子与句子构成list，每个句子的分词以空格隔开）
	with open(file_data,"r+",encoding='UTF-8') as raw_dt:
		lines=raw_dt.readlines()
		count=len(lines)
	return count

def build_vocab(data_befvec):     #特征提取，返回字典list
	vectorize_tf = CountVectorizer(max_df=0.9, min_df=10)  #tf-idf,至少出现10次
	vectorize = vectorize_tf.fit_transform(data_befvec)
	return list(vectorize_tf.vocabulary_.keys())  #返回字典
	
def save_dict(dict,file_path):
	output = open(file_path, 'wb')
	pickle.dump(dict, output, -1)
	output.close()
	
data_befvec=data_prevocab(file_pos)
data_befvec+=data_prevocab(file_neg)
#data_befvec+=data_prevocab(file_aim)
len_aim=count_lines(file_aim)
dict=build_vocab(data_befvec)
save_dict(dict,file_dict)
print(len(dict))
	
def data_tovec(file_data,flag):       #特征向量表示，返回向量list
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
data.extend(data_tovec(file_pos,[1,0]))
#data.extend(data_tovec(file_mid,[0,1,0]))
data.extend(data_tovec(file_neg,[0,1]))
data.extend(data_tovec(file_aim,[0,1]))
#print(data)
print(len(data))
#random.shuffle(data)  #打乱顺序

test_size = int(len(data) * 0.2)  #取20%数据为测试数据
data = np.array(data)
#model1 = SelectKBest(chi2, k=200)
#data_x=model1.fit_transform(list(data[:,0]), list(data[:,1]))
data_x=list(data[:,0])
data_y=list(data[:,1])  #卡方过滤

test_size=len_aim

train_data_raw_x = np.array(data_x[:-test_size])
train_data_raw_y = np.array(data_y[:-test_size])
test_data_x = data_x[-test_size:]
test_data_y = data_y[-test_size:]
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(train_data_raw_x)))
print(train_data_raw_x[0:3])
train_data_x = train_data_raw_x[shuffle_indices]
train_data_y = train_data_raw_y[shuffle_indices]
#print(train_data_x[0:3])
#print(train_data_y[0:3])

print('test_size = {}'.format(test_size))
#print 'size of train_dataset is {}'.format(train_dataset)

#神经网络定义及训练（双隐层网络）

n_input_layer = len(dict)  #输入向量维度
n_layer_1 = 500  
n_layer_2 = 500 
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
	layer_output=define_layer(layer_2,n_layer_2,n_output_layer)
	return layer_output
	
batch_size = 20
X = tf.placeholder('float', [None, n_input_layer],name='X')  #占位符
Y = tf.placeholder('float',name='Y')

#使用数据训练神经网络
def train_neural_network(X, Y):
    predict = define_network(X)    #定义神经网络
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=Y))   #定义代价函数，这里用交叉熵损失实现
    #optimizer = tf.train.AdamOptimizer().minimize(cost_func)   #adam优化器
    optimizer=tf.train.GradientDescentOptimizer(0.1).minimize(cost_func)
	
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
        '''tf.add_to_collection('predict', predict)
        saver = tf.train.Saver(tf.all_variables())		
        saver_path = saver.save(session, "D:/python/model/tensorflow/model_tan_1.ckpt",global_step=20)
        print("saveer path:",saver_path)'''
        correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
        accurqcy = tf.reduce_mean(tf.cast(correct, 'float'))
        res=session.run(tf.argmax(predict,1), feed_dict={X:list(test_data_x), Y:list(test_data_y)})

        '''with open("D:/python/data/res_6.txt","a+",encoding='UTF-8') as res_f:
            with open(file_aim,"r+",encoding='UTF-8') as aim_f:
                lines=aim_f.readlines()
                for num,line in enumerate(lines):
                    if res[num]==0:
                        res_f.write("正向  "+line+"\n")
                    if res[num]==1:
                        res_f.write("负向  "+line+"\n")       '''
        print('准确率: {}'.format(accurqcy.eval({X:list(test_data_x), Y:list(test_data_y)})))
        #等价: print session.run(accuracy, feed_dict={X:list(test_x), Y:list(test_y)})
        

train_neural_network(X, Y)