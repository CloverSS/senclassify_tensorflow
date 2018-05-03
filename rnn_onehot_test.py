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
file_aim="D:/python/data/data_tan_neg.txt"
#file_mid="D:/python/data/data_mid.txt"
file_stopwd="D:/python/data/stopwd.txt"
file_dict="D:/python/data/dict.pkl"

def stopwordslist(file_stop):    #加载停用词词典
	stopwords = [line.strip() for line in open(file_stop, 'r', encoding='utf-8').readlines()]  
	return stopwords  
stopwdlist=stopwordslist(file_stopwd)

def line_cutstop_list(line):   #返回分词后句子（list)
	result=jieba.cut(line.strip())
	outstr=[]
	for word in result:  
		if word not in stopwdlist:  
			if word != '\t':  
				outstr.append(word)
	return outstr

def count_lines(file_data):   #返回分词后的句子list （句子与句子构成list，每个句子的分词以空格隔开）
	with open(file_data,"r+",encoding='UTF-8') as raw_dt:
		lines=raw_dt.readlines()
		count=len(lines)
	return count

def data_tovec(file_data,flag,dict):       #特征向量表示，返回向量list
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

pkl_file = open(file_dict, 'rb')
dict=pickle.load(pkl_file)
data=[]
data.extend(data_tovec(file_pos,[1,0],dict))
#data.extend(data_tovec(file_mid,[0,1],dict))
data.extend(data_tovec(file_neg,[0,1],dict))
#data.extend(data_tovec(file_aim,[0,1],dict))
#print(data)
print(len(data))
#random.shuffle(data)  #打乱顺序

#test_size = int(len(data) * 0.2)  #取20%数据为测试数据
data = np.array(data)
#model1 = SelectKBest(chi2, k=200)
#data_x=model1.fit_transform(list(data[:,0]), list(data[:,1]))
data_x=list(data[:,0])
data_y=list(data[:,1])  #卡方过滤

#test_size=len_aim

#train_data_raw_x = np.array(data_x[:-test_size])
#train_data_raw_y = np.array(data_y[:-test_size])
#test_data_x = data_x[-test_size:]
#test_data_y = data_y[-test_size:]
#np.random.seed(10)
#shuffle_indices = np.random.permutation(np.arange(len(data_y)))
#print(train_data_raw_x[0:3])
test_data_x = np.array(data_x)
test_data_y = np.array(data_y)
#print("shape x:",np.array(test_data_x).shape())
#print("shape y:",np.array(test_data_y).shape())

#print(train_data_x[0:3])
#print(train_data_y[0:3])

#print('test_size = {}'.format(test_size))
#print 'size of train_dataset is {}'.format(train_dataset)

#神经网络定义及训练（双隐层网络）

n_input_layer = len(dict)  #输入向量维度
'''n_layer_1 = 100  
n_layer_2 = 100 
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
	
batch_size = 20'''

print("dict len",len(dict))

def write_res(res,res_file_path,aim_file_path,dis):
	with open("D:/python/data/res_1.txt","a+",encoding='UTF-8') as res_f:
		with open(aim_file_path,"r+",encoding='UTF-8') as aim_f:
			lines=aim_f.readlines()
			for num,line in enumerate(lines):
				if res[num+dis]==0:
					res_f.write("正向  "+line+"\n")
				if res[num+dis]==1:
					res_f.write("负向  "+line+"\n")

with tf.Session() as session:
    new_saver = tf.train.import_meta_graph('D:/python/model/tensorflow/model_1.ckpt-20.meta')  
    new_saver.restore(session, 'D:/python/model/tensorflow/model_1.ckpt-20')   
    predict = tf.get_collection('predict')[0]  
    graph = tf.get_default_graph()
    X = graph.get_operation_by_name('X').outputs[0]
    Y = graph.get_operation_by_name('Y').outputs[0]

    correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
    accurqcy = tf.reduce_mean(tf.cast(correct, 'float'))
    res=session.run(tf.argmax(predict,1), feed_dict={X:list(test_data_x), Y:list(test_data_y)})

    write_res(res,"D:/python/data/res_2.txt",file_pos,0)
    dis=count_lines(file_pos)
    write_res(res,"D:/python/data/res_2.txt",file_neg,dis)
        
    print("shape x:",len(test_data_x[0]))
    print("shape y:",len(test_data_y[0]))
    print('准确率: {}'.format(accurqcy.eval({X:list(test_data_x), Y:list(test_data_y)})))
    #等价: print session.run(accuracy, feed_dict={X:list(test_x), Y:list(test_y)})
