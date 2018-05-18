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

class BpTest(object):
	def __init__ (
	  self,file_aim,file_res,file_dict,file_chi2_model,file_tensor_model,num_classes):
	  
		#file_aim="D:/python/data/data_tan_test_2.txt"
		#file_tensor_model="D:/python/model/tensorflow/model_tan_1.ckpt-20"
		#file_res="D:/python/data/res_tan_test_2.txt"
		file_stopwd="D:/python/data/stopwd.txt"
		#file_dict="D:/python/data/dict_tan.pkl"

		stopwdlist=data_handler.stopwordslist(file_stopwd)
		pkl_file = open(file_dict, 'rb')
		dict=pickle.load(pkl_file)
		data=[]
		data.extend(data_handler.data_tovec(file_aim,[0,0],dict,stopwdlist))
		print(len(data))
		data = np.array(data)
		pkl_file2 = open(file_chi2_model, 'rb')
		model1=pickle.load(pkl_file2)
		data_x=model1.transform(list(data[:,0]))
		#data_x=list(data[:,0])

		print("dict len",len(dict))

		def write_res(res,res_file_path,aim_file_path):
			with open(res_file_path,"a+",encoding='UTF-8') as res_f:
				with open(aim_file_path,"r+",encoding='UTF-8') as aim_f:
					lines=aim_f.readlines()
					for num,line in enumerate(lines):
						if res[num]==0:
							res_f.write("正向  "+line+"\n")
						if res[num]==1:
							res_f.write("负向  "+line+"\n")
						if res[num]==2:
							res_f.write("中性  "+line+"\n")

		with tf.Session() as session:
			new_saver = tf.train.import_meta_graph(file_tensor_model+".meta")  
			new_saver.restore(session, file_tensor_model)   
			predict = tf.get_collection('predict')[0]  
			graph = tf.get_default_graph()
			X = graph.get_operation_by_name('X').outputs[0]
			Y = graph.get_operation_by_name('Y').outputs[0]

			res=session.run(tf.argmax(predict,1), feed_dict={X:list(data_x)})
			write_res(res,file_res,file_aim)
			print("结果写入：%s"%file_res)
			tf.reset_default_graph()
