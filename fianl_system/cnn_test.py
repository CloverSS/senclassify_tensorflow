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

class CNNTest(object):
	def __init__ (
	  self,file_aim,file_res,file_tensor_model,num_classes):
		file_stopwd="D:/python/data/stopwd.txt"
		#file_aim="D:/python/data/data_tan_own.txt"
		#file_res="D:/python/data/res_cnn_own_1.txt"
		#file_tensor_model="D:/python/model/tensorflow/model_data_cnn_1.ckpt-20"
		sentence_length=200

		stopwdlist=data_handler.stopwordslist(file_stopwd)
		test_data_x,test_data_y=data_handler.data_tovec_w2v(file_aim,[0,0],sentence_length,stopwdlist)
		test_data_x=np.array(test_data_x)

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

		# Training
		# ==================================================
		with tf.Session() as sess:
			new_saver = tf.train.import_meta_graph(file_tensor_model+".meta")  
			new_saver.restore(sess, file_tensor_model)   
			predictions = tf.get_collection('predictions')[0]  
			graph = tf.get_default_graph()
			input_x = graph.get_operation_by_name('input_x').outputs[0]
			input_y = graph.get_operation_by_name('input_y').outputs[0]
			dropout_keep_prob=graph.get_operation_by_name('dropout_keep_prob').outputs[0]

			res=sess.run(predictions, feed_dict={input_x:test_data_x,input_y:test_data_y,dropout_keep_prob:1.0})
			write_res(res,file_res,file_aim)
			print("结果写入：%s"%file_res)