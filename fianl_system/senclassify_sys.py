from rnn_classify import BpClassify
from rnn_test import BpTest
from cnn_classify import CNNClassify
from cnn_test import CNNTest
import os

print("            文本情绪分析工具              ")
print("******************************************")
print("请输入序号选择要使用的模式")
print("1、标准模式(CNN)  2、快速模式(BP)")
mode=input()
while mode!="1" and mode!="2":
	print("输入不符合规范，请重新输入")
	mode=input()
if mode=="1":  #cnn算法
	print("请输入序号选择模型训练方式")
	print("1、自定义标注数据集   2、使用默认模型")
	train=input()
	while train!="1" and train!="2":
		print("输入不符合规范，请重新输入")
		train=input()
	print("请输入分类类别数(2/3)")
	num_classes=input()
	while num_classes!="2" and num_classes!="3":
		print("输入不符合规范，请重新输入")
		num_classes=input()
	if train=="1": #自定义数据集
		file_aim=input("请输入目标数据文件(每个文本一行)：")
		while os.path.isfile(file_aim)==False:
			file_aim=input("文件不存在，请重新输入：")
		file_res=input("请输入分析结果文件地址：")
		file_pos=input("请输入正向标注数据集地址(每个文本一行)：")
		while os.path.isfile(file_pos)==False:
			file_pos=input("文件不存在，请重新输入：")
		file_neg=input("请输入负向标注数据集地址(每个文本一行)：")
		while os.path.isfile(file_neg)==False:
			file_neg=input("文件不存在，请重新输入：")
		if num_classes=="2":  #二分类
			senclass=CNNClassify(file_pos=file_pos,file_neg=file_neg,file_aim=file_aim,
			  file_res=file_res,num_classes=int(num_classes))
		else :  #三分类
			file_mid=input("请输入中性标注数据集地址(每个文本一行)：")
			while os.path.isfile(file_mid)==False:
				file_mid=input("文件不存在，请重新输入：")
			senclass=CNNClassify(file_pos=file_pos,file_neg=file_neg,file_aim=file_aim,
			  file_res=file_res,num_classes=int(num_classes),file_mid=file_mid)

	else: #使用默认模型
		file_aim=input("请输入目标数据文件(每个文本一行)：")
		while os.path.isfile(file_aim)==False:
			file_aim=input("文件不存在，请重新输入：")
		file_res=input("请输入分析结果文件地址：")
		if num_classes=="2":  #二分类
			senclass=CNNTest(file_aim=file_aim,file_res=file_res,
			  file_tensor_model="D:/python/model/tensorflow/model_data_cnn_1.ckpt-20",
			  num_classes=int(num_classes))
		else :  #三分类
			senclass=CNNTest(file_aim=file_aim,file_res=file_res,
			  file_tensor_model="D:/python/model/tensorflow/model_data_cnn_1.ckpt-20",
			  num_classes=int(num_classes))


else:   #bp算法
	print("请输入序号选择模型训练方式")
	print("1、自定义标注数据集   2、使用默认模型")
	train=input()
	while train!="1" and train!="2":
		print("输入不符合规范，请重新输入")
		train=input()
	print("请输入分类类别数(2/3)")
	num_classes=input()
	while num_classes!="2" and num_classes!="3":
		print("输入不符合规范，请重新输入")
		num_classes=input()
	if train=="1": #自定义数据集
		file_aim=input("请输入目标数据文件(每个文本一行)：")
		while os.path.isfile(file_aim)==False:
			file_aim=input("文件不存在，请重新输入：")
		file_res=input("请输入分析结果文件地址：")
		file_pos=input("请输入正向标注数据集地址(每个文本一行)：")
		while os.path.isfile(file_pos)==False:
			file_pos=input("文件不存在，请重新输入：")
		file_neg=input("请输入负向标注数据集地址(每个文本一行)：")
		while os.path.isfile(file_neg)==False:
			file_neg=input("文件不存在，请重新输入：")
		if num_classes=="2":  #二分类
			senclass=BpClassify(file_pos=file_pos,file_neg=file_neg,file_aim=file_aim,
			  file_res=file_res,num_classes=int(num_classes))
		else :  #三分类
			file_mid=input("请输入中性标注数据集地址(每个文本一行)：")
			while os.path.isfile(file_mid)==False:
				file_mid=input("文件不存在，请重新输入：")
			senclass=BpClassify(file_pos=file_pos,file_neg=file_neg,file_aim=file_aim,
			  file_res=file_res,num_classes=int(num_classes),file_mid=file_mid)

	else: #使用默认模型
		file_aim=input("请输入目标数据文件(每个文本一行)：")
		while os.path.isfile(file_aim)==False:
			file_aim=input("文件不存在，请重新输入：")
		file_res=input("请输入分析结果文件地址：")
		if num_classes=="2":  #二分类
			senclass=BpTest(file_aim=file_aim,file_res=file_res,
			  file_dict="D:/python/data/dict_tan.pkl",
			  file_tensor_model="D:/python/model/tensorflow/model_tan_1.ckpt-20",
			  num_classes=int(num_classes))
		else :  #三分类
			senclass=BpTest(file_aim=file_aim,file_res=file_res,
			  file_dict="D:/python/data/dict_class_3.pkl",
			  file_tensor_model="D:/python/model/tensorflow/model_class_3.ckpt-20",
			  num_classes=int(num_classes))