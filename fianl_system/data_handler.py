import jieba
import jieba.posseg as pseg
import jieba.analyse
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
import numpy as np

def stopwordslist(file_stop):    #加载停用词词典
	stopwords = [line.strip() for line in open(file_stop, 'r', encoding='utf-8').readlines()]  
	return stopwords  

def line_cutstop_str(line,stopwdlist):     #返回分词后句子(str)
	result=jieba.cut(line.strip())   #结巴分词
	outstr=''
	for word in result:  
		if word not in stopwdlist:  
			if word != '\t':  
				outstr += word  
				outstr += " "
	return outstr
		
def line_cutstop_list(line,stopwdlist):   #返回分词后句子（list)
	result=pseg.cut(line.strip())
	outstr=[]
	outflag=[]
	for w in result:  
		if w.word not in stopwdlist:  
			if w.word != '\t':  
				outstr.append(w.word)
				outflag.append(w.flag)
	return outstr,outflag
	
def data_prevocab(file_data,stopwdlist):   #返回分词后的句子list （句子与句子构成list，每个句子的分词以空格隔开）
	jieba.load_userdict("D:/python/data/jieba_dict.txt")
	with open(file_data,"r+",encoding='UTF-8') as raw_dt:
		lines=raw_dt.readlines()
		data_aftcut=[]
		for line in lines:
			data_aftcut.append(line_cutstop_str(line,stopwdlist))
	return data_aftcut

def count_lines(file_data):   #返回分词后的句子list （句子与句子构成list，每个句子的分词以空格隔开）
	with open(file_data,"r+",encoding='UTF-8') as raw_dt:
		lines=raw_dt.readlines()
		count=len(lines)
	return count

def build_vocab(data_befvec,min):     #特征提取，返回字典list
	vectorize_tf = CountVectorizer(max_df=0.9, min_df=min)  #tf-idf,至少出现10次
	vectorize = vectorize_tf.fit_transform(data_befvec)
	return list(vectorize_tf.vocabulary_.keys())  #返回字典

def max_sentence(file_data,sentence_length,stopwdlist):     #统计最大句子长度
	with open(file_data,"r+",encoding='UTF-8') as raw_dt:
		lines=raw_dt.readlines()
		for line in lines:
			result=jieba.cut(line.strip())   #结巴分词
			i=0
			for word in result:  
				if word not in stopwdlist:  
					if word != '\t':
						i+=1
			if i>sentence_length:
				sentence_length=i
	return sentence_length

def data_tovec(file_data,flag,dict,stopwdlist):       #特征向量表示，返回向量list
	jieba.load_userdict("D:/python/data/jieba_dict.txt")
	data=[]
	with open(file_data,"r",encoding='UTF-8') as raw_dt:
		lines=raw_dt.readlines()
		i=0
		for line in lines:
			word_lists,word_flags=line_cutstop_list(line,stopwdlist)
			line_vec=np.zeros(len(dict))
			i=i+1
			for num,word in enumerate(word_lists):
				if word in dict:
					if word_flags[num]=='d':
						line_vec[dict.index(word)]+=2
					elif word_flags[num]=='a':
						line_vec[dict.index(word)]+=2
					elif word_flags[num]=='v':
						line_vec[dict.index(word)]+=1.5
					else:
						line_vec[dict.index(word)]+=1
					#line_vec[dict.index(word)]+=1
			data.append([line_vec,flag])
	return data		

def data_tovec_w2v(file_data,flag,sentence_length,stopwdlist):       #特征向量表示，返回向量list
	jieba.load_userdict("D:/python/data/jieba_dict.txt")
	data_x=[]
	data_y=[]
	model = Word2Vec.load('D:/python/model/word2vec/wiki.zh.text.model')
	with open(file_data,"r",encoding='UTF-8') as raw_dt:
		lines=raw_dt.readlines()
		for line in lines:
			word_lists,_=line_cutstop_list(line,stopwdlist)
			line_vec=np.zeros((sentence_length,400))
			for num,word in enumerate(word_lists):
				if(num>=sentence_length):
					break
				try:
					line_vec[num]=np.array(model[word])
				except:
					pass
			data_x.append(line_vec)
			data_y.append(flag)
	return data_x,data_y		
