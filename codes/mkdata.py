#coding:utf-8
import json
import re
import codecs
import pickle
import os
import csv

def load(filename):
	with open(filename,'r') as f:
		patt = re.compile(r'"timestamp" : NumberLong\("\d+"\),')
		data = patt.sub('', f.read())
		res = json.loads(data)
	return res

def csvwrite(fn,data):
	with codecs.open(fn,'w') as f:
		writer = csv.writer(f)
		writer.writerows(data)

def mktrain(data):
	train = {}
	train['originalMid']=data['originalMid']
	train['type'] = data['type']
	train['originalWeibo'] = data['originalWeibo']
	train['props'] = []
	ob_cnt = 0; pre_cnt = 0
	ori_t = int(data['originalWeibo']['time'])
	for item in data['props']:
		gap = (int(item['time'])-ori_t)*1.0/3600000
		if 0<= gap< 1: #传播时间小于1h
			train['props'].append(item)
			ob_cnt += 1
		elif 1 <= gap < 3:
			pre_cnt += 1

	train['ob_cnt'] = ob_cnt
	train['pre_cnt'] = pre_cnt

	trainjson = json.dumps(train, encoding='utf-8', ensure_ascii=False)
	return trainjson

def saveFile(filename,data):
	with codecs.open(filename,'w','utf-8') as f:
		f.write(data)




if __name__ == '__main__':
	path = '/media/Data/wjc/WeiboForwardWJC/data/'
	observe = []; predict = []

	for file in os.listdir(path):
		real_path = os.path.join(path,file)
		if os.path.isfile(real_path) == True:
			with codecs.open(real_path,'r','utf-8') as f:
				data = json.loads(f.read())
			train = mktrain(data)
		ftn = '/media/Data/wjc/WeiboForwardWJC/train_1/' + 'train_' + file[6:-5] + '.json'
		saveFile(ftn, train)

	print('done')