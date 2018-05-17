#coding:utf-8

import json
import pickle
import re
import codecs
import os


path = '/media/Data/wjc/WeiboForwardWJC/data/'

def preprocess(): # split a single json file into several files.
	print('...Preprocessing...')
	with open("splittest.json",'r') as f:
		s = '';i = 0
		for line in f:
			if '&##&*^&' in line: #'&w##&j*^&c':
				s = s.split(',', 1)
				s = '{' + ''.join(s[1])
				fn = 'weibo_' + str(i) + '.json'
				with open(fn,'w') as fw:
					fw.write(s)
				s = '';i = i+1
			s += line
		# s = f.read()
		# s = s.split('&w##&j*^&c')
		# for i, item in enumerate(s[:-1]):
		# 	item = item.split(',',1)
		# 	item = '{' + ''.join(item[1])
		# 	fn = 'weibo_' + str(i) + '.json'
		# 	with open(fn,'w') as f:
		# 		f.write(item)

def clean():
	for i in range(1,2):#29362):
		fn = path + "weibo_" + str(i) + ".json"
		with codecs.open(fn,'r','utf-8') as fr:
			try:
				patt = re.compile(r'"timestamp" : NumberLong\("\d+"\),')
				data = patt.sub('', fr.read())
			except:
				print(str(i) + " failed")
				break
		with codecs.open(fn,'w','utf-8') as fw:
			fw.write(data)

def statics():
	weibo_forwards = []; faild = []
	for i in range(0,29362):
		fn = path + "weibo_" + str(i) + ".json"
		with codecs.open(fn,'r') as f:
			try:
				res = json.loads(f.read())
				weibo_forwards.append(len(res['props']))
			except:
				print(str(i) + ' failed')
				break	
	weibo_forwards.sort(reverse = True)
	with open("weibo_forwards.pkl",'wb') as f:
		pickle.dump(weibo_forwards, f)

def removeNeg():
	#ignoreList = []
	le1_c = 0; le10_c = 0;re_c = 0

	for file in os.listdir(path):
		real_path = os.path.join(path,file)
		if os.path.isfile(real_path) == True:
			with codecs.open(real_path,'r','utf-8') as f:
				data = json.loads(f.read())

			flag = 0; dropPopsIndex = []
			ori_t = int(data['originalWeibo']['time'])
			
			for j,item in enumerate(data['props']):
				this_t = int(item['time'])
				gap = (this_t - ori_t)*1.0/3600000
				if gap < 0:
					if gap >= -1:
						le1_c += 1
						item['time'] = ori_t #修正gap > -1的为0
					elif -10 <= gap <-1:
						le10_c += 1
						dropPopsIndex.append(j) #drop掉gap在10h以内的prop drop掉
					else:
						flag = 1; #将gap>10h的事件ignore掉
						break
			if flag == 1:
				#ignoreList.append(i)
				re_c += 1
				os.remove(real_path)
				continue

			print('less than 1:', str(le1_c))
			print('less than 10:',str(le10_c))
			for l,k in enumerate(dropPopsIndex):
				data['props'].pop(k-l)

			datajson = json.dumps(data,encoding='utf-8', ensure_ascii=False)
			with codecs.open(real_path,'w','utf-8') as fw:
				fw.write(datajson)
	print('removed ',str(re_c))
	# with open("ignoreList.pkl",'wb') as f:
	# 	pickle.dump(ignoreList, f)


def negativeCount():
	neg_weight = []
	neg_1 = 0; neg_10 = 0; neg_24 = 0;neg_l24 = 0
	for i in range(0,50):#29362):
		fn = path + "weibo_" + str(i) + ".json"
		neg_count = 0;#每条事件中负gap所占百分比
		with codecs.open(fn,'r') as f:
			data = json.loads(f.read())

		ori_t = int(data['originalWeibo']['time'])
		this_len = len(data['props']) + 0.001
		#print(this_len)

		for item in data['props']:
			this_t = int(item['time'])
			gap = (this_t - ori_t)*1.0/3600000
			if gap < 0:
			 	neg_count += 1
				if -1 <= gap:
					neg_1 += 1
				elif -10 <= gap <-1:
					neg_10 += 1
				elif -24 <= gap < -10:
					neg_24 += 1
				else:
					neg_l24 += 1
	print('neg1:',neg_1)
	print('neg_10',neg_10)
	print('neg_24',neg_24)
	print('neg_l24',neg_l24)
	# 	neg_weight.append(neg_count*1.0/this_len)

	# with open("negtiveGapWeight.pkl",'wb') as f:
	# 	pickle.dump(neg_weight,f)


def timeDistribution():
	avg_one = []; sum_one = 0;avg_all = 0; len_all = 0;
	hist = []; leq0 = 0; laq5 = 0

	with open('ignoreList.pkl','rb') as f: #load ignoreList
		ignoreList = pickle.load(f) 
	for i in range(0,29362):
		if i in ignoreList:
			continue
			
		fn = path + "weibo_" + str(i) + ".json"
		with codecs.open(fn,'r') as f:
			data = json.loads(f.read())

		ori_t = int(data['originalWeibo']['time'])

		for item in data['props']:
			this_t = int(item['time'])
			gap = (this_t - ori_t)*1.0/3600000 # hour
			if gap < 0:
				print('file ' + str(i) + '<0, it is ' + str(gap))
				leq0 += 1

			hist.append(gap)
			sum_one += gap
		length = len(data['props']) + 1
		avg_one.append(sum_one/length)
		sum_one = 0;
	print('# less than 0: ' + str(leq0))
	print('# larger than 500: ' + str(laq5))
	print('avg_all : ' + str(avg_all))
	print('sum len: ' + str(len_all))
	with open("timeDistAvg.pkl",'wb') as f:
		pickle.dump(avg_one, f)
	with open("timeDist.pkl",'wb') as f:
		pickle.dump(hist, f)

if __name__ == '__main__':
	#preprocess()
	#statics()
	#clean()
	#timeDistribution()
	#negativeCount()
	removeNeg()










