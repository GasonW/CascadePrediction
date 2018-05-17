#coding:utf-8

import json
import pickle
import codecs
import csv
import os
import jieba
import jieba.posseg as pseg
from scipy.stats import mode

def csvwrite(fn,data):
	with codecs.open(fn,'w') as f:
		writer = csv.writer(f)
		writer.writerows(data)


def FeatureExtract(data):
	#content feature
	contain_pic = int(bool(data['originalWeibo']['piclist']))

	#poster feature
	n_forwards = len(data['props'])
	userCertify = int(data['originalWeibo']['userCertify']) if data['originalWeibo']['userCertify'] else 0 # 将null值当做0处理
	gender = 0 if data['originalWeibo']['userGender'] == 'male' else 1
	n_follower = int(data['originalWeibo']['userFollowCount']) if data['originalWeibo']['userFollowCount'] else 0
	n_fan = int(data['originalWeibo']['userFanCount']) if data['originalWeibo']['userFanCount'] else 0
	n_weibo = int(data['originalWeibo']['userWeiboCount']) if data['originalWeibo']['userWeiboCount'] else 0
	contain_faces = int(bool(data['originalWeibo']['faces']))
	contain_location = int(bool(data['originalWeibo']['weiboLocation']))
	contain_Music = int(bool(data['originalWeibo']['containsMusic']))
	contain_Video = int(bool(data['originalWeibo']['containsVideo']))
	contain_Url = int(bool(data['originalWeibo']['containsUrl']))

	# share feature
	uCertify_avg = 0; forwards_avg = 0; praise_avg = 0;
	# temperal feature
	gap_avg = 0; ori_t = int(data['originalWeibo']['time'])

	for item in data['props']:
		forwards_avg += int(item['forward']) if item['forward'] else 0
		uCertify_avg += int(item['userCertify']) if item['userCertify'] else 0
		praise_avg += int(item['praise']) if item['praise'] else 0
		this_t = int(item['time'])
		gap_avg += (this_t - ori_t) * 1.0 / 60000 # in minutes

	if n_forwards == 0:
		uCertify_avg = 0.0
		forwards_avg = 0.0
		praise_avg = 0.0
		gap_avg = 1000
	else:
		uCertify_avg = uCertify_avg * 1.0 / n_forwards
		forwards_avg = forwards_avg * 1.0 / n_forwards
		praise_avg = praise_avg * 1.0 /n_forwards
		gap_avg = gap_avg * 1.0 / n_forwards

	features = [contain_pic, n_forwards, userCertify, gender, n_follower,
	n_fan, n_weibo, contain_faces, contain_location, contain_Music, contain_Video,
	contain_Url, uCertify_avg, forwards_avg, praise_avg, gap_avg]

	return features


def jqFeatures(data):
	content = data['originalWeibo']['content']
	wordlist = list(jieba.cut(content))
	# wordflag = pseg.cut(content)

	# -------- content features -------- #
	n_forwards = len(data['props'])
	n_letters = len(content)
	n_words = len(wordlist)
	n_exclamation_m = wordlist.count('！' or '!') # number of !
	n_at_m = wordlist.count('@') # number of @
	n_topic = wordlist.count('#')/2
	n_pictures = len(data['originalWeibo']['piclist'])
	contain_mult_pics = 1 if n_pictures > 1 else 0
	w_exclamation_m = n_exclamation_m * 1.0 / n_letters # weight of !

	n_fan = int(data['originalWeibo']['userFanCount']) if data['originalWeibo']['userFanCount'] else -0.5
	n_weibo = int(data['originalWeibo']['userWeiboCount']) if data['originalWeibo']['userWeiboCount'] else -0.5
	
	# n_quesion_m = wordlist.count('？' or '?') # number of ?
	# contain_pic = int(bool(data['originalWeibo']['piclist']))
	# w_quesion_m = n_quesion_m * 1.0 /n_letters # weight of ?
	# hot_rate = int(data['originalWeibo']['hotrate'])
	# userCertify = int(data['originalWeibo']['userCertify']) if data['originalWeibo']['userCertify'] else -1 # 将null值当做0处理
	# n_follower = int(data['originalWeibo']['userFollowCount']) if data['originalWeibo']['userFollowCount'] else -1
	#n_praise = int(data['originalWeibo']['praise'])
	# description = data['originalWeibo']['userDescription']
	# if description:
	# 	is_official = int(bool('官方' in description))
	# else:
	# 	is_official = 0

	# n_neg = 0; n_pos = 0;
	# for word in wordlist: # compute pos and neg word num
	# 	if word in neglist:
	# 		n_neg += 1
	# 	if word in poslist:
	# 		n_pos += 1

	# ns = 0; n_name = 0;  nt = 0
	# for word in wordflag:
	# 	if word.flag == 'nr':
	# 		n_name += 1
	# 	elif word.flag == 'ns':
	# 		ns += 1
	# 	elif word.flag == 'nt':
	# 		nt += 1

	# ---------- temperal features -------- #
	gap_list = []; ori_t = int(data['originalWeibo']['time'])
	gap_avg = 0;gap_min = 0; gap_max = 0; n_gap_less_mean = 0;n_gap_less_60=0
	gap_avg_last10 = 0; # 观测最近是否在密集转发
	n_gap_mode=0; # 观测密集转发程度

	if len(data['props'])==0:
		gap_avg = 200 # set the gap infity for non-prop  180min for 3hour
		gap_min = 200; gap_max = 200; 
		n_gap_less_mean = 0; n_gap_less_60 = 0

	else:
		for item in data['props']:
			gap_list.append((int(item['time'])-ori_t)*1.0/60000)
		
		gap_list.sort()
		gap_avg = sum(gap_list)/len(data['props'])
		gap_min = gap_list[0]; gap_max = gap_list[-1]
		n_gap_mode = mode(gap_list)[1][0]


		for gap in gap_list:
			if gap < gap_avg:
				n_gap_less_mean += 1
			if gap < 60:
				n_gap_less_60 += 1
		
		max_gap = gap_list[-1] # used to compute the gap between last 10 gap.
		for i,lastgap in enumerate(reversed(gap_list)):
			if i > 10:
				break
			else:
				gap_avg_last10 += max_gap - lastgap
		if len(gap_list) > 10:
			gap_avg_last10 = gap_avg_last10*1.0/10
		else:
			gap_avg_last10 = gap_avg_last10*1.0/len(gap_list)

	# -----------label -----------#
	label = 1 if data['pre_cnt'] > 3*data['ob_cnt'] else 0  # classification label
	# label = data['pre_cnt'] # regression label
	# -----------feature list --------#
	features = [n_forwards, n_letters, n_words, n_exclamation_m, n_at_m, n_topic, n_pictures, 
		contain_mult_pics, w_exclamation_m, n_fan, n_weibo, gap_avg, gap_min, gap_max, 
		n_gap_less_mean, n_gap_less_60, n_gap_mode, gap_avg_last10, label]
	return features 

if __name__ == '__main__':
	
	print("running....")
	path = '/media/Data/wjc/WeiboForwardWJC/train_10/'
	# path = '../data/train_1'
	fwn = './class_features_3_10_3_2.csv' 
	# features = [["contain_pic", "n_forwards", "userCertify", "gender", "n_follower",
	# "n_fan", "n_weibo", "contain_faces", "contain_location", "contain_Music", "contain_Video",
	# "contain_Url", "uCertify_avg", "forwards_avg", "praise_avg", "gap_avg"]]

	features = [["n_forwards", "n_letters", "n_words", "n_exclamation_m", "n_at_m", "n_topic", "n_pictures",
	"contain_mult_pics", "w_exclamation_m", "n_fan", "n_weibo", "gap_avg", "gap_min", "gap_max", 
	"n_gap_less_mean", "n_gap_less_60","n_gap_mode","gap_avg_last10","class"]]
	
	# with codecs.open('../reference/negtive.txt','r','utf-8') as f:
	# 	neglist = f.read().split('\n')
	# with codecs.open('../reference/positive.txt','r','utf-8') as f:
	# 	poslist = f.read().split('\n')

	i = 0
	for file in os.listdir(path):
		print(i); i+=1
		real_path = os.path.join(path,file)
		with codecs.open(real_path,'r','utf-8') as fr:
			data = json.loads(fr.read())
		# features.append(FeatureExtract(data))
		features.append(jqFeatures(data))
	csvwrite(fwn, features)

	print("succeed")
