import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as pyplot
from scipy import stats 
import warnings 
import time 
from sklearn.metrics import log_loss

warnings.filterwarnings('ignore')

def avgPraiseConvert(data):

	data.loc[(data['avgPraise'] <= 1), 'avgPraise'] = 0
	data.loc[(data['avgPraise'] <=3)&(data['avgPraise'] >1), 'avgPraise' ] = 1
	data.loc[(data['avgPraise'] <=6)&(data['avgPraise'] >3), 'avgPraise' ] = 2
	data.loc[(data['avgPraise'] <=10)&(data['avgPraise'] >6), 'avgPraise' ] = 3
	data.loc[(data['avgPraise'] <=20)&(data['avgPraise'] >10), 'avgPraise' ] = 4
	data.loc[(data['avgPraise'] <=39)&(data['avgPraise'] >20), 'avgPraise' ] = 5 
	data.loc[(data['avgPraise'] <=89)&(data['avgPraise'] >39), 'avgPraise' ] = 6
	data.loc[(data['avgPraise'] <=282)&(data['avgPraise'] >89), 'avgPraise' ] = 7
	data.loc[(data['avgPraise'] >282), 'avgPraise' ] = 8

	return data


def avgFacntConvert(data):

	data.loc[(data['avgFacnt'] <= 39147), 'avgFacnt'] = 0
	data.loc[(data['avgFacnt'] <= 115078)&(data['avgFacnt'] >39147), 'avgFacnt' ] = 1
	data.loc[(data['avgFacnt'] <= 240000)&(data['avgFacnt'] >115078), 'avgFacnt' ] = 2
	data.loc[(data['avgFacnt'] <= 455850)&(data['avgFacnt'] >240000), 'avgFacnt' ] = 3
	data.loc[(data['avgFacnt'] <= 809278)&(data['avgFacnt'] >455850), 'avgFacnt' ] = 4
	data.loc[(data['avgFacnt'] <= 1288691)&(data['avgFacnt'] >809278), 'avgFacnt' ] = 5
	data.loc[(data['avgFacnt'] <= 2046127)&(data['avgFacnt'] >1288691), 'avgFacnt' ] = 6
	data.loc[(data['avgFacnt'] <= 3142127)&(data['avgFacnt'] >2046127), 'avgFacnt' ] = 7
	data.loc[(data['avgFacnt'] <= 5139460)&(data['avgFacnt'] >3142127), 'avgFacnt' ] = 8
	data.loc[(data['avgFacnt'] >5139460), 'avgFacnt' ] = 9
	return data


def add_attri(data):
	var3 = 'hasOriginOrg'
	data[var3] = 0
	data.loc[data['originOrgCnt'] != 0, var3] = 1
	# print(data[[var3, var2]].groupby(var3, as_index=False).mean())
	var3 = 'hasClarity'
	data[var3] = 0
	data.loc[data['clarityRate'] != 0, var3] = 1
	# var3 = 'hasThirdParty'
	# data[var3] = 0
	# data.loc[data['thirdPartyRate'] != 0, var3] = 1
	# var3 = 'hasQM'
	# data[var3] = 0
	# data.loc[data['rateQMs'] != 0, var3] = 1
	return data


def drawHeatmap(data, kwd):
	#绘制热力图
	corrmat = data.astype(float).corr(method = kwd)
	pyplot.subplots(figsize=(10, 8))
	sns.heatmap(corrmat, vmax=.8, annot = True, annot_kws={'size': 5}, square=True)
	pyplot.show()
	return data

'''
pos_data = pd.read_table('./truth_eventfeature.txt', delim_whitespace = True)
neg_data = pd.read_table('./rumor_eventfeature.txt', delim_whitespace = True)
all_data = pd.concat([pos_data, neg_data], ignore_index = True)
all_data.to_csv('./eventfeatures', sep = " ", index = False)
'''
'''
pos_data = pd.read_table('./truth_singleweibofeature.txt', delim_whitespace = True)
neg_data = pd.read_table('./rumor_singleweibofeature.txt', delim_whitespace = True)
all_data = pd.concat([pos_data, neg_data], ignore_index = True)
all_data.to_csv('./singleweibofeatures', sep = " ", index = False)
'''

data = pd.read_csv('./eventfeatures', delim_whitespace = True)
# print(data.shape)
# print(data.describe())
# print(data.dtypes)
# print(data.head(3))
# data.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
# pyplot.show()
# pd.scatter_matrix(data)
# pyplot.show()
features = ['avgPraise','rateQM','rateQMs','rateEM','rateEMs','disAuth','rateAuth','disAt','disLoc','avgFacnt','ratePerApp','rateImg','rateImgs','rateImgs2','rateDisImg','clarityRate','thirdPartyRate','fan1000Rate','originOrgCnt']
var2 = 'label'
# print(data[['disAuth', var2]].groupby('disAuth', as_index=False).mean())
# for var1 in features:
	# print(pd.concat([(data[[var1, var2]].groupby(var2, as_index=False).mean(), data[[var1, var2]].groupby(var2, as_index=False).std()], axis = 1))


'''
data['categoricalFacnt'] = pd.qcut(data['avgFacnt'], 10, duplicates = 'drop')
print(data['categoricalFacnt'].value_counts())
print(data[['categoricalFacnt', var2]].groupby('categoricalFacnt', as_index=False).mean())
'''
data.pipe(avgPraiseConvert)\
	.pipe(avgFacntConvert)\
	.pipe(add_attri)\
	.pipe(drawHeatmap, 'pearson')
data.to_csv('./processed_features.csv', index = False)
