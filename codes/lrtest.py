#coding:utf-8
import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb 
# import lightgbm as lbg 
from xgboost.sklearn import XGBClassifier
# import seaborn as sns

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
							GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sklearn import datasets

import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)  


names = ["contain_pic", "n_forwards", "userCertify", "gender", "n_follower",
	"n_fan", "n_weibo", "contain_faces", "contain_location", "contain_Music", "contain_Video",
	"contain_Url", "uCertify_avg", "forwards_avg", "praise_avg", "gap_avg"]

x = pd.read_csv('../data/processed_features.csv', header = None, sep = ',', names = names)
y = pd.read_csv('../data/y_1.csv', header = None, sep = ',', names = ['label'])

assert x.shape[0] == y.shape[0]
data = pd.concat([x,y], axis = 1)
data = shuffle(data)


# --- preprocess features --- #
data['n_forwards'] = np.log1p(data['n_forwards'])
data['n_follower'] = np.log1p(data['n_follower'])
data['n_fan'] = np.log1p(data['n_fan'])
data['n_weibo'] = np.log1p(data['n_weibo'])
data['forwards_avg'] = np.log1p(data['forwards_avg'])
data['praise_avg'] = np.log1p(data['praise_avg'])
data = data.drop(['contain_Music','contain_Video','contain_Url'], axis = 1)

"""
# --- standard data --- #
sc = StandardScaler()
sc.fit(data[names])
data[names] = sc.transform(data[names])
"""


train, test = train_test_split(data, test_size = 0.2)

# --- mk x and y --# 
y_train = train['label']#.ravel()
train = train.drop(['label'], axis=1)
x_train = train.values


y_test = test['label'].ravel()
test = test.drop(['label'], axis=1) 
x_test = test.values


rf_params = {
	# 'n_jobs' : -1,
	# 'n_estimators': 500,
	# 'warm_start': True,
	# #'max_features' : 0.2,
	 'max_depth': 8,
	# 'min_samples_leaf':2,
	# 'max_features' : 'sqrt',
	# 'verbose': 0
	# 'oob_score': True,
	# 'random_state':42
}

lr = LogisticRegression(C = 1000)
lr.fit(x_train, y_train)
test_result = lr.predict(x_test)
train_result = lr.predict(x_train)
# rf = RandomForestClassifier()#**rf_params)
# rf.fit(x_train,y_train)
# train_result = rf.predict(x_train)
# test_result = rf.predict(x_test)


# print(y_test,sum(y_test))
# print(test_result,sum(test_result))
print(sum(train_result))
print(sum(test_result))
print('pos_train:',sum(y_train), 'neg:',len(y_train)-sum(y_train))
print('pos:',sum(y_test),'neg: ', len(y_test)-sum(y_test))
print('train precision:',precision_score(y_train,train_result))
print('train recall:', recall_score(y_train,train_result))
print('test precision:',precision_score(y_test,test_result))
print('test recall:',recall_score(y_test,test_result))


























