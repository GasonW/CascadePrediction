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

import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.inf)  


# --------- PREPROCESS DATA ------------ #
x = pd.read_csv('../data/processed_features.csv', header = None, sep = ',', names = ["contain_pic", "n_forwards", "userCertify", "gender", "n_follower",
	"n_fan", "n_weibo", "contain_faces", "contain_location", "contain_Music", "contain_Video",
	"contain_Url", "uCertify_avg", "forwards_avg", "praise_avg", "gap_avg"])
y = pd.read_csv('../data/y_1-4.csv', header = None, sep = ',', names = ['label'])

# x = preprocessing.scale(x) # scale x into [0,1]

assert x.shape[0] == y.shape[0]
data = pd.concat([x,y], axis = 1)
# print(data.describe())

# feat = ['forwards_avg', 'label']
# print(data[feat].groupby('label',as_index = False).mean())
"""
data['n_forwards'] = np.log1p(data['n_forwards'])
data['n_follower'] = np.log1p(data['n_follower'])
data['n_fan'] = np.log1p(data['n_fan'])
data['n_weibo'] = np.log1p(data['n_weibo'])
data['forwards_avg'] = np.log1p(data['forwards_avg'])
data['praise_avg'] = np.log1p(data['praise_avg'])
data = data.drop(['contain_Music','contain_Video','contain_Url'], axis = 1)
"""

data = shuffle(data)
train, test = train_test_split(data, test_size = 0.2)


n_train = train.shape[0]
n_test = test.shape[0]

SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
# kf = KFold(n_train, n_folds=NFOLDS, random_state=SEED)
skf = StratifiedKFold(n_splits = NFOLDS)

# ------------- CLASS TO EXTEND THE SKLEARN CLASSIFIER --------- #
class SklearnHelper(object):
	def __init__(self, clf, seed=0, params=None):
		params['random_state'] = seed
		self.clf = clf(**params)

	def train(self, x_train, y_train):
		self.clf.fit(x_train, y_train)

	def predict(self, x):
		return self.clf.predict(x)

	def predict_proba(self, x):
		return self.clf.predict_proba(x)

	def feature_importances(self, x, y):
		return self.clf.fit(x,y).feature_importances_

# class to extend XGboost classifier
class XgbWrapper(object):
	def __init__(self, seed=0, params = None):
		self.param = params
		self.param['seed'] = seed
		self.nrounds = params.pop('nrounds',250)

	def train(self, x_train, y_train):
		dtrain = xgb.DMatrix(x_train, label=y_train)
		self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

	def predict(self, x):
		return self.gbdt.predict(xgb.DMatrix(x))

	def feature_importances(self):
		return self.gbdt.feature_importances_

def get_oof(clf, x_train, y_train, x_test, isXgb = False):
	oof_train = np.zeros((n_train,))
	oof_test = np.zeros((n_test,))
	oof_test_skf = np.empty((NFOLDS, n_test))
	precisions = 0

	# print('ytrain:',y_train)
	# for i, (train_index, val_index) in enumerate(kf):
	for i,(train_index, val_index) in enumerate(skf.split(x_train,y_train)):
		x_tr = x_train[train_index]
		y_tr = y_train[train_index]
		x_val = x_train[val_index]
		clf.train(x_tr,y_tr)

		if isXgb:
			oof_train[val_index] = clf.predict(x_val)
			result = oof_train[val_index]
			# result = [1 if rst > 0.5 else 0 for rst in result]
			result[result>0.5] = 1
			result[result<=0.5] = 0
			precisions += precision_score(y_train[val_index],result)
			oof_test_skf[i, :] = clf.predict(x_test)

		else:
			oof_train[val_index] = clf.predict(x_val)
			result = oof_train[val_index]
			#result = [1 if rst > 0.5 else 0 for rst in result]
			result[result>0.5] = 1
			result[result<=0.5] = 0
			precisions += precision_score(y_train[val_index], result)
			oof_test_skf[i, :] = clf.predict(x_test)
		print(i,':',precision_score(y_train[val_index],result))
	oof_test[:] = oof_test_skf.mean(axis=0)
	# oof_test = [1 if otest > 0.5 else 0 for otest in oof_test]
	# oof_train = [1 if otrain > 0.5 else 0 for otrain in oof_train]
	
	oof_test[oof_test > 0.5] = 1
	oof_test[oof_test <= 0.5] = 0
	oof_train[oof_train > 0.5] = 1
	oof_train[oof_train <= 0.5] = 0

	precisions = precisions / NFOLDS
	print(precisions)
	return oof_train.reshape(-1,1), oof_test.reshape(-1,1)

def metrics(y_test, predictions):
	print('log_loss: ', log_loss(y_test,predictions))
	predictions[predictions > 0.5] = 1
	predictions[predictions <=0.5] = 0
	print("precision_score: ", precision_score(y_test,predictions))
	print("recall: ",recall_score(y_test,predictions))


# Random Forest parameters
rf_params = {
	'n_jobs' : -1,
	'n_estimators': 300,
	'warm_start': True,
	# max_features : 0.2,
	'max_depth': 6,
	'min_samples_leaf':2,
	'max_features' : 'sqrt',
	'verbose': 0
}

# Extra Trees Parameters
et_params = {
	'n_jobs':-1,
	'n_estimators':500,
	# max_features:0.5,
	'max_depth':8,
	'min_samples_leaf':2,
	'verbose':0
}

# Gradient Boosting parameters
gb_params = {
	'n_estimators':300,
	'max_features':0.3,
	'max_depth':8,
	'min_samples_leaf':2,
	'verbose':0
}

# AdaBoost parameters
ada_params = {
	'n_estimators':300,
	'learning_rate':0.1
}

# LogisticRegression paramters
lr_params = {
	'penalty':'l2',
	'dual':False,
	'tol':0.001,
	'C':1
}

# Support Vector Classifier parameters
svc_params = {
	'kernel':'rbf',
	'C':0.025,
	'probability':True
}

# XGBoost parameters
xgb_params = {
	# 'seed': 0,
	# 'colsample_bytree': 0.7,
	# 'silent': 1,
	# 'subsample': 0.7,
	# 'learning_rate': 0.075,
	# 'objective': 'reg:linear',
	# 'max_depth': 7,
	# 'num_parallel_tree': 1,
	# 'min_child_weight': 1,
	# 'gamma': 1,
	# 'eval_metric': 'error',
	# 'nrounds': 350

	'learning_rate' : 0.1,
	'n_estimators': 200,
	'max_depth' : 7,
	'min_child_weight' : 2,
	'gamma' : 0.7,                        
	'subsample' : 0.8,
	'colsample_bytree' : 0.5,
	'objective': 'binary:logistic',
	'nthread' : -1,
	'scale_pos_weight':1,
	'silent' : 1,
	'eval_metric':'logloss'	
}


# ------- Create 5 objects that represent our 4 models --------- #
xg = XgbWrapper(seed = SEED, params = xgb_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed = SEED, params=et_params)
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
lr = SklearnHelper(clf=LogisticRegression, seed=SEED, params = lr_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

# ------- Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models --------- #
# print(train.dtypes)
# print('train count: ', train['label'].value_counts())
y_train = train['label'].ravel()
train = train.drop(['label'], axis=1)
x_train = train.values
# x_train = preprocessing.scale(train.values) # Creates an array of the train data
# print('train scale :', x_train[1:3])


# print('test count: ', test['label'].value_counts())
# print('test neg:pos', test[])
y_test = test['label'].ravel()
test = test.drop(['label'], axis=1) 
# x_test = preprocessing.scale(test.values) # Creats an array of the test data
x_test = test.values
# print('test scale :', x_test[1:3])



# ------------
# Create our OOF train and test predictions. These base results will be used as new features
print("precision on cv")
xg_oof_train, xg_oof_test = get_oof(xg, x_train, y_train, x_test, isXgb = True) 
# et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)
# lr_oof_train, lr_oof_test = get_oof(lr, x_train, y_train, x_test) # LogisticRegression 
# rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test)  # Random Forest
# ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
# gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
# svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier
# et.train(x_train, y_train)
# result = et.predict(x_test)
# print('precision:', precision_score(y_test,result))
# print('recall:',recall_score(y_test,result))
# print(str(y_test))
# print(str(result))

print("precision on test data")
# print('ytest:', y_test)
# print('predict', et_oof_test)
print('xgboost train precision:', precision_score(y_train,xg_oof_train),'recall:',recall_score(y_train,xg_oof_train))
print('xgboost precision: ', precision_score(y_test,xg_oof_test),'recall:',recall_score(y_test,xg_oof_test))
# print('extra tree: ', precision_score(y_test,et_oof_test))
# print('LR: ', precision_score(y_test,lr_oof_test))
# print('Random Forest: ', precision_score(y_test,rf_oof_test))
# print('AdaBoost: ', precision_score(y_test,ada_oof_test))
# print('Gradient Boost: ', precision_score(y_test,gb_oof_test))
# print('SVC: ', precision_score(y_test,svc_oof_test))
print("Training is complete")


















