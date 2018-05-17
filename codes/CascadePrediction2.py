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
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

# ---------- PREPROCESS DATA ----------- #
data = pd.read_csv('../data/class_features_1_3_3.csv')

data['n_forwards'] = np.log1p(data['n_forwards'])
data['n_letters'] = np.log1p(data['n_letters'])
data['n_words'] = np.log1p(data['n_words'])
# data['n_follower'] = np.log1p(data['n_follower'])
data['n_fan'] = np.log1p(data['n_fan'])
data['n_weibo'] = np.log1p(data['n_weibo'])
data['gap_avg'] = np.log1p(data['gap_avg'])
data['gap_min'] = np.log1p(data['gap_min'])
data['gap_max'] = np.log1p(data['gap_max'])
data['n_gap_less_mean'] = np.log1p(data['n_gap_less_mean'])
data['n_gap_less_60'] = np.log1p(data['n_gap_less_60'])

data = shuffle(data)
y = data['class'].ravel()
x = data.drop(['class'], axis=1)

# balance positive and negative data
print('Before balancing: 0:1 = %s:%s'%(np.sum(y==0),np.sum(y==1)))
sm = SMOTE(kind='regular')
x_res,y_res = sm.fit_sample(x,y)
print('After balancing: 0:1 = %s:%s'%(np.sum(y_res==0),np.sum(y_res==1)))

# split train and test data
x_train, x_test, y_train, y_test = train_test_split(x_res,y_res,test_size=0.2)

n_train = x_train.shape[0]
n_test = x_test.shape[0]

# some variables
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
	precisions = 0; recall = 0

	# for i, (train_index, val_index) in enumerate(kf):
	for i,(train_index, val_index) in enumerate(skf.split(x_train,y_train)):
		x_tr = x_train[train_index]
		y_tr = y_train[train_index]
		x_val = x_train[val_index]
		clf.train(x_tr,y_tr)

		if isXgb:
			oof_train[val_index] = clf.predict(x_val)
			result = oof_train[val_index]
			result[result>0.5] = 1
			result[result<=0.5] = 0
			precisions += precision_score(y_train[val_index],result)
			recall += recall_score(y_train[val_index],result)
			oof_test_skf[i, :] = clf.predict(x_test)

		else:
			oof_train[val_index] = clf.predict(x_val)
			result = oof_train[val_index]
			result[result>0.5] = 1
			result[result<=0.5] = 0
			precisions += precision_score(y_train[val_index], result)
			recall += recall_score(y_train[val_index],result)
			oof_test_skf[i, :] = clf.predict(x_test)

		print(i,':',precision_score(y_train[val_index],result),'recall:',recall_score(y_train[val_index],result))
	oof_test[:] = oof_test_skf.mean(axis=0)

	oof_test[oof_test > 0.5] = 1
	oof_test[oof_test <= 0.5] = 0
	oof_train[oof_train > 0.5] = 1
	oof_train[oof_train <= 0.5] = 0

	precisions = precisions / NFOLDS
	recall = recall /NFOLDS
	print(precisions,"recall: ", recall)
	print()
	return oof_train.reshape(-1,1), oof_test.reshape(-1,1)


def metrics(y_test, predictions):
	print('log_loss: ', log_loss(y_test,predictions))
	predictions[predictions > 0.5] = 1
	predictions[predictions <=0.5] = 0
	print("precision_score: ", precision_score(y_test,predictions))
	print("recall: ",recall_score(y_test,predictions))


def my_confusion_matrix(y_true, y_pred):
	labels = list(set(y_true))
	conf_mat = confusion_matrix(y_true,y_pred,labels=labels)

	print("=== Confusion Matrix ===")
	print("%-6s" %'0',"%-6s" %'1',"  <----classeified as")
	print("%-6s"%conf_mat[0][0],"%-6s"%conf_mat[0][1]," | 0")
	print("%-6s"%conf_mat[1][0],"%-6s"%conf_mat[1][1]," | 1")

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


def trainModel():
	# -------- BUILD MODEL --------- #
	xg = XgbWrapper(seed = SEED, params = xgb_params)
	# et = SklearnHelper(clf=ExtraTreesClassifier, seed = SEED, params=et_params)
	rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
	lr = SklearnHelper(clf=LogisticRegression, seed=SEED, params = lr_params)
	# ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
	gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
	# svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

	# --------- CROSS VALIDATION ------ #
	print("Precision and Recall on cv")
	print("xgboost:"); 		xg_oof_train, xg_oof_test = get_oof(xg, x_train, y_train, x_test, isXgb = True)
	# print("extra tree:"); 	et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)
	print("LR:"); 			lr_oof_train, lr_oof_test = get_oof(lr, x_train, y_train, x_test) # LogisticRegression 
	print("RandomForest:"); rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test)  # Random Forest
	# print("AdaBoosting: "); ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
	print("GB:");			gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
	# print("SVC:");			svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

	# --------- PREDICTION ----------- #
	print("Precision and Recall on test data")
	target_names = ['class 0','class 1']

	print('xgboost precision: ', precision_score(y_test,xg_oof_test),'recall:',recall_score(y_test,xg_oof_test))
	my_confusion_matrix(y_test,xg_oof_test)
	print(classification_report(y_test,xg_oof_test,target_names = target_names));print()

	# print('extra tree: ', precision_score(y_test,et_oof_test), 'recall:', recall_score(y_test,et_oof_test))
	# my_confusion_matrix(y_test,et_oof_test)
	# print(classification_report(y_test,et_oof_test,target_names = target_names));print()

	print('LR: ', precision_score(y_test,lr_oof_test), 'recall:', recall_score(y_test,lr_oof_test))
	my_confusion_matrix(y_test,lr_oof_test)
	print(classification_report(y_test,lr_oof_test,target_names = target_names));print()

	print('Random Forest: ', precision_score(y_test,rf_oof_test), 'recall:', recall_score(y_test,rf_oof_test))
	my_confusion_matrix(y_test,rf_oof_test)
	print(classification_report(y_test,rf_oof_test,target_names = target_names));print()

	# print('AdaBoost: ', precision_score(y_test,ada_oof_test), 'recall:', recall_score(y_test,ada_oof_test))
	# my_confusion_matrix(y_test,ada_oof_test)
	# print(classification_report(y_test,ada_oof_test,target_names = target_names));print()

	print('Gradient Boost: ', precision_score(y_test,gb_oof_test), 'recall:', recall_score(y_test,gb_oof_test))
	my_confusion_matrix(y_test,gb_oof_test)
	print(classification_report(y_test,gb_oof_test,target_names = target_names));print()

	# print('SVC: ', precision_score(y_test,svc_oof_test), 'recall:', recall_score(y_test,svc_oof_test))

	print("Training is complete")

trainModel()
































