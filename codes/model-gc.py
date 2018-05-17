import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import lightgbm as lgb
from xgboost.sklearn import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as pyplot
# % matplotlib inline

# Going to use these 5 base models for the stacking
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


import warnings
warnings.filterwarnings('ignore')


# Some useful parameters which will come in handy later on
data = pd.read_csv('../data/processed_features.csv')
#dat为数据集,含有feature和label.
data = shuffle(data)
train, test = train_test_split(data, test_size = 0.1)
ntrain = train.shape[0]
ntest = test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED)

# Class to extend the Sklearn classifier
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

    def feature_importances(self,x,y):
        return self.clf.fit(x,y).feature_importances_
    
# Class to extend XGboost classifer
class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))

    def feature_importances(self):
        return self.gbdt.feature_importances_

def get_oof(clf, x_train, y_train, x_test, isXgb = False):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    precisions = 0

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        clf.train(x_tr, y_tr)

        if isXgb:
	        oof_train[test_index] = clf.predict(x_te)
	        result = oof_train[test_index]
	        result[result>0.5] = 1
	        result[result<=0.5] = 0
	        precisions = precisions + precision_score(y_train[test_index],result)
	        oof_test_skf[i, :] = clf.predict(x_test)
        else:
        	oof_train[test_index] = clf.predict(x_te)
	        result = oof_train[test_index]
	        result[result>0.5] = 1
	        result[result<=0.5] = 0
	        precisions = precisions + precision_score(y_train[test_index],result)
	        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    oof_test[oof_test > 0.5] = 1
    oof_test[oof_test <= 0.5] = 0
    oof_train[oof_train > 0.5] = 1
    oof_train[oof_train <= 0.5] = 0
    precisions = precisions / NFOLDS
    print(precisions)
    return oof_train.reshape(-1,1), oof_test.reshape(-1,1)

def drawHeatmap(data, kwd):
	#绘制热力图
	corrmat = data.astype(float).corr(method = kwd)
	pyplot.subplots(figsize=(10, 8))
	sns.heatmap(corrmat, vmax=.8, annot = True, annot_kws={'size': 5}, square=True)
	pyplot.show()
	return data


def metrics(y_test, predictions):
	print('log_loss:',log_loss(y_test,predictions))
	predictions[predictions > 0.5] = 1
	predictions[predictions <=0.5] = 0
	# y_sub_1 = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
	print("precision_score:",precision_score(y_test,predictions))
	print("recall_score:",recall_score(y_test,predictions))



# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 300,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 300,
    'max_features': 0.3,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 300,
    'learning_rate' : 0.1
}

# LogisticRegression parameters
lr_params = {
	
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025,
    'probability': True
    }

#
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
    # 'eval_metric': 'error',
    # 'nrounds': 350

    'learning_rate' : 0.1,
 	'n_estimators': 200,
 	'max_depth' : 7,
 	'min_child_weight' : 2,
 	#gamma=1,
 	'gamma' : 0.7,                        
 	'subsample' : 0.8,
 	'colsample_bytree' : 0.5,
 	'objective': 'binary:logistic',
 	'nthread' : -1,
 	'scale_pos_weight':1,
 	'silent' : 1
}

# Create 5 objects that represent our 4 models
xg = XgbWrapper(seed = SEED, params = xgb_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed = SEED, params=et_params)
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
lr = SklearnHelper(clf=LogisticRegression, seed=SEED, params = lr_params)
# ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)

# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
print(train.dtypes)
y_train = train['label'].ravel()
train = train.drop(['label'], axis=1)
x_train = train.values # Creates an array of the train data
print(test['label'].value_counts())
y_test = test['label'].ravel()
test = test.drop(['label'], axis=1) 
x_test = test.values # Creats an array of the test data
print("precision on cv")
# Create our OOF train and test predictions. These base results will be used as new features
xg_oof_train, xg_oof_test = get_oof(xg, x_train, y_train, x_test, isXgb = True)
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)
lr_oof_train, lr_oof_test = get_oof(lr, x_train, y_train, x_test) # LogisticRegression
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
# ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

print("precision on test data")
print(precision_score(y_test,xg_oof_test))
print(precision_score(y_test,et_oof_test))
print(precision_score(y_test,lr_oof_test))
print(precision_score(y_test,rf_oof_test))
print(precision_score(y_test,gb_oof_test))
print(precision_score(y_test,svc_oof_test))
print("Training is complete")

# xg_features = xg.feature_importances()
rf_features = rf.feature_importances(x_train,y_train)
# lr_feature = lr.feature_importances(x_train, y_train)
et_features = et.feature_importances(x_train, y_train)
gb_features = gb.feature_importances(x_train,y_train)

cols = train.columns.values
# Create a dataframe with features
feature_dataframe = pd.DataFrame( {'features': cols,
	# 'Xgboost feature importances': xg_features,
     'Random Forest feature importances': rf_features,
      'ExtraTrees feature importances': et_features,
    'Gradient Boost feature importances': gb_features
    })
feature_dataframe['mean'] = feature_dataframe.mean(axis = 1)
feature_dataframe.to_csv('./feature_importances.csv')

base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train.ravel(),
	'ExtraTrees':et_oof_train.ravel(),
	'Xgboost': xg_oof_train.ravel(),
     'LogisticRegression': lr_oof_train.ravel(),
     # 'AdaBoost': ada_oof_train.ravel(),
      'GradientBoost': gb_oof_train.ravel(),
      'SVM': svc_oof_train.ravel(),
      'label': y_train.ravel()
    })
base_predictions_test = pd.DataFrame( {'RandomForest': rf_oof_test.ravel(),
	'ExtraTrees' :et_oof_test.ravel(),
	'Xgboost': xg_oof_test.ravel(),
     'LogisticRegression': lr_oof_test.ravel(),
     # 'AdaBoost': ada_oof_test.ravel(),
      'GradientBoost': gb_oof_test.ravel(),
      'SVM': svc_oof_test.ravel(),
      'label' : y_test.ravel()
      # 'label': y_test
    })
print(base_predictions_train.head(10))
print(base_predictions_test.head(10))
# print(ba)
# base_predictions_train.pipe(drawHeatmap, 'pearson') 

# print(et_oof_test.shape, lr_oof_test.shape)
x_train = np.concatenate((xg_oof_train, et_oof_train, lr_oof_train, rf_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test = np.concatenate((xg_oof_test,et_oof_test, lr_oof_test, rf_oof_test, gb_oof_test, svc_oof_test), axis=1)
# print(x_train.shape)
# print(x_test.head(3))
# print(list(y_test).value_counts())
'''
print('xgboost')


xgb = xgb.XGBClassifier(
 learning_rate = 0.1,
 n_estimators= 200,
 max_depth= 5,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train, y_train)
xgb_predictions = xgb.predict(x_test)
'''

print('lightgbm')

# create dataset for lightgbm
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc', 'binary_logloss'},
    'num_leaves': 64,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

print('Start training...')
# train

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

print('Start predicting...')
# predict
gbm_predictions = gbm.predict(x_test, num_iteration=gbm.best_iteration)
gbm.save_model('./gbm.txt', num_iteration = gbm.best_iteration)

# feature names
print('Feature names:', gbm.feature_name())

# feature importances
print('Feature importances:', list(gbm.feature_importance()))
lr.train(x_train, y_train)
lr_predictions = lr.predict_proba(x_test)[:,1]
joblib.dump(lr,'lr.pkl')
# clf=joblib.load('filename.pkl')
vote_predictions = x_test.mean(axis = 1)


# print('xgboost metrics:')
# metrics(y_test, xgb_predictions)
print('lightgbm metrics:')
metrics(y_test, gbm_predictions)
print('vote metrics:')
metrics(y_test, vote_predictions)
print('LR metric:')
metrics(y_test, lr_predictions)
# print(gbm_predictions.shape,lr_predictions.shape,vote_predictions.shape)
# final_data= pd.DataFrame({"vote" : vote_predictions ,"lr":lr_predictions, "gbm":gbm_predictions})
# # print(final_data.shape)
# final_predictions = final_data.mean(axis = 1)
# print('final_predictions:')
# metrics(y_test, final_predictions)