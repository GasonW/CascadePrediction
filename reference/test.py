import pandas as pd
import matplotlib.pyplot as pyplot
import seaborn as sns
import numpy as np
# from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
# % matplotlxib inline
# bring in the six packs
# df_train = pd.read_csv('./input/train.csv')
# check the decoration
# total = df_train.isnull().sum().sort_values(ascending=False)
# print(total)
# 读取训练数据集和测试数据集
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")


# 合并两个训练集
# test['SalePrice'] = None
# all = pd.merge(train, test)
# set_option('display.line_width', 120)
# print(train.corr(method='pearson'))
# train.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
# train.plot(kind='box', subplots=True, layout=(6, 7),
#            sharex=False, sharey=False, fontsize=8)
# pd.scatter_matrix(train)
# pyplot.show()

combined = pd.concat([train, test], axis=0, ignore_index=True)
ntrain = train.shape[0]
# print(ntrain)
Y_train = train["SalePrice"]
X_train = train.drop(["Id", "SalePrice"], axis=1)
print("train data shape:\t", train.shape)
print("test data shape:\t", test.shape)
print("combined data shape:\t", combined.shape)
'''
# 特征分析
print(Y_train.skew())
# Y_train.plot.kde()
# 偏度分析
np.abs(combined[:ntrain].skew()).sort_values(ascending=False).head(20)
# pyplot.show()
# 缺失值分析
cols_missing_value = combined.isnull().sum() / combined.shape[0]
# print(cols_missing_value)
cols_missing_value = cols_missing_value[cols_missing_value > 0]
# print(cols_missing_value)
cols_missing_value.sort_values(ascending=False).head(10).plot.barh()
pyplot.show()
'''

print(Y_train.describe())
# print(train.corr(method='pearson'))
# sns.distplot(Y_train)
# pyplot.show()
print("Skewness: %f" % train['SalePrice'].skew())
print("Kurtosis: %f" % train['SalePrice'].kurt())
'''
var = 'TotalBsmtSF'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
'''
# pyplot.show()

'''
var = 'YearBuilt'
data = pd.concat([train['SalePrice'], train[var]], axis=1)
f, ax = pyplot.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
pyplot.xticks(rotation=90)
pyplot.show()
'''
'''
corrmat = train.corr(method='pearson')
pyplot.subplots(figsize=(10, 8))
sns.heatmap(corrmat, vmax=.8, square=True)
pyplot.show()
'''
'''
corrmat = train.corr(method='pearson')
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={
                 'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
pyplot.show()
'''
'''
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea',
        'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size=1)
pyplot.show()
'''
'''
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum() / train.isnull().count()
           ).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data[missing_data['Total'] > 0])
train = train.drop((missing_data[missing_data['Total'] > 1]).index, 1)
train = train.drop(train.loc[train['Electrical'].isnull()].index)
# print(train.isnull().sum().max())
saleprice_scaled = StandardScaler().fit_transform(
    train['SalePrice'][:, np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][0:10]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:-1]
print('outer rage (low)of the distribution:\n', low_range)
print('\nouter range(high) of the distribution:\n', high_range)
'''
'''
var = 'GrLivArea'
data = train[['SalePrice', var]]
# data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
# pyplot.show()
'''
'''
train.sort_values(by='GrLivArea', ascending=False)[:2]
# train = train.drop(train[train['Id'] == 1299].index)
# train = train.drop(train[train['Id'] == 524].index)

var = 'TotalBsmtSF'
data = train[['SalePrice', var]]
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
pyplot.show()
'''
train = train.drop(train[train['Id'] == 1299].index)
train = train.drop(train[train['Id'] == 524].index)
'''
sns.distplot(train['SalePrice'], fit=stats.norm)
fig = pyplot.figure()
res = stats.probplot(train['SalePrice'], dist="norm", plot=pyplot)
pyplot.show()
'''
train['SalePrice'] = np.log(train['SalePrice'])
'''
sns.distplot(train['SalePrice'], fit=stats.norm)
fig = pyplot.figure()
res = stats.probplot(train['SalePrice'], plot=pyplot)
pyplot.show()
'''
train['GrLivArea'] = np.log(train['GrLivArea'])
'''
sns.distplot(train['GrLivArea'], fit=stats.norm)
fig = pyplot.figure()
res = stats.probplot(train['GrLivArea'], plot=pyplot)
pyplot.show()
'''
train['HasBsmt'] = pd.Series(len(train['TotalBsmtSF']), index=train.index)
train['HasBsmt'] = 0
train.loc[train['TotalBsmtSF'] > 0, 'HasBsmt'] = 1

train.loc[train['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log1p(
    train['TotalBsmtSF'])

sns.distplot(train[train['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit=stats.norm)
fig = pyplot.figure()
res = stats.probplot(train[train['TotalBsmtSF'] > 0]
                     ['TotalBsmtSF'], plot=pyplot)
# pyplot.show()
# print("Skewness: %f" % train[train['TotalBsmtSF'] > 0]['TotalBsmtSF'].skew())
train = pd.get_dummies(train)
print(train.describe())
