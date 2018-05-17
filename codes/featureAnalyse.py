import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import numpy as np
warnings.filterwarnings('ignore')

x = pd.read_csv('../data/processed_features.csv', header = None, sep = ',', names = ["contain_pic", "n_forwards", "userCertify", "gender", "n_follower",
	"n_fan", "n_weibo", "contain_faces", "contain_location", "contain_Music", "contain_Video",
	"contain_Url", "uCertify_avg", "forwards_avg", "praise_avg", "gap_avg"])

def draw(filed):
	plt.subplot(221)
	sns.distplot(x[filed])
	plt.title(filed + " dist")

	plt.subplot(222)
	sns.distplot(np.log1p(x[filed]))
	plt.title(filed + " log dist")

	plt.subplot(223)
	sns.distplot(StandardScaler().fit_transform(x['n_fan'][:, np.newaxis]))
	plt.ylabel('StandardScaler')

	plt.subplot(224)
	#sns.distplot((np.log1p(StandardScaler().fit_transform(x['n_fan'][:, np.newaxis]))))
	sns.distplot(preprocessing.scale(x['n_fan']))
	plt.ylabel('scale')

	plt.show()

draw('uCertify_avg')