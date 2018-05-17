#coding:utf-8
from matplotlib import pyplot as plt
import json
import pickle
import numpy
from numpy import mean,median
from scipy.stats import mode
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def numFeature(arr):
	print ("forward weibo sum is",sum(arr))
	print ("user num is",len(arr))
	print ("mean is",mean(arr))
	print ("median is",median(arr))
	print ("mode is",mode(arr))
	print ("min is",min(arr))
	print ("max is",max(arr))
	print ("25% ",numpy.percentile(arr,25))
	print ("50% ",numpy.percentile(arr,50))
	print ("75% ",numpy.percentile(arr,75))
	print ("80% ",numpy.percentile(arr,80))
	print ("85% ",numpy.percentile(arr,85))
	print ("90% ",numpy.percentile(arr,90))
	print ("95% ",numpy.percentile(arr,95))
	print ("96% ",numpy.percentile(arr,96))
	print ("97% ",numpy.percentile(arr,97))
	print ("98% ",numpy.percentile(arr,98))
	print ("99% ",numpy.percentile(arr,99))
	print ("100% ",numpy.percentile(arr,100))


# 参数依次为list,title,X轴标签,Y轴标签,XY轴的范围
def draw_hist(myList,Title,Xlabel,Ylabel,Xmin,Xmax):
	xmajorLocator = MultipleLocator(10) #将x主刻度设置为10的倍数
	xmajorFormatter = FormatStrFormatter('%5.1f') #设置x轴标签文本的格式
	xminorLocator = MultipleLocator(2) #将x轴次刻度标签设置为2的倍数

	ax = plt.subplot(111)
	# 设置主刻度标签的位置,标签文本的格式
	ax.xaxis.set_major_locator(xmajorLocator)
	ax.xaxis.set_major_formatter(xmajorFormatter)

	# 设置次刻度标签的位置
	ax.xaxis.set_minor_locator(xminorLocator)

	ax.xaxis.grid(True, which = 'major') #x坐标轴的网格使用主刻度

	plt.hist(myList,1000)# 指定有多少个bin(箱子)
	plt.xlabel(Xlabel)
	plt.xlim(Xmin,Xmax)

	plt.ylabel(Ylabel)
	#plt.ylim(Ymin,Ymax)
	plt.title(Title)
	plt.show()

if __name__ == '__main__':
	with open('cnt.pkl', 'rb') as f:
		a = pickle.load(f)
	draw_hist(a,'prop in 1 hour','#props in 1 hour','num',0,100)   # 直方图展示

	#numFeature(a)
