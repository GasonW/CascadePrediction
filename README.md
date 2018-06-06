# CascadePrediction
A cascade prediction model

&emsp; The model is used to predict weibo cascade. 

## Dataset Description

&emsp; In this project, I used WeiboForwards dataset, which contains 15807 weibo and its comments and forwards. An example of the data is like that:

{
	"originalMid" : "4099456032803526",
	"type" : "forward",
	"originalWeibo" : {
		"mid" : "4099456032803526",
		"url" : "http://weibo.com/1953891524/EFJiXbLka",
		"userurl" : "http://weibo.com/311619000?refer_flag=1001030103_",
		"name" : "南宁热搜榜",
		"content" : "\n\t\t#南宁身边事# 【辟谣：对于网传的房屋倒塌砸车事故。】南宁交警反馈：从今天4月22日早上到现在，没有接到大沙田片区有建筑物倾倒砸中公交车和行人的报警。六大队领导又询问了银海建设路口附近岗亭的民警，同时与施工方进行了核实，楼房拆除过程中，没有发生建筑物及砖块砸中公交车和行人的情况。视频因 ​\n\t\t\t...展开全文c\n\t\t",
		"time" : "1492869945000",
		"forward" : "32",
		"comment" : "74",
		"hotrate" : 0,
		"isOrigin" : true,
		"piclist" : [
			"http://wx3.sinaimg.cn/large/747604c4gy1fevt4gboedj208w06owey.jpg"
		],
		"userCertify" : 1,
		"sourcePlatform" : "秒拍网页版",
		"praise" : "131",
		"userId" : "1953891524",
		"userGender" : "female",
		"userFollowCount" : "1113",
		"userFanCount" : "230000",
		"userWeiboCount" : "6253",
		"userLocation" : "广西",
		"userDescription" : "微博本地资讯博主（南宁）http://t.cn/RUmMhZ7",
		"faces" : null,
		"weiboLocation" : null,
		"containsMusic" : false,
		"containsVideo" : false,
		"containsUrl" : false
	},
	"props" : [
		{
			"mid" : "4100007974570547",
			"url" : "http://weibo.com/5665979346/F00cRjb0v",
			"userurl" : "http://weibo.com/5665979346",
			"username" : "敬軒的小迷妹",
			"content" : "//@yellowUpBDown：拆房子这么拆？没有隔离设备？那么大的力量石头蹦出来多危险？还有脸说？",
			"time" : "1493001480000",
			"forward" : "0",
			"isComment" : false,
			"isForward" : true,
			"originalWid" : "4099456032803526",
			"faces" : null,
			"userCertify" : 0,
			"praise" : "0",
			"userId" : "5665979346",
			"userGender" : null,
			"userFollowCount" : null,
			"userLocation" : null,
			"userFanCount" : null,
			"userWeiboCount" : null,
			"userDescription" : null
		},
		{
			"mid" : "4099985140744048",
			"url" : "http://weibo.com/5175660940/EFX4m37yM",
			"userurl" : "http://weibo.com/5175660940",
			"username" : "小蚂蚁的山河",
			"content" : "转发微博",
			"time" : "1492996080000",
			"forward" : "0",
			"isComment" : false,
			"isForward" : true,
			"originalWid" : "4099456032803526",
			"faces" : null,
			"userCertify" : 0,
			"praise" : "0",
			"userId" : "5175660940",
			"userGender" : null,
			"userFollowCount" : null,
			"userLocation" : null,
			"userFanCount" : null,
			"userWeiboCount" : null,
			"userDescription" : null
		}
		]
}

## Preprocess flow
1. Use `uniq.py` to delete the repeat weibos, 
2. Use `mktrain.py` to produce train data
3. Use `mkFeatures.py` to extract features from it.
4. Use `CascadePrediction.py` to predict
5. Use `dump.js` to export data from mongodb, the command is "mongo database dump.js>yourfile.json"