package com.ict.mcg.veryfication.feature;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import org.ansj.domain.Term;






//import com.google.gson.Gson;
import com.ict.mcg.gather.entity.WeiboEntity;
import com.ict.mcg.model.SingleRFClassifer;
import com.ict.mcg.processs.ICTAnalyzer;
import com.ict.mcg.processs.NamedEntity;
import com.ict.mcg.processs.Partition;
import com.ict.mcg.processs.SentimentAnalysis;
import com.ict.mcg.processs.WordNode;
import com.ict.mcg.util.FileIO;


public class SingleWeiboFeatureExtractor {
	
	private String content = "";
	private ArrayList<WordNode> originTerms = new ArrayList<WordNode>();  //词性与字数、词数分析
	private ArrayList<WordNode> filteredTerms = new ArrayList<WordNode>();  //词数与情感分析
	private Map<String,Integer> emotionInfo =  new HashMap<String,Integer>();
	public static Partition p = new Partition();
	private String emotionFile = FileIO.getFilePath()+"emotionWords.txt";
	private String emotionFile_resource = FileIO.getResourcePath()+"emotionWords.txt";
	private static SingleWeiboFeatureExtractor extractor = new SingleWeiboFeatureExtractor();
	
	public static SingleWeiboFeatureExtractor getInstance(){
		return extractor;
	}
	
	private SingleWeiboFeatureExtractor(){
		loadFile();
	}
	
	public void loadFile(){
		try {
			InputStream is = this.getClass().getResourceAsStream(emotionFile_resource);
			if (is == null) {
				is = new FileInputStream(emotionFile);
			}
			BufferedReader emotionwordsreader = new BufferedReader(new InputStreamReader(is, "utf-8"));
			String str = "";
			while((str=emotionwordsreader.readLine())!=null){
				String []split = str.split("\t");
//				System.out.println(split[0]);
				emotionInfo.put(split[1], integerConvert(split[0]));
			}
			emotionwordsreader.close();
		} catch (NumberFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public double[] extractFeature(WeiboEntity weibo){
		double [] features= new double[30]; 
		content = weibo.getContent();
		this.segWord();
		int []wordCnt = getWordCnt(); //字数、词数
		ArrayList<double []> emoInfo = emotionCompute(); 
		double[] basicEmoInf = emoInfo.get(0);
		double[] emoTypeVal = emoInfo.get(1);
		int [] peCnt = getPeCnt();
		int [] nameEntityCnt = getNameEntityCnt();
		int [] maskCnt = getMaskCnt();
		int [] emoTypeCnt = getEmotionTypeCnt(weibo.getFaces());
		ArrayList<String> piclist = weibo.getPiclist();
		
		features[0] = wordCnt[0];
		features[1] = wordCnt[1];
		features[2] = basicEmoInf[1];
		features[3] = basicEmoInf[2];
		features[4] = peCnt[0];
		features[5] = peCnt[2];
		features[6] = nameEntityCnt[0];
		features[7] = nameEntityCnt[1];
		features[8] = nameEntityCnt[2];
		
		if(piclist!=null&&piclist.size()>0){
			features[9] = piclist.size();
			features[10] = 1.0;
			if(piclist.size()>1){
				features[11] = 1.0;
			}else{
				features[11] = 0.0;
			}
		}else{
			features[9] = 0;
			features[10] = 0;
			features[11] = 0;
		}
		
		features[12] = integerConvert(weibo.getUserFanCount());
		features[13] = integerConvert(weibo.getUserFollowCount());
		features[14] = integerConvert(weibo.getUserWeiboCount());
		features[15] = weibo.getUserCertify();
		features[16] = integerConvert(weibo.getForword());
		features[17] = integerConvert(weibo.getComment());
		features[18] = maskCnt[0];
		features[19] = maskCnt[1];
		//hasloc features[20] = //
		if(weibo.getWeiboLocation()!=null&&weibo.getWeiboLocation().length()>0){
			features[20] = 1.0; 
		}else{
			features[20] = 0.0;
		}
		
		if(weibo.getUserDescription()!=null&&weibo.getUserDescription().length()>0){
			features[21] = 1.0; 
		}else{
			features[21] = 0.0;
		}
		
		if(weibo.getUserGender()!=null&&(weibo.getUserGender().equals("male")||weibo.getUserGender().equals("男"))){
			features[22] = 1.0;
		}else{
			features[22] = 0.0;
		}
		
		if(weibo.getFaces()!=null&&weibo.getFaces().size()>0){
			features[23] = 1;
			if(weibo.getFaces().size()>1){
				features[24] = 1;
			}else{
				features[24] = 0.0;
			}
			features[25] = weibo.getFaces().size();
			features[26] = emoTypeCnt[0];
		}else{
			features[23] = 0;
			features[24] = 0;
			features[25] = 0;
			features[26] = 0;
		}
		features[27] = emoTypeVal[5];
		features[28] = emoTypeVal[6];
		features[29] = emoTypeVal[7];
		
		return features;
	}
	
	private void segWord() {
//		String file = "./file/dic";
//		Partition p = new Partition(file);
		// 分词
//		ArrayList<WordNode> al = p.participleAndMerge(content);
		//过滤单字
		ArrayList<WordNode> alOringin = p.participleWithFilter(content, 0);
		ArrayList<WordNode> alFiltered = p.participleWithFilter(content, 1);
		NamedEntity ne = new NamedEntity();
		ne.setProps(alOringin);
		ne.setProps(alFiltered);
		originTerms = alOringin;
		filteredTerms = alFiltered;
	}
	
	//统计微博的字数|词数  标点符号不算进去，英文单词算作一个字和一个词
	private int[] getWordCnt(){
		int wcnt = originTerms.size();
		int count = 0;
		for (WordNode w : originTerms) {
			if (w.getPos().charAt(0) == 'x'||w.getPos().equals("en")) {
				count++;
			} else {
				count += w.getWord().length();
			}
		}
		int[] r = {count,wcnt };
		return r;
	}
	
	
	//统计正负情感词数，情感值
	//统计各类情感值1:happy 2:fine 3:angry 4:sad 5:scared 6:evil 7:shock
	
	private ArrayList<double[]> emotionCompute(){
		ArrayList<String> words = new ArrayList<String>();
		for(WordNode node:filteredTerms){
			words.add(node.getWord());
		}
		
		SentimentAnalysis sa = new SentimentAnalysis();
		double emoVal = sa.getEmotionFromSentence(words);
		int negCnt = sa.getNegWordcloud().size();
		int posCnt = sa.getPosWordcloud().size();
		
		double []result1 = new double[]{posCnt*1.0,negCnt*1.0,emoVal};
		
		double []result2 = sa.getEmotionEachTypeFromSentence(words);
		
		ArrayList<double[]> result = new ArrayList<double[]>();
		result.add(result1);
		result.add(result2);
		return result;
	}
	
	//统计第一二三人称
	private int[] getPeCnt(){
		int []count = new int[3];
		for(WordNode node:originTerms){
			String s = node.getWord().substring(0,1);
			if(s.equals("我")){
//				System.out.println(term);
				count[0]++;
			}else if(s.equals("你")){
//				System.out.println(term);
				count[1]++;
			}else if(s.equals("他")||s.equals("她")||s.equals("它")){
//				System.out.println(term);
				count[2]++;
			}
		}
		return count;
	}
	//统计人名、地名、机构名
	private int[] getNameEntityCnt(){
		int []count = new int[3];
		for (WordNode w : originTerms) {
			if (w.getProps() == NamedEntity.PERSON) {
				count[0]++;
			} else if (w.getProps() == NamedEntity.REGION) {
				count[1]++;
			}else if (w.getProps() == NamedEntity.ORGANIZATION) {
				count[2]++;
			} 
		}
		return count;
	}
	
	//统计？ ！个数
	private int[] getMaskCnt(){
		int [] count = new int[2];
		//去除????
		content = content.replaceAll("\\?\\?\\?\\?", "");
		
		for(int i=0;i<content.length();i++){
			if(content.charAt(i)=='?'||content.charAt(i)=='？'){
				count[0]++;
			}else if(content.charAt(i)=='!'||content.charAt(i)=='！'){
				count[1]++;
			}
		}
		return count;
	}
	
	//获取发布平台类型
	public int getPlatformType(String platform) {
		int platformType = 0;  //无
		String[] mobilePhrase = { "iPad", "iPhone","手机", "平板","HTC",  "Android", "vivo", "联想", "华为", "小米", "酷派", "荣耀","HUAWEI","乐1s","安卓","魅蓝","中兴", "OPPO", "金立", "三星", "索尼", "TCL", "红米", "魅族", "一加", "锤子", "诺基亚","微博","客户端"};
		if (platform!=null&&platform.length() > 0) {
			int i = 0;
			for(i = 0;i<mobilePhrase.length;i++){
				if(platform.contains(mobilePhrase[i])){
					break;
				}
			}
			if(i<2)
				platformType = 1;  //ios
			else if(i<mobilePhrase.length-2)
				platformType = 2;  //Andriod
			else if(i==mobilePhrase.length-2)
				platformType = 3;  //weibo.com
			else if(i==mobilePhrase.length-1)
				platformType = 4;  //第三
			else
				platformType = 4;
		}
//			if(platformType==3)System.out.println(platform);
		return platformType;
	}
	
	//统计各类表情个数 0happy,1angry,2sad
	public int[] getEmotionTypeCnt(Map<String,Double> faces){
		int [] emotionTypeCnt = new int[3];// 0 happy //1 angry //2 sad
		if(faces==null||faces.size()==0) return emotionTypeCnt;
		for(Entry<String, Double> entry:faces.entrySet()){
			String face = String.valueOf(entry.getKey());
			if(emotionInfo.containsKey(face)){
				int type = emotionInfo.get(face);
				emotionTypeCnt[type]+=1;
			}// if
		}// for
		return emotionTypeCnt;
	}
	
	private int integerConvert(String inte){
		if(inte==null||inte==""||inte.trim().length()==0){
			return 0;
		}
		return Integer.valueOf(inte);
	}
	
	/*public static void main(String []args){
//		String weibostr = "{\"Mid\":\"4008619718633990\",\"url\":\"http://weibo.com/5122654941/E3CeLAe5U?refer_flag\u003d1001030103_\",\"userurl\":\"http://weibo.com/elinyi?refer_flag\u003d1001030103_\",\"name\":\"临沂大小事\",\"content\":\"\n\t\t#王宝强离婚#马蓉回应王宝强，王宝强亲子鉴定 被网友辟谣；王宝强以前和妈妈的一段直播视频被翻出来，任何人的成功背后都有泪和汗水，看到了满满的质朴和不忘初心，本来以为苦尽甘来，有了幸福的婚姻，可爱的儿女，却终成伤 秒拍视频\n\t\t\",\"time\":\"1471212880000\",\"forword\":\"2104\",\"comment\":\"694\",\"hotrate\":0.0,\"isOrigin\":true,\"piclist\":[\"http://ww4.sinaimg.cn/orj480/736f0c7ejw1f6sp099rv9j20ho0a0dh9.jpg\"],\"faces\":{\"泪1\":0.9464285714285714,\"泪\":0.9285714285714286},\"userCertify\":0,\"classtitle\":\"\",\"sourcePlatform\":\" 元实互动 iPhone 6s\",\"praise\":\"12203\",\"userId\":\"5122654941\",\"userGender\":\"male\",\"userFollowCount\":\"198\",\"userFanCount\":\"320000\",\"userWeiboCount\":\"35562\",\"userLocation\":\"山东临沂\",\"userDescription\":\"知名本地博主 微博区域新媒体大使（山东） 资讯视频博主 微电商达人\"}";
//		String weibostr = "{\"Mid\":\"4008651368799362\",\"url\":\"http://weibo.com/2035397314/E3D3OAV7c?refer_flag\u003d1001030103_\",\"userurl\":\"http://weibo.com/333213387?refer_flag\u003d1001030103_\",\"name\":\"杜佛爷\",\"content\":\"\n\t\t#王宝强离婚# 宝哥啊中国的女人多的使既然马蓉不要脸，经纪人给你带女帽子，中国女人大把的不差马蓉一个，你的财产和孩子她也不想要得到，也不给她抚养权让这个潘金莲彻底死去，我们粉丝团帮你找个\n\t\t\",\"time\":\"1471220426000\",\"forword\":\"0\",\"comment\":\"0\",\"hotrate\":0.0,\"isOrigin\":true,\"piclist\":[],\"userCertify\":0,\"classtitle\":\"\",\"sourcePlatform\":\"微话题\",\"praise\":\"0\",\"userId\":\"2035397314\",\"userGender\":\"male\",\"userFollowCount\":\"178\",\"userFanCount\":\"702\",\"userWeiboCount\":\"2498\",\"userLocation\":\"广东\",\"userDescription\":\"广州市花都区凤凰广场志愿驿站 站长\"}";
//		String weibostr = "{\"Mid\":\"4001126367566240\",\"url\":\"http://weibo.com/2768696947/E0tiIvKk8?refer_flag\u003d1001030103_\",\"userurl\":\"http://weibo.com/u/2768696947?refer_flag\u003d1001030103_\",\"name\":\"初吻给了韩氏娜多\",\"content\":\"\n\t\t天门灾区人民强烈要求政府赔偿。不是补偿。是你们把水放进来地，如果是自然溃口，我们无怨无悔，我们认了。你们抽闸泄洪，扒堤放水。淹我良田，毁我家园。如今雨过天晴，你们上游开闸泄洪，下游堵闸限流。让本不会淹的地方洪水滔滔。不是天灾是人祸啊。保武汉保天门城区保京山地区，凭什么要牺牲我们，\n\t\t\t...展开全文c\n\t\t\",\"time\":\"1469426326000\",\"forword\":\"1\",\"comment\":\"1\",\"hotrate\":0.0,\"isOrigin\":true,\"piclist\":[],\"userCertify\":0,\"classtitle\":\"\",\"sourcePlatform\":\"iPhone 6 Plus\",\"praise\":\"1\",\"userId\":\"2768696947\",\"userGender\":\"female\",\"userFollowCount\":\"133\",\"userFanCount\":\"27\",\"userWeiboCount\":\"193\",\"userLocation\":\"广东广州\"}";
//		String weibostr = "{\"Mid\":\"4001184921063175\",\"url\":\"http://weibo.com/1919836305/E0uPa4sAW?refer_flag\u003d1001030103_\",\"userurl\":\"http://weibo.com/400868709?refer_flag\u003d1001030103_\",\"name\":\"JennieyyyD\",\"content\":\"\n\t\t又到武汉啦 从洪水模式转换成蒸笼模式了\n\t\t\",\"time\":\"1469440286000\",\"forword\":\"0\",\"comment\":\"0\",\"hotrate\":0.0,\"isOrigin\":true,\"piclist\":[],\"faces\":{\"奥特曼\":0.9},\"userCertify\":0,\"classtitle\":\"\",\"sourcePlatform\":\"iPhone客户端\",\"praise\":\"0\",\"userId\":\"1919836305\",\"userGender\":\"male\",\"userFollowCount\":\"185\",\"userFanCount\":\"1\",\"userWeiboCount\":\"3\",\"userLocation\":\"青海海北\"}";
//		String weibostr = "{\"Mid\":\"4001271721263351\",\"url\":\"http://weibo.com/2384152914/E0x5a5iED?refer_flag\u003d1001030103_\",\"userurl\":\"http://weibo.com/u/2384152914?refer_flag\u003d1001030103_\",\"name\":\"湖北天气\",\"content\":\"\n\t\t【气象预警】武汉中心气象台7月25日23时28分发布暴雨黄色预警信号:预计未来6小时，房县局部有50毫米以上降水，阵风6-8级，伴有雷电，请注意防范山区山洪、地质灾害、中小河流洪水。\n\t\t\",\"time\":\"1469460980000\",\"forword\":\"1\",\"comment\":\"0\",\"hotrate\":0.0,\"isOrigin\":true,\"piclist\":[\"http://ww4.sinaimg.cn/large/8e1b4952gw1f66ktispgxg203i030jr9.gif\"],\"userCertify\":0,\"classtitle\":\"\",\"sourcePlatform\":\"微博 weibo.com\",\"praise\":\"2\",\"userId\":\"2384152914\",\"userGender\":\"female\",\"userFollowCount\":\"207\",\"userFanCount\":\"200000\",\"userWeiboCount\":\"12750\",\"userLocation\":\"湖北武汉\",\"userDescription\":\"关注湖北天气微博，关注湖北气象微信\"}";
//		String weibostr = "{\"Mid\":\"3996731006358831\",\"url\":\"http://weibo.com/1895394815/DEApKqGdN?refer_flag\u003d1001030103_\",\"userurl\":\"http://weibo.com/xianchanghb?refer_flag\u003d1001030103_\",\"name\":\"荆楚网\",\"content\":\"\n\t\t#湖北防汛抗旱#【气象预警】武汉中心气象台7月13日10时40分发布暴雨红色预警信号：预计未来3小时，潜江、荆州、洪湖局部降水将达100毫米以上，伴有雷电，阵风7-9级，中小河流洪水气象风险等级高，请注意防范。\n\t\t\",\"time\":\"1468378390000\",\"forword\":\"0\",\"comment\":\"0\",\"hotrate\":0.0,\"isOrigin\":true,\"piclist\":[\"http://ww2.sinaimg.cn/large/70f96dffgw1f5s3bh4r4jj204t04mdfx.jpg\"],\"userCertify\":0,\"classtitle\":\"\",\"sourcePlatform\":\"微博 weibo.com\",\"praise\":\"1\",\"userId\":\"1895394815\",\"userGender\":\"female\",\"userFollowCount\":\"735\",\"userFanCount\":\"1980000\",\"userWeiboCount\":\"52734\",\"userLocation\":\"湖北武汉\",\"userDescription\":\"荆楚网新浪官方微博\"}";
//		String weibostr = "{\"Mid\":\"3955445921669319\",\"url\":\"http://weibo.com/1655890975/DngoU70gv\",\"userurl\":\"http://weibo.com/panduolala?refer_flag=1001030103_\",\"name\":\"Pandora占星小巫\",\"content\":\"\n\t\t【小巫本周塔罗运势】金牛座牌面：正位星辰、逆位太阳、逆位宝剑侍从。事业：之前为自己定下的目标，也随着自己的努力，初见成效。学业：不要被无形的压力打垮，试着放松你的身心，用实际行动打破所有谣言和不公。爱情：不要去幻想你的爱情，爱情需要你主动去沟通经营。幸运色：黑色；开运物：小说。\n\t\t\",\"time\":\"1458535260000\",\"forword\":\"124\",\"comment\":\"87\",\"hotrate\":0,\"isOrigin\":true,\"piclist\":[],\"userCertify\":0,\"sourcePlatform\":\"iPhone 6s\",\"praise\":\"148\",\"userId\":\"1655890975\",\"userGender\":\"female\",\"userFollowCount\":\"34\",\"userFanCount\":\"2930000\",\"userWeiboCount\":\"17907\",\"userLocation\":\"海外美国\",\"userDescription\":\"关注小巫就是关注真实的你。（微博占星的唯一领地）合作及原创约稿QQ：614543187。\"}";
//		String weibostr = "{\"Mid\":\"4055260894959958\",\"url\":\"http://weibo.com/1974576991/EnbAtkOjk?refer_flag\u003d1001030103_\",\"userurl\":\"http://weibo.com/huanqiushibaoguanwei?refer_flag\u003d1001030103_\",\"name\":\"环球时报\",\"content\":\"\n\t\t【狠心母亲卖掉两个亲生女儿 女儿还帮她数钱】大人在数钱，小女孩在旁玩耍，还蹲下帮着一起数钱，而这段视频背后的真相让人不寒而栗！数钱的女子是小女孩的妈妈，手中是刚刚卖掉小女孩得到的1万3千元。而小女孩，已经不是这位妈妈卖掉的第一个女儿！亲生母亲何以狠心至此？戳↓（央视新闻）\n\t\t\t...展开全文c\n\t\t\",\"time\":\"1482333003000\",\"forword\":\"479\",\"comment\":\"767\",\"hotrate\":0.0,\"isOrigin\":true,\"piclist\":[\"http://ww4.sinaimg.cn/large/60718250jw1ev2ip5yg4bj20ox0ian0f.jpg\"],\"faces\":{\"怒\":0.13793103448275862},\"userCertify\":2,\"classtitle\":\"\",\"sourcePlatform\":\"微博 weibo.com\",\"praise\":\"442\",\"userId\":\"1974576991\",\"userGender\":\"male\",\"userFollowCount\":\"636\",\"userFanCount\":\"5900000\",\"userWeiboCount\":\"111795\",\"userLocation\":\"北京\",\"userDescription\":\"《环球时报》微博\"}";
//		String weibostr = "{\"Mid\":\"4054553202528385\",\"url\":\"http://weibo.com/5812757187/EmTb2aBKp?refer_flag\u003d1001030103_\",\"userurl\":\"http://weibo.com/u/5812757187?refer_flag\u003d1001030103_\",\"name\":\"Circlesun6002\",\"content\":\"\n\t\t#每天和朴灿烈说晚安##朴灿烈#最近微微有些不顺，呵呵！听了一晚上的#staywithme#,做了一傍晚上的翻译心情不言而喻（毕竟我爱学习），唱了会儿歌好了一会儿，然后一直纠结导入要不要换成罗一笑，睡醒后继续开始背背背……晚安，好梦！\n\t\t\",\"time\":\"1482164276000\",\"forword\":\"0\",\"comment\":\"5\",\"hotrate\":0.0,\"isOrigin\":true,\"piclist\":[\"http://ww2.sinaimg.cn/large/006lnJGbly1fawk43uf1nj30qo142n1z0.jpg\",\"http://ww3.sinaimg.cn/large/006lnJGbly1fawk44es24j30qo140grj0.jpg\"],\"userCertify\":0,\"classtitle\":\"\",\"sourcePlatform\":\"金立智能手机\",\"praise\":\"2\",\"userId\":\"5812757187\",\"userGender\":\"female\",\"userFollowCount\":\"205\",\"userFanCount\":\"56\",\"userWeiboCount\":\"372\",\"userLocation\":\"黑龙江哈尔滨\",\"userDescription\":\"Shero\"}";
//		String weibostr = "{\"Mid\":\"3875870903056847\",\"url\":\"http://weibo.com/1618051664/CvPKqcPdZ?refer_flag\u003d1001030103_\",\"userurl\":\"http://weibo.com/breakingnews?refer_flag\u003d1001030103_\",\"name\":\"头条新闻\",\"content\":\"\n\t\t#天津滨海爆炸#【“编外消防队”先入火场伤亡不明】财新网报道，从多方面确认，天津爆炸事故发生时，编制并不属于中国消防系统的天津港公安局消防支队三支队伍，先于公安消防官兵抵达现场救火。但爆炸发生后，他们的伤亡情况暂未被当地官方提及。附报道原文：|【独家】“...\n\t\t\",\"time\":\"1439563096000\",\"forword\":\"7343\",\"comment\":\"6392\",\"hotrate\":0.0,\"isOrigin\":true,\"piclist\":[\"http://ww4.sinaimg.cn/large/60718250jw1ev2ip5yg4bj20ox0ian0f.jpg\",\"http://ww1.sinaimg.cn/large/60718250jw1ev2ip66i6lj20ox0i1aeg.jpg\",\"http://ww1.sinaimg.cn/large/60718250jw1ev2ip6i7xhj20oy0irdkn.jpg\",\"http://ww3.sinaimg.cn/large/60718250jw1ev2ipbg35pj20qe0hnn17.jpg\",\"http://ww4.sinaimg.cn/large/60718250jw1ev2ipbhl9wj20qe0hnjvs.jpg\",\"http://ww1.sinaimg.cn/large/60718250jw1ev2ipbve2rj20qe0hn0v3.jpg\",\"http://ww4.sinaimg.cn/large/60718250jw1ev2ipgnte2j20qe0dwabz.jpg\",\"http://ww4.sinaimg.cn/large/60718250jw1ev2ipgvy6rj20qe0dwwin.jpg\",\"http://ww1.sinaimg.cn/large/60718250jw1ev2iph8jsej20qe0dwn09.jpg\"],\"userCertify\":0,\"classtitle\":\"\",\"sourcePlatform\":\"微博 weibo.com\",\"praise\":\"3167\",\"userId\":\"1618051664\",\"userGender\":\"female\",\"userFollowCount\":\"703\",\"userFanCount\":\"51230000\",\"userWeiboCount\":\"119862\",\"userLocation\":\"北京海淀区\",\"userDescription\":\"新浪新闻中心24小时播报全球重大新闻\"}";
		String weibostr = "{\"Mid\":\"3875882844106854\",\"url\":\"http://weibo.com/5329228691/CvQ3GhenA?refer_flag\u003d1001030103_\",\"userurl\":\"http://weibo.com/u/5329228691?refer_flag\u003d1001030103_\",\"name\":\"秋峰落叶123\",\"content\":\"\n\t\t#天津塘沽大爆炸#愿不再受伤\n\t\t\",\"time\":\"1439565943000\",\"forword\":\"0\",\"comment\":\"0\",\"hotrate\":0.0,\"isOrigin\":true,\"piclist\":[\"http://ww1.sinaimg.cn/large/005OETPtjw1ev2k3n8nooj307509f3z1.jpg\"],\"userCertify\":0,\"classtitle\":\"\",\"sourcePlatform\":\"OPPO智能手机\",\"praise\":\"0\",\"userId\":\"5329228691\",\"userGender\":\"female\",\"userFollowCount\":\"41\",\"userFanCount\":\"64\",\"userWeiboCount\":\"825\",\"userLocation\":\"广东江门\"}";
//		Gson gson = new Gson();
		WeiboEntity entity = gson.fromJson(weibostr, WeiboEntity.class);
		System.out.println(gson.toJson(entity));
		SingleWeiboFeatureExtractor extractor = SingleWeiboFeatureExtractor.getInstance();
		double [] features= extractor.extractFeature(entity);
		for(int i=0;i<features.length;i++){
			double flo = features[i];
			System.out.print(i+":"+flo+",");
		}
		//80.0,40.0,2.0,-0.8,0.0,0.0,0.0,2.0,0.0,1.0,1.0,0.0,200000.0,207.0,12750.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,7.0,0.0
		//89.0,58.0,0.0,0.5555555555555556,1.0,0.0,3.0,0.0,0.0,2.0,1.0,1.0,56.0,205.0,372.0,0.0,0.0,5.0,0.0,2.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
		//54,39,0,0.555556,0,0,0,0,0,0,0,0,3190773,364,624,1,3358,4665,0,0,0,1,1,1,0,1,1,0,0,0,1
		SingleRFClassifer classifer = null;
		try {
			classifer = SingleRFClassifer.getInstance();
			double result = classifer.predict(features);
			System.out.println(result);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}*/
	
}
/*
@attribute cntChar numeric
@attribute cntWord numeric
@attribute cntNegWor numeric
@attribute avgEmo numeric
@attribute cntP1 numeric
@attribute cntP3 numeric
@attribute cntPeo numeric
@attribute cntPla numeric
@attribute cntOrg numeric
@attribute cntPic numeric
@attribute hasImg {0,1}
@attribute hasMutImg {0,1}
@attribute cntUsrFan numeric
@attribute cntUsrFol numeric
@attribute cntUsrWei numeric
@attribute approveType {0,1}
@attribute cntFor numeric
@attribute cntCom numeric
@attribute cntQ numeric
@attribute cntE numeric
@attribute hasLoc numeric
@attribute hasUsrDesc numeric
@attribute usrGender numeric
@attribute hasEmo numeric
@attribute hasMutEmo numeric
@attribute cntEmo numeric
@attribute cntHappyEmoCnt numeric
@attribute avgScared numeric
@attribute avgEvil numeric
@attribute avgShock numeric*/