package com.ict.mcg.processs;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.ict.mcg.event.EmotionAnalyze;
import com.ict.mcg.util.FileIO;
import com.ict.mcg.util.ICTSegmentation;



/**
 * Sentiment analysis based on lexicon
 * consider word intensity and context relation
 * @author senochow
 *
 */
public class SentimentAnalysis {
	private HashMap<String, Integer> positiveWords = new HashMap<String, Integer>();
	private HashMap<String, Integer> negativeWords = new HashMap<String, Integer>();
	private List<String> negationWords = new ArrayList<String>();
	private HashMap<String, Double> intensifierWords = new HashMap<String, Double>();
	private HashMap<String, Integer[]> sentimentWords =  new HashMap<String, Integer[]>();
	private int wordWindow = 2;
	private List<String> posWordcloud = new ArrayList<String>();
	private List<String> negWordcloud = new ArrayList<String>();
	private String posFile = FileIO.getFilePath() + "positiveWords.txt";
	private String negFile = FileIO.getFilePath() + "negativeWords.txt";
	private String negationFile = FileIO.getFilePath() + "negationWords.txt";
	private String intensifyFile = FileIO.getFilePath() + "intensifierWords.txt";
	private String sentimentFile = FileIO.getFilePath() + "sentimentWords.txt";
	
	private String posFile_resource = FileIO.getResourcePath() + "positiveWords.txt";
	private String negFile_resource = FileIO.getResourcePath() + "negativeWords.txt";
	private String negationFile_resource = FileIO.getResourcePath() + "negationWords.txt";
	private String intensifyFile_resource = FileIO.getResourcePath() + "intensifierWords.txt";
	private String sentimentFile_resource = FileIO.getResourcePath() + "sentimentWords.txt";
	/**
	 * Initialize, load file into memory
	 */
	
	public SentimentAnalysis(){
		loadFile();
	}
	public SentimentAnalysis(String posFile, String negFile, String negationFile, String intensifyFile){
		this.posFile = posFile;
		this.negFile = negFile;
		this.negationFile = negationFile;
		this.intensifyFile = intensifyFile;
		this.posFile_resource=this.negationFile_resource=this.negationFile_resource=this.intensifyFile_resource="";
		loadFile();
	}
	
	/**
	 * load file 
	 */
	private void loadFile(){
		try {
			InputStream is = this.getClass().getResourceAsStream(posFile_resource);
			if (is == null) {
				is = new FileInputStream(posFile);
			}
			
			BufferedReader readerPos = new BufferedReader(new InputStreamReader(is,"utf-8"));
			String line = "";
			while ((line = readerPos.readLine()) != null) {
				String[] wordVal = line.split("\t");
				positiveWords.put(wordVal[0], Integer.parseInt(wordVal[1]));
			}
			readerPos.close();
			is.close();
		} catch (IOException e) {
			System.out.println("load positiveWords.txt file failed!");
			e.printStackTrace();
		}
		try {
			InputStream is = this.getClass().getResourceAsStream(negFile_resource);
			if (is == null) {
				is = new FileInputStream(negFile);
			}
			BufferedReader readerNeg = new BufferedReader(new InputStreamReader(is,"utf-8"));
			String line = "";
			while ((line = readerNeg.readLine()) != null) {
				String[] wordVal = line.split("\t");
				negativeWords.put(wordVal[0], Integer.parseInt(wordVal[1]));
			}
			readerNeg.close();
			is.close();
		} catch (IOException e) {
			System.out.println("load negativeWords.txt file failed!");
			e.printStackTrace();
		}
		try {
			InputStream is = this.getClass().getResourceAsStream(negationFile_resource);
			if (is == null) {
				is = new FileInputStream(negationFile);
			}
			BufferedReader readerNega = new BufferedReader(new InputStreamReader(is,"utf-8"));
			String line = "";
			while ((line = readerNega.readLine()) != null) {
				negationWords.add(line);
			}
			readerNega.close();
			is.close();
		} catch (IOException e) {
			System.out.println("load negationWords.txt file failed!");
			e.printStackTrace();
		}
		try {
			InputStream is = this.getClass().getResourceAsStream(intensifyFile_resource);
			if (is == null) {
				is = new FileInputStream(intensifyFile);
			}
			BufferedReader readerInten = new BufferedReader(new InputStreamReader(is,"utf-8"));
			String line = "";
			while ((line = readerInten.readLine()) != null) {
				String[] wordVal = line.split("\t");
				intensifierWords.put(wordVal[0], Double.parseDouble(wordVal[1]));
			}
			readerInten.close();
			is.close();
		} catch (IOException e) {
			System.out.println("load intensifierWords.txt file failed!");
			e.printStackTrace();
		}
		try {
			InputStream is = this.getClass().getResourceAsStream(sentimentFile_resource);
			if (is == null) {
				is = new FileInputStream(sentimentFile);
			}
			BufferedReader readerSen = new BufferedReader(new InputStreamReader(is,"utf-8"));
			String line = "";
			while ((line = readerSen.readLine()) != null) {
				String[] wordVal = line.split("\t");
				Integer []vals = {Integer.parseInt(wordVal[1]),Integer.parseInt(wordVal[3])};
				sentimentWords.put(wordVal[0], vals);
			}
			readerSen.close();
			is.close();
		} catch (IOException e) {
			System.out.println("load sentimentWords.txt file failed!");
			e.printStackTrace();
		}
	}
	/**
	 * set word's emotion weight, Max emotion polarity is 9 ,
	 * according to PKU_Wan's empirical analysis, negative words usually 
	 * contribute more to the overall semantic orientation
	 * @param word
	 * @return
	 */
	private double getWordEmotionVal(String word){
		if (positiveWords.containsKey(word)) {
			this.posWordcloud.add(word);
			return (double)positiveWords.get(word)/9;
		}else if (negativeWords.containsKey(word)) {
			this.negWordcloud.add(word);
			return -1.2*(double)negativeWords.get(word)/9;
		}else {
			return 0.0;
		}
	}
	
	/**
	 * get emotionVal of each type in seven Types from one sentence by gc
	 * @param sentence
	 * @return
	 */
	//1:happy 2:fine 3:angry 4:sad 5:scared 6:evil 7:shock
	public double[] getEmotionEachTypeFromSentence(List<String> sentence){
		double []typeEmotionVal = new double[8];
		int []typeEmotionCnt = new int[8];
		int negCnt = 0;
		int wordCnt = sentence.size();
		if (wordCnt < this.wordWindow) {
			for (int i = 0; i < wordCnt; i++) {
				if(negationWords.contains(sentence.get(i))){
					negCnt+=1;
				}
				if (sentimentWords.containsKey(sentence.get(i))) {
					Integer []result = sentimentWords.get(sentence.get(i));				
					typeEmotionCnt[result[0]]++;
					typeEmotionVal[result[0]]+=typeEmotionVal[result[0]]/9;
				}
			}
		} else {
			for (int i = 0; i < wordCnt; i++) {
//					String word = sentence.get(i);
				if(negationWords.contains(sentence.get(i))){
					negCnt+=1;
				}
				if(sentimentWords.containsKey(sentence.get(i))){
					Integer []result = sentimentWords.get(sentence.get(i));
					int type = result[0];
					double val = result[1];
					if (val != 0) {
						int beginPos = 0;
						// set begin position of slider window
						if (i >= wordWindow) {
							beginPos = i - wordWindow;
						}
						val = getIntensifierValue(beginPos, i, sentence) * val;
						typeEmotionVal[type] += val;
						typeEmotionCnt[type]++;
					}
				}
			}
		}
		for(int i = 1;i< typeEmotionVal.length;i++){
			if(typeEmotionCnt[i]!=0){
				typeEmotionVal[i] = typeEmotionVal[i]/typeEmotionCnt[i];
			}
		}
		return typeEmotionVal;
	}
	/**
	 * get emotion from one sentence
	 * @param sentence
	 * @return
	 */
	public double getEmotionFromSentence(List<String> sentence){
		double emotionVal= 0.0;
		int emotionWordCnt = 0;
		if (sentence==null||sentence.size()==0) {
			return 0.0;
		}
		int wordCnt = sentence.size();
		if (wordCnt<this.wordWindow) {
			for (int i = 0; i < wordCnt; i++) {
				if (getWordEmotionVal(sentence.get(i))!=0) {
					emotionVal += getWordEmotionVal(sentence.get(i));
					emotionWordCnt++;
				}
			}
		}else {
			for (int i = 0; i < wordCnt; i++) {
				String word = sentence.get(i);
				double val = getWordEmotionVal(word);
				if (val!=0) {
					int beginPos = 0;
					//set begin position of slider window 
					if (i>=wordWindow) {
						beginPos = i-wordWindow;
					}
					if (isContainNegationWord(beginPos, i, sentence)) {
						val = -val;
					}
					val = getIntensifierValue(beginPos, i, sentence)*val;
					emotionVal += val;
					emotionWordCnt++;
				}
			}		
		}
		if (emotionWordCnt==0) {
			return 0;
		}else {
			return emotionVal/emotionWordCnt;
		}
		
	}
	
	/**
	 * get emotion from one word map with weight
	 * @param Map<String, Double> wordMap
	 * @return
	 */
	public double getEmotionFromWordMap(Map<String, Double> sentence){
		double emotionVal= 0.0;
//		int emotionWordCnt = 0;
		double totalWeight = 0.0;
		if (sentence==null||sentence.size()==0) {
			return 0.0;
		}
		int wordCnt = sentence.size();
//		ArrayList<Map.Entry<String, Double>> entry = sentence.entrySet();
		for (Map.Entry<String, Double> entry: sentence.entrySet()) {
			if (getWordEmotionVal(entry.getKey())!=0) {
				emotionVal += getWordEmotionVal(entry.getKey())*entry.getValue();
				totalWeight += entry.getValue();
			}
		}
		
		if (totalWeight == 0) {
			return 0.0;
		}else {
			return emotionVal/totalWeight;
		}
		
	}
	/**
	 * get emotion from one word map with weight
	 * @param Map<String, Integer> wordMap
	 * @return
	 */
	public double getEmotionFromWordIntegerMap(Map<String, Integer> sentence){
		double emotionVal= 0.0;
//		int emotionWordCnt = 0;
		Integer totalWeight = 0;
		if (sentence==null||sentence.size()==0) {
			return 0.0;
		}
		int wordCnt = sentence.size();
//		ArrayList<Map.Entry<String, Double>> entry = sentence.entrySet();
		for (Map.Entry<String, Integer> entry: sentence.entrySet()) {
			if (getWordEmotionVal(entry.getKey())!=0) {
				emotionVal += getWordEmotionVal(entry.getKey())*entry.getValue();
				totalWeight += entry.getValue();
			}
		}
		
		if (totalWeight == 0) {
			return 0.0;
		}else {
			return emotionVal/totalWeight;
		}
		
	}
	
	/**
	 * get emotion from one word map with weight
	 * @param Map<String, Integer> wordMap
	 * @return [pos_emo, neg_emo]
	 */
	public double[] getBothEmotionFromWordIntegerMap(Map<String, Integer> sentence){
		double[] emotionVal= new double[]{0.0, 0.0};
//		int emotionWordCnt = 0;
		Integer[] totalWeight = new Integer[]{0, 0};
		if (sentence==null||sentence.size()==0) {
			return emotionVal;
		}
		int wordCnt = sentence.size();
//		ArrayList<Map.Entry<String, Double>> entry = sentence.entrySet();
		for (Map.Entry<String, Integer> entry: sentence.entrySet()) {
			double emoVal = getWordEmotionVal(entry.getKey())*entry.getValue();
			if (emoVal>0) {
				emotionVal[0] += emoVal;
				totalWeight[0] += entry.getValue();
			}
			if (emoVal<0) {
				emotionVal[1] += emoVal;
				totalWeight[1] += entry.getValue();
			}
		}
		
		if (totalWeight[0] == 0) {
			emotionVal[0] = 0;
		} else {
			emotionVal[0] = emotionVal[0]/totalWeight[0];
		}
		if (totalWeight[1] == 0) {
			emotionVal[1] = 0;
		} else {
			emotionVal[1] = emotionVal[1]/totalWeight[1];
		}
		return emotionVal;
		
	}
	/**
	 * get sentiment value form a list of sentence list
	 * @param sentences
	 * @return
	 */
	public double getEmotionFromSentenceList(List<List<String>> sentences){
		if (sentences==null||sentences.size() ==0) {
			return 0;
		}
		double emotionVal = 0.0;
		for (List<String> list : sentences) {
			emotionVal += getEmotionFromSentence(list);
		}
		return emotionVal/sentences.size();
	}
	/**
	 * [beginPos, endPos) if contain negation words in this interval, consider multi-negation
	 * @param bPos
	 * @param ePos
	 * @param words
	 * @return
	 */
	private boolean isContainNegationWord(int bPos, int ePos,List<String> words){
		boolean m = false;
		for (int i = bPos; i < ePos; i++) {
			if (negationWords.contains(words.get(i))) {
//				System.out.println(words.get(i)+words.get(ePos));
				m = !m;
			}
		}
		return m;
	}
	/**
	 * [beginPos, endPos) if contain intensifier words in this interval, 
	 * if contains return the intensifier value ,else return 1
	 * @param bPos
	 * @param ePos
	 * @param words
	 * @return
	 */
	private double getIntensifierValue(int bPos, int ePos,List<String> words){
		double m = 1.0;
		for (int i = bPos; i < ePos; i++) {
			if (intensifierWords.containsKey(words.get(i))) {
//				System.out.println(words.get(i)+words.get(ePos));
				m *= intensifierWords.get(words.get(i));
			}
		}
		return m;
	}
	
	public List<String> getPosWordcloud(){
		List<String> wordcloud = new ArrayList<String>();
		wordcloud = getSortedWords(posWordcloud);
		return wordcloud;
	}
	public List<String> getNegWordcloud(){
		List<String> wordcloud = new ArrayList<String>();
		wordcloud = getSortedWords(negWordcloud);
		return wordcloud;
	}
	public List<String> getEmotionWordCloud() {
		// TODO Auto-generated method stub
		List<String> allWordCloud = new ArrayList<String>();
		allWordCloud.addAll(this.posWordcloud);
		allWordCloud.addAll(this.negWordcloud);
		return allWordCloud;
	}
	
	private List<String> getSortedWords(List<String> words){
		List<String> resWordcloud = new ArrayList<String>();
		HashMap<String, Integer> wordcloud = new HashMap<String, Integer>();
		for (String string : words) {
			if (wordcloud.containsKey(string)) {
				int val = wordcloud.get(string);
				wordcloud.put(string, val+1);
			}else {
				wordcloud.put(string, 1);
			}
		}
		ICTSegmentation segmentation = new ICTSegmentation();
		List<Map.Entry<String, Integer>> sortedWordcloud = segmentation.getSortList(wordcloud);
		if (sortedWordcloud.size()<10) {
			for (int i = 0; i < sortedWordcloud.size(); i++) {
				resWordcloud.add(sortedWordcloud.get(i).getKey());
			}
		}else {
			for (int i = 0; i < 10 ; i++) {
				resWordcloud.add(sortedWordcloud.get(i).getKey());
			}
		}
		return resWordcloud;
	}
	
	public static void main(String[] args) {
		new SentimentAnalysis();
	}
}
