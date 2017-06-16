package com.technobium;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.doccat.DocumentSampleStream;
import opennlp.tools.util.PlainTextByLineStream;

public class OpenNLPCategorizer {
	DoccatModel model;

	public static void main(String[] args) {
		OpenNLPCategorizer twitterCategorizer = new OpenNLPCategorizer();
		twitterCategorizer.trainModel();
		twitterCategorizer.classifyNewTweet("Great tool.");
		twitterCategorizer.classifyNewTweet("The tool is ugly, I cant wait to return it.");
	}

	public void trainModel() {
		try (InputStream dataIn = new FileInputStream("input/tweets.txt")) {
			int cutoff = 2;
			int trainingIterations = 30;
			model = DocumentCategorizerME.train("en", 
					new DocumentSampleStream(new PlainTextByLineStream(dataIn, "UTF-8")), 
					cutoff, trainingIterations);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public void classifyNewTweet(String tweet) {
		DocumentCategorizerME myCategorizer = new DocumentCategorizerME(model);
		double[] outcomes = myCategorizer.categorize(tweet);
		String category = myCategorizer.getBestCategory(outcomes);
		if (category.equalsIgnoreCase("1")) {
			System.out.println("The tweet [" + tweet + "] is positive :) ");
		} else {
			System.out.println("The tweet [" + tweet + "] is negative :( ");
		}
	}
}
