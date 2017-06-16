package com.technobium;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.concurrent.atomic.AtomicLong;

import io.teknek.model.ITuple;
import io.teknek.model.Operator;
import opennlp.tools.doccat.DoccatModel;
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.doccat.DocumentSampleStream;
import opennlp.tools.util.PlainTextByLineStream;

public class SentimentOperator extends Operator {

	public static final AtomicLong bad = new AtomicLong(0);
	public static final AtomicLong good = new AtomicLong(0);
	static DoccatModel model;
	{
		try (InputStream dataIn = new FileInputStream("input/tweets.txt")) {
			int cutoff = 2;
			int trainingIterations = 30;
			model = DocumentCategorizerME.train("en",
					new DocumentSampleStream(new PlainTextByLineStream(dataIn, "UTF-8")), cutoff, trainingIterations);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public void handleTuple(ITuple tuple) {
		classifyNewTweet((String) tuple.getField("statusAsText"));
	}

	public void classifyNewTweet(String tweet) {
		DocumentCategorizerME myCategorizer = new DocumentCategorizerME(model);
		double[] outcomes = myCategorizer.categorize(tweet);
		String category = myCategorizer.getBestCategory(outcomes);
		if (category.equalsIgnoreCase("1")) {
			good.incrementAndGet();
		} else {
			System.out.println("The tweet [" + tweet + "] is negative :( ");
			System.out.println("bad: " + bad.incrementAndGet() + " good: " + good);
		}
	}
}
