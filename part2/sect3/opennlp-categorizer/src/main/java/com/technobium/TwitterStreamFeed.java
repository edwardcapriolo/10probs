package com.technobium;


import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

import com.twitter.hbc.ClientBuilder;
import com.twitter.hbc.core.Constants;
import com.twitter.hbc.core.endpoint.StatusesSampleEndpoint;
import com.twitter.hbc.core.processor.StringDelimitedProcessor;
import com.twitter.hbc.httpclient.BasicClient;
import com.twitter.hbc.httpclient.auth.Authentication;
import com.twitter.hbc.httpclient.auth.OAuth1;

import io.teknek.feed.Feed;
import io.teknek.feed.FeedPartition;
import io.teknek.model.ITuple;

/**
 * Connects to twitters streaming API and produces tuples with a field status that hold
 * a JSON string that represents a status message (a tweet)
 * @author edward
 *
 */
public class TwitterStreamFeed extends Feed {
  
  public static final String CONSUMER_KEY = "twitter.stream.consumer.key";
  public static final String CONSUMER_SECRET = "twitter.stream.consumer.secret";
  public static final String TOKEN = "twitter.stream.token";
  public static final String SECRET = "twitter.stream.secret";

  public TwitterStreamFeed(Map<String, Object> properties) {
    super(properties);
  }

  @Override
  /**
   * Currently produces a single partition numbered 0
   * TODO: This needs to be shardable to make it "web scale"
   */
  public List<FeedPartition> getFeedPartitions() {
    List<FeedPartition> results = new ArrayList<FeedPartition>();
    results.add(new TwitterStreamFeedPartition(this, "0"));
    return results;
  }

  @Override
  public Map<String, String> getSuggestedBindParams() {
    return new HashMap<String,String>();
  }

}

class TwitterStreamFeedPartition extends FeedPartition {

  private BlockingQueue<String> queue = new LinkedBlockingQueue<String>(100);
  private BasicClient client;
  
  public TwitterStreamFeedPartition(Feed feed, String partitionId) {
    super(feed, partitionId);
  }

  @Override
  public void initialize() {
    StatusesSampleEndpoint endpoint = new StatusesSampleEndpoint();
    endpoint.stallWarnings(false);
    Authentication auth = new OAuth1((String) feed.getProperties().get(TwitterStreamFeed.CONSUMER_KEY), 
            (String) feed.getProperties().get(TwitterStreamFeed.CONSUMER_SECRET), 
            (String) feed.getProperties().get(TwitterStreamFeed.TOKEN), 
            (String) feed.getProperties().get(TwitterStreamFeed.SECRET));
    client = new ClientBuilder()
      .name("sampleExampleClient")
      .hosts(Constants.STREAM_HOST)
      .endpoint(endpoint)
      .authentication(auth)
      .processor(new StringDelimitedProcessor(queue))
      .build();
    client.connect();
  }

  @Override
  public boolean next(ITuple tupleRef) {
    if (client.isDone()) {
      System.out.println("Client connection closed unexpectedly: " + client.getExitEvent().getMessage());
      //todo maybe better to throw exception here
      return false;
    } else {
      String msg = null;
      try {
        msg = queue.take();
      } catch (InterruptedException e) {
        // TODO Auto-generated catch block
        e.printStackTrace();
      }
      tupleRef.setField("message", msg);
      boolean done =  client.isDone();
      if (done){
        System.out.println("Client connection closed unexpectedly: " + client.getExitEvent().getMessage());
      }
      return !done;
    }
  }

  @Override
  public void close() {
    client.stop();
  }

  @Override
  public boolean supportsOffsetManagement() {
    return false;
  }

  @Override
  public String getOffset() {
    return null;
  }

  @Override
  public void setOffset(String offset) {
  }
  
}