package com.technobium;

import io.teknek.daemon.BeLoudOperator;
import io.teknek.daemon.TeknekDaemon;
import io.teknek.feed.FixedFeed;
import io.teknek.plan.FeedDesc;
import io.teknek.plan.OperatorDesc;
import io.teknek.plan.Plan;
import io.teknek.util.MapBuilder;
import io.teknek.zookeeper.EmbeddedZooKeeperServer;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Map;
import java.util.Properties;
 
import org.codehaus.jackson.JsonGenerationException;
import org.codehaus.jackson.map.JsonMappingException;
import org.codehaus.jackson.map.ObjectMapper;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class EndToEndTest extends EmbeddedZooKeeperServer {

  TeknekDaemon td = null;
  Plan p;

  @Before
  public void setup() {
    Properties props = new Properties();
    props.put(TeknekDaemon.ZK_SERVER_LIST, zookeeperTestServer.getConnectString());
    td = new TeknekDaemon(props);
    td.init();
  }

  public static Map<String,Object> getCredentialsOrDie(){
    URL u = Thread.currentThread().getContextClassLoader().getResource("credentials.json");
    File f = new File(u.getFile());
    if (!f.exists()){
      throw new RuntimeException("credentials.json is not found. It must have twitter credentials to run integration tests");
    }
    ObjectMapper om = new ObjectMapper();
    Map<String,Object> result = null;
    try {
      result = om.readValue(f, Map.class);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return result;
  }
    
  @Test
  public void hangAround() throws JsonGenerationException, JsonMappingException, IOException {
    p = new Plan().withFeedDesc(new FeedDesc()
    		.withFeedClass(TwitterStreamFeed.class.getName()).withProperties(getCredentialsOrDie()));
    p.withRootOperator(new OperatorDesc(new EmitStatusAsTextOperator()).
    		withNextOperator(new OperatorDesc(new SentimentOperator())));
    
    p.setName("yell");
    p.setMaxWorkers(1);
    td.applyPlan(p);
    try {
      Thread.sleep(10000);
    } catch (InterruptedException e) {
      e.printStackTrace();
    }
  }

  @After
  public void shutdown() {
    td.deletePlan(p);
    td.stop();
  }
}