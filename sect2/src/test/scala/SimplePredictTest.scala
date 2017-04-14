import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row
import org.apache.spark.sql.SQLContext
import org.apache.spark.mllib.fpm.AssociationRules
import org.apache.spark.mllib.fpm.AssociationRules.Rule
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.SparkContext
import org.scalatest._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.sql.SparkSession

class SimplePredictTest extends FlatSpec {
  
  def sampleData(sparkSession: SparkSession): DataFrame = {
    return sparkSession.createDataFrame(
      Seq(
        (1.0, "300x250", "sitea", "male", 5 ),
        (0.0, "300x250", "siteb", "male", 5)) ) 
      .toDF("label", "adformat", "supplyvendor", "gender", "age")
  }
  
  def siteaRecord(sparkSession: SparkSession): DataFrame = {
    return sparkSession.createDataFrame(
      Seq((0.0, "300x250", "sitea", "male", 5)))
      .toDF("label", "adformat", "supplyvendor", "gender", "age")
  }
  
  def sitebRecord(sparkSession: SparkSession): DataFrame = {
    return sparkSession.createDataFrame(
      Seq((0.0, "300x250", "siteb", "male", 5)))
      .toDF("label", "adformat", "supplyvendor", "gender", "age")
  }  
  
  "Logistic regression" should "predict sitea and not siteb" in {
    val sparkSession = SparkSession.builder.appName("nas")
      .master("local[2]").getOrCreate()
    val modelLogic = new ModelLogic(SimplePredict.categoricalFeatures, 
        SimplePredict.featureColumns,
      new LogisticRegression() .setMaxIter(6).setThreshold(0.41)
      .setFitIntercept(true).setStandardization(true))
    val (fittedModel, indexes) = modelLogic.generateModelAndIndex(sparkSession, sampleData(sparkSession))
    fittedModel.summary.predictions.show()
    
    val predict1 = modelLogic.indexAndPredict(sparkSession, 
            siteaRecord(sparkSession), fittedModel, indexes)
    predict1.show()
    assert(predict1.collectAsList().get(0).get(8) == 1.0)
    
    val predict2 = modelLogic.indexAndPredict(sparkSession, 
            sitebRecord(sparkSession), fittedModel, indexes)
    predict2.show()
    assert(predict2.collectAsList().get(0).get(8) == 0.0)
  }
}
