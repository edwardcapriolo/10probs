
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import java.io.File

object SimplePredict {
  val categoricalFeatures = List("adformat", "supplyvendor", "gender")
  val featureColumns = List("adformatIndex", "supplyvendorIndex", "genderIndex", "age")
}

class ModelLogic(
        categoricalFeatures : List[String], 
        featureColumns: List[String], 
        lr: LogisticRegression) {
  
   /* Create indexers for each non numeric column */  
   def indexTransformers () : List[StringIndexer] = {
     return categoricalFeatures.map(
      cname => new StringIndexer()
      .setInputCol(cname)
      .setOutputCol(s"${cname}Index")
      .setHandleInvalid("skip")
    )
   }
   
   /* for each indexer run them producting models that will index future data sets */
   def indexerModels(trainingData : DataFrame) : List[StringIndexerModel] = {
     return indexTransformers().map { x => x.fit(trainingData) }
   }
   
   /* run each indexer and then drop the column that it indexed */
   def index(data: DataFrame, indexerModels : List[StringIndexerModel]) : DataFrame = {
     var copy = data
     for (indexerModel <- indexerModels){
       copy = indexerModel.transform(copy).drop(indexerModel.getInputCol)
     }
     return copy
   }
   
   /* create a single feature column from all the features */
   def featurize(data: DataFrame) : DataFrame = {
     val assembler = new VectorAssembler().setInputCols(featureColumns.toArray).setOutputCol("features")
     return assembler.transform(data)
   }
   
   def generateModelAndIndex(sparkSession:SparkSession, trainingData: DataFrame) 
      : (LogisticRegressionModel, List[StringIndexerModel]) = {
     val indexerModel = indexerModels(trainingData)
     val indexedData = index(trainingData, indexerModel)
     val featurizedData = featurize(indexedData)
     val fittedModel = lr.fit(featurizedData)
     return (fittedModel, indexerModel)
   }
   
   /* Given a dataset predic using the given model */
   def predict(data: DataFrame, fittedModel : LogisticRegressionModel) : DataFrame = {
     val summary = fittedModel.evaluate(data)
     return summary.predictions
   }
   
   /* make a prediction */
   def indexAndPredict(sparkSession:SparkSession, dataToPredict: DataFrame,
           lrModel: LogisticRegressionModel, indexers: List[StringIndexerModel]) : DataFrame = {
     val indexed = index(dataToPredict, indexers)
     val feat = featurize(indexed)
     return predict(feat, lrModel)
   }
}