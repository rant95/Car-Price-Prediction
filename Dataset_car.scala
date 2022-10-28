// Databricks notebook source
// MAGIC %md
// MAGIC 
// MAGIC # Car Price Prediction
// MAGIC 
// MAGIC Lova Rant.
// MAGIC 
// MAGIC ------
// MAGIC For the project, we will clean the dataset car.csv, do an exploratory analysis of the data, Train  the models for the prediction of the price and compare the performance metrics.
// MAGIC 
// MAGIC ------
// MAGIC * Linear Regression
// MAGIC * Random Forest Regression
// MAGIC * Decision Tree Regression
// MAGIC 
// MAGIC 
// MAGIC 
// MAGIC 
// MAGIC This notebook is written in **Scala** so the default cell type is Scala. NOTE: The data and the data challenge is confidential and must not be shared or published!

// COMMAND ----------

// MAGIC %md
// MAGIC ## Import libraries

// COMMAND ----------

import org.apache.spark.sql.types.{StructType, StructField, StringType, IntegerType, FloatType};
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.{RegressionEvaluator}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoder}
import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row

//Model ML
import org.apache.spark.ml.regression.{DecisionTreeRegressor,DecisionTreeRegressionModel}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}

//import com.microsoft.ml.spark.{LightGBMRegressionModel,LightGBMRegressor}
//import ml.dmlc.xgboost4j.scala.spark.{XGBoostRegressionModel,XGBoostRegressor}

import org.apache.spark.ml.feature.{PCA, VectorAssembler, StandardScaler}
import org.apache.spark.ml.linalg.Vectors



// COMMAND ----------

// MAGIC %md
// MAGIC ## Import the dataset

// COMMAND ----------

val cclass = spark.table("default.car_data_csv").withColumn("transmission",col("Transmission").cast(StringType)).withColumn("year",col("year").cast(IntegerType))    

val dataScheme = (new StructType)
.add("model", StringType)
.add("year", IntegerType)
.add("price", IntegerType)
.add("transmission", StringType)
.add("mileage", IntegerType)
.add("fuelType", StringType)
.add("engineSize", FloatType)


//val cclass=sqlContext.read.schema(dataScheme).option("maxRowsInMemory",20).option("header", "true").csv("dbfs:/FileStore/tables/car_data.csv")

cclass.printSchema()
cclass.show(5)

// COMMAND ----------

display(cclass.summary())

// COMMAND ----------

// MAGIC %md
// MAGIC We calculated the summary statistics for all columns in the DataFrame.
// MAGIC We can see the output to return the count, 25rd percentile, 50th percentile, and 75th percentile.
// MAGIC With the column "model" we have only 1 model C Class so we can drop this column for the train because it's no impact for the prediction of the price

// COMMAND ----------

//Top 10 cars in the dataset cclass
cclass.orderBy(col("price").desc).show(10)

// COMMAND ----------

//Calculate disticnt count for every columns
val exprs = cclass.columns.map((_ -> "approx_count_distinct")).toMap
display(cclass.agg(exprs))

// COMMAND ----------

display(cclass.select(cclass.columns.map(c => count(when(col(c).isNull || col(c) === "" || col(c).isNaN, c)).alias(c)): _*))

// COMMAND ----------

// MAGIC %md
// MAGIC We have 5 missing values in the column "year" and 7 in "mileage".

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ## Clean dataset
// MAGIC 
// MAGIC I have two options to clean in the dataset. I can remove the rows with missing value (NULL) or I can calculate average of the 2 colonms with missing value and I fill gaps in the dataset.

// COMMAND ----------

//Calculate average age for filling gaps in dataset
val averageYear = cclass.select("year")
  .agg(avg("year"))
  .collect() match {case Array(Row(avg: Double)) => avg  case _ => 0}

//Calculate average fare for filling gaps in dataset
val averageMileage = cclass.select("mileage")
  .agg(avg("mileage"))
  .collect() match {
  case Array(Row(avg: Double)) => avg
  case _ => 0
}

val filledDF = cclass.na.fill(Map("mileage" -> averageMileage, "year" -> averageYear))

// COMMAND ----------

// MAGIC %md
// MAGIC ### Check the dataset

// COMMAND ----------

display(filledDF.select(filledDF.columns.map(c => count(when(col(c).isNull || col(c) === "" || col(c).isNaN, c)).alias(c)): _*))

// COMMAND ----------

// MAGIC %md
// MAGIC ## Exploratory Data Analysis

// COMMAND ----------

// MAGIC %md
// MAGIC ### Visualize the data

// COMMAND ----------

//We calculate mean price vs year per transmission
display(filledDF.orderBy(col("year").asc))

// COMMAND ----------

//We calculate mean price vs year per fuelType
display(filledDF.orderBy(col("year").asc)) //.filter($"year" >= 2012)

// COMMAND ----------

//We calculate mean price vs year per engineSize
display(filledDF.orderBy(col("year").asc))

// COMMAND ----------

// MAGIC %md
// MAGIC We are left-skewed distributions. There is a long tail in the negative direction on the number line. The mean is also to the left of the peak.

// COMMAND ----------

//the median is the 50th percentile.
filledDF.agg(expr("percentile(year, 0.5)").as("50_percentile or Median")).show()

// COMMAND ----------

display(filledDF.orderBy(col("year").asc))

// COMMAND ----------

// MAGIC %md
// MAGIC ### Scatter Plot

// COMMAND ----------

//We will calculate price vs year
display(filledDF.orderBy(col("year").asc))

// COMMAND ----------

// MAGIC %md
// MAGIC As the year increases, the price reases.

// COMMAND ----------

display(filledDF.orderBy(col("year").asc))

// COMMAND ----------

// MAGIC %md
// MAGIC As the year increases, the mileage decreases.

// COMMAND ----------

//We calculate the price vs the mileage
display(filledDF.orderBy(col("mileage").asc))

// COMMAND ----------

// MAGIC %md
// MAGIC As the mileage increases, the price decreases.

// COMMAND ----------

//Fuel types
display(filledDF.groupBy("fuelType").count())

// COMMAND ----------

// MAGIC %md
// MAGIC We have only 6 rows for the "other" fuel type that represents the electric model and 151 for Hybrid you have enough information to predict correctly the price for this specific car

// COMMAND ----------

// MAGIC %md
// MAGIC ### Statistics

// COMMAND ----------

//display(cclass.groupBy("mileage").count())
display(filledDF.describe("mileage"))

// COMMAND ----------

//the median is the 50th percentile.
filledDF.agg(expr("percentile(mileage, 0.5)").as("50_percentile or Median")).show()

// COMMAND ----------

display(filledDF.describe("price"))

// COMMAND ----------

//the median is the 50th percentile.
filledDF.agg(expr("percentile(price, 0.5)").as("50_percentile or Median")).show()

// COMMAND ----------

// MAGIC %md
// MAGIC ### Scaling the features
// MAGIC 
// MAGIC The variables can be classified as either categorical or quantitative. We converts categorical features into a binary vector.

// COMMAND ----------

val Variables = Array("fuelType","transmission","year","mileage") //
val categoricalIndexers = Variables
  .map(i => new StringIndexer().setHandleInvalid("skip").setInputCol(i).setOutputCol(i+"Index"))

val categoricalEncoders = Variables
  .map(e => new OneHotEncoder().setInputCol(e + "Index").setOutputCol(e + "Vec"))

val assembler = new VectorAssembler()
  .setInputCols(Array("year","mileage") ++ Variables.map(s => s+ "Vec") )
  .setOutputCol("features")

// COMMAND ----------

// MAGIC %md
// MAGIC We encode quantitative variables to Indexer variable to know the correlation between variable "price". The correlation evaluates the strength of the linear relationship between two variables. The pipeline assembles several steps that will be categorical Indexers, categorical Encoders, linear Vector  and quantitative Indexers.

// COMMAND ----------

val steps: Array[org.apache.spark.ml.PipelineStage] = categoricalIndexers ++ categoricalEncoders  ++ Array(assembler)
val pipeline = new Pipeline().setStages(steps)
val transformed = pipeline.fit(filledDF).transform(filledDF) //.select("price","features")

transformed.select("fuelTypeVec","transmissionVec","year","yearVec","mileageVec","features").show(3)

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ### Correlation
// MAGIC 
// MAGIC We have 3 categorical variables and 2 quantitative variables 

// COMMAND ----------

// MAGIC %md
// MAGIC Correlation between two variables

// COMMAND ----------

//Correlation between "fuelTypeIndex" and "price" the label
transformed.stat.corr("fuelTypeIndex","price")

// COMMAND ----------

transformed.stat.corr("transmissionIndex","price")

// COMMAND ----------

transformed.stat.corr("yearIndex","price")

// COMMAND ----------

transformed.stat.corr("mileageIndex","price")

// COMMAND ----------

// MAGIC %md
// MAGIC The 2 variables that correlate most with the label we need to predict are engineSizeIndex and fuelTypeIndex

// COMMAND ----------

// MAGIC %md
// MAGIC ##  PCA - Principal Component Analysis
// MAGIC 
// MAGIC We generated vectors containing 5 features of the dataset to predict "price".  We will be able to reduce the dimensionality of the dataset using PCA. We need to normalize the features first.

// COMMAND ----------

val pca = new PCA()
  .setInputCol("features")
  .setOutputCol("pcaFeatures")
  .setK(5)
.fit(transformed)
pca.explainedVariance

// COMMAND ----------

// MAGIC %md
// MAGIC Now We can compute the 2 principal components. The first eigenvector (and therefore the first principal component) in the ordered list of principal components explains 99% of the variance; the second explains 0.00001 %, and so on. We will check if we can reduce the dimension of the dataset

// COMMAND ----------

val pca = new PCA()
  .setInputCol("features")
  .setOutputCol("pcaFeatures")
  .setK(2)
.fit(transformed)
pca.explainedVariance

// COMMAND ----------

val pcaDF = pca.transform(transformed)
val result = pcaDF.select("pcaFeatures")
display(result)

// COMMAND ----------

// MAGIC %md
// MAGIC The result (RMSE) of the PCA dataset was lower than that of the completed dataset. Due to the fact that the dataset is small enough, we will keep the completed dataset.

// COMMAND ----------

// MAGIC %md
// MAGIC ## Split dataset

// COMMAND ----------

// MAGIC %md
// MAGIC We then split up our dataset 75/25 for training and validation. We create two training and test data. One to test the PCA and the other without the dimension reduction by PCA 

// COMMAND ----------

// Split Dataset with dimension reduction
val Array(trainingpca, testpca) = pcaDF.randomSplit(Array(0.75, 0.25), seed = 12345)
trainingpca.cache()
testpca.cache()

// COMMAND ----------

// MAGIC %md
// MAGIC We will compare the Root Mean Square Error between the dataset with and without dimension reduction. We convert only "fuelType","transmission","engineSize" into a binary vector because they are the most correlation with the label "price".

// COMMAND ----------

// Split Dataset without dimension reduction
val Variables = Array("fuelType","transmission") 
val categoricalIndexers = Variables
  .map(i => new StringIndexer().setHandleInvalid("skip").setInputCol(i).setOutputCol(i+"Index"))

val categoricalEncoders = Variables
  .map(e => new OneHotEncoder().setInputCol(e + "Index").setOutputCol(e + "Vec"))


val assembler = new VectorAssembler()
  .setInputCols(Array("year","mileage") ++ Variables.map(s => s+ "Vec") )
  .setOutputCol("features")

val steps: Array[org.apache.spark.ml.PipelineStage] = categoricalIndexers ++ categoricalEncoders  ++ Array(assembler)
val pipeline = new Pipeline().setStages(steps)
val transform = pipeline.fit(filledDF).transform(filledDF) //.select("price","features")


val Array(training, test) = transform.select("fuelType","transmission","year","mileage","price","features").randomSplit(Array(0.75, 0.25), seed = 12345)
training.cache()
test.cache()

// COMMAND ----------

// MAGIC %md
// MAGIC We applied to preprocess the data and now we'll convert the dataset to the form required by the LinearRegression classifier. The Regression is used to predict the values. Linear regression, Random Forest and Decision Tree are used for performing different tasks like price prediction.

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC 
// MAGIC ## Linear Regression
// MAGIC 
// MAGIC I create a pipeline with a Linear Regression  and a Parameter grid which executes a 5-fold Cross Validation at each Grid point. The grid itself contains 5 values for the elasticNetParam, 2 for maxIter and 2 for regParam, i.e. a total of 5 * 2 * 2=20 points in the Hyperparameter space. This pipeline will return the best model using the cv.fit() call.

// COMMAND ----------

// MAGIC %md
// MAGIC ### Hyperparemater tuning

// COMMAND ----------

val lrpca = new LinearRegression()
  .setLabelCol("price")
  .setFeaturesCol("pcaFeatures")

// We use a ParamGridBuilder to construct a grid of parameters to search over.
// TrainValidationSplit will try all combinations of values and determine best model using
// the evaluator.
val paramGrid = new ParamGridBuilder()
  .addGrid(lrpca.regParam, Array(0.5,0.1,0.01))
  .addGrid(lrpca.fitIntercept)
  .addGrid(lrpca.elasticNetParam, Array(00.5,0.7, 1.0))
  .build()

// COMMAND ----------

val numFolds = 5

val evaluator = new RegressionEvaluator()
  .setLabelCol("price")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

val cv = new CrossValidator() 
      .setEstimator(lrpca) 
      .setEvaluator(evaluator) 
      .setEstimatorParamMaps(paramGrid) 
      .setNumFolds(numFolds) 



// COMMAND ----------

// MAGIC %md
// MAGIC ### Train Model
// MAGIC Now I create a pipeline containing VectorAssembler, PCA and LinearRegression and pass our data-frame as my input.

// COMMAND ----------

val cvModelpca = cv.fit(trainingpca)
cvModelpca.getEstimatorParamMaps.zip(cvModelpca.avgMetrics)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Test model and predict

// COMMAND ----------

// Make predictions on test documents. cvModel uses the best model found (cvModel).
val cvPredictionLRpca = cvModelpca.transform(testpca)
val rmse = evaluator.evaluate(cvPredictionLRpca)
println(rmse)

// COMMAND ----------

// MAGIC %md
// MAGIC Train without PCA dimension reduction

// COMMAND ----------

val lr = new LinearRegression()
  .setLabelCol("price")
  .setFeaturesCol("features")

// We use a ParamGridBuilder to construct a grid of parameters to search over.
// TrainValidationSplit will try all combinations of values and determine best model using
// the evaluator.
val paramGrid = new ParamGridBuilder()
  .addGrid(lr.regParam, Array(0.5,0.1,0.01))
  .addGrid(lr.fitIntercept)
  .addGrid(lr.elasticNetParam, Array(00.5,0.7, 1.0))
  .build()


// COMMAND ----------

// MAGIC %md
// MAGIC A CrossValidator performs k-fold cross-validation and grid search for hyperparameter tuning and model selection. For this project, I use  k=5 folds, k-fold cross-validation will generate 5 training and test dataset pairs. It selects the hyperparameters with the best accuracy

// COMMAND ----------

val numFolds = 5

val evaluator = new RegressionEvaluator()
  .setLabelCol("price")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

val cv = new CrossValidator() 
      .setEstimator(lr) 
      .setEvaluator(evaluator) 
      .setEstimatorParamMaps(paramGrid) 
      .setNumFolds(numFolds) 

// COMMAND ----------

//Train Model
val cvModel = cv.fit(training)
cvModel.avgMetrics

// COMMAND ----------

cvModel.getEstimatorParamMaps.zip(cvModel.avgMetrics)

// COMMAND ----------

//val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
//val parms = bestModel.stages.last.asInstanceOf[LinearRegressionModel].explainParams

// COMMAND ----------

// Make predictions on test documents. cvModel uses the best model found (cvModel).
val cvPredictionLR = cvModel.transform(test)
val rmse = evaluator.evaluate(cvPredictionLR)
println(rmse)

// COMMAND ----------

// Select (prediction, true label) and compute test error for test data
val rmse  = evaluator.evaluate(cvPredictionLR)
println("Root Mean Squared Error (RMSE) on test data = "  + rmse )
// Select (prediction, true label) and compute test error.
val evaluator_r2 = new RegressionEvaluator().setLabelCol("price").setPredictionCol("prediction")
  .setMetricName("r2")
val r2 = evaluator_r2.evaluate(cvPredictionLR)
println("R-squared (r^2) on test data = " + r2)
// Select (prediction, true label) and compute test error.
val evaluator_mae = new RegressionEvaluator().setLabelCol("price").setPredictionCol("prediction")
  .setMetricName("mae")
val mae = evaluator_mae.evaluate(cvPredictionLR)
println("Mean Absolute Error (MAE) on test data = " + mae)
// Select (prediction, true label) and compute test error.
val evaluator_mse = new RegressionEvaluator().setLabelCol("price").setPredictionCol("prediction")
  .setMetricName("mse")
val mse = evaluator_mse.evaluate(cvPredictionLR)
println("Mean Squared Error (MSE) on test data = " + mse)

// COMMAND ----------

val modeltest = cvPredictionLR.withColumn("per_error", (($"prediction"-$"price" )/$"price") * lit(100))
modeltest.select(avg(abs($"per_error")),min(abs($"per_error")), max(abs($"per_error"))).show()


// COMMAND ----------

display(modeltest.select("price","prediction","per_error"))


// COMMAND ----------

// MAGIC %md
// MAGIC ## Random Forest Regressor

// COMMAND ----------

val NumTrees = Seq(15,20,25)  
val MaxBins = Seq(23,27,30)  

val MaxIter: Seq[Int] = Seq(20) 
val MaxDepth: Seq[Int] = Seq(20) 

// COMMAND ----------

//import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
val rf = new RandomForestRegressor()
  .setLabelCol("price")
  .setFeaturesCol("features")

val paramGrid = new ParamGridBuilder() 
      .addGrid(rf.numTrees, NumTrees) 
      .addGrid(rf.maxDepth, MaxDepth) 
      .addGrid(rf.maxBins, MaxBins) 
      .build() 

// COMMAND ----------

// MAGIC %md
// MAGIC CrossValidator Spark also offers for hyper-parameter tuning with K=5 folds

// COMMAND ----------

val numFolds = 5 

val evaluator = new RegressionEvaluator()
  .setLabelCol("price")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

val cv = new CrossValidator() 
      .setEstimator(rf) 
      .setEvaluator(evaluator) 
      .setEstimatorParamMaps(paramGrid) 
      .setNumFolds(numFolds) 

// COMMAND ----------

// Train Model
val cvModel = cv.fit(training) 

// COMMAND ----------

val avgMetricsParamGrid = cvModel.avgMetrics
val combined = paramGrid.zip(avgMetricsParamGrid)

// COMMAND ----------

val avgMetricsParamGrid = cvModel.avgMetrics
val combined = paramGrid.zip(avgMetricsParamGrid)

// COMMAND ----------

cvModel.getEstimatorParamMaps.zip(cvModel.avgMetrics)

// COMMAND ----------


//val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
//val parms = bestModel.stages.last.asInstanceOf[RandomForestRegressionModel].explainParams

// COMMAND ----------

// Make predictions on test documents. cvModel uses the best model found (cvModel).
val cvPredictionRF = cvModel.transform(test)
val rmse = evaluator.evaluate(cvPredictionRF)
println(rmse)

// COMMAND ----------

// Select (prediction, true label) and compute test error.
val rmse  = evaluator.evaluate(cvPredictionRF)
println("Root Mean Squared Error (RMSE) on test data = "  + rmse )
// Select (prediction, true label) and compute test error.
val evaluator_r2 = new RegressionEvaluator().setLabelCol("price").setPredictionCol("prediction")
  .setMetricName("r2")
val r2 = evaluator_r2.evaluate(cvPredictionRF)
println("R-squared (r^2) on test data = " + r2)
// Select (prediction, true label) and compute test error.
val evaluator_mae = new RegressionEvaluator().setLabelCol("price").setPredictionCol("prediction")
  .setMetricName("mae")
val mae = evaluator_mae.evaluate(cvPredictionRF)
println("Mean Absolute Error (MAE) on test data = " + mae)
// Select (prediction, true label) and compute test error.
val evaluator_mse = new RegressionEvaluator().setLabelCol("price").setPredictionCol("prediction")
  .setMetricName("mse")
val mse = evaluator_mse.evaluate(cvPredictionRF)
println("Mean Squared Error (MSE) on test data = " + mse)

// COMMAND ----------


val modeltest_rf2 = cvPredictionRF.withColumn("per_error", (($"prediction"-$"price" )/$"price") * lit(100))
modeltest_rf2.select(avg(abs($"per_error")),min(abs($"per_error")), max(abs($"per_error"))).show()

// COMMAND ----------

display(modeltest_rf2.select("price","prediction","per_error"))

// COMMAND ----------

// MAGIC %md
// MAGIC ## Decision Tree Regression

// COMMAND ----------

//import org.apache.spark.ml.regression.DecisionTreeRegressionModel
//import org.apache.spark.ml.regression.DecisionTreeRegressor
val dt = new DecisionTreeRegressor()
  .setLabelCol("price")
  .setFeaturesCol("features")



// COMMAND ----------

val numFolds = 5 

val evaluator = new RegressionEvaluator()
  .setLabelCol("price")
  .setPredictionCol("prediction")
  .setMetricName("rmse")

val paramGrid = new ParamGridBuilder()
    .addGrid(dt.maxBins, Array(40,60))
    .addGrid(dt.maxDepth, Array(20,30))
   // .addGrid(dt.impurity, Array("entropy", "gini"))
    .build()

val cv = new CrossValidator() 
      .setEstimator(dt) 
      .setEvaluator(evaluator) 
      .setEstimatorParamMaps(paramGrid) 
      .setNumFolds(numFolds) 

// COMMAND ----------

// Train Model
val cvModel = cv.fit(training) 

// COMMAND ----------

val avgMetricsParamGrid = cvModel.avgMetrics
val combined = paramGrid.zip(avgMetricsParamGrid)

// COMMAND ----------

cvModel.getEstimatorParamMaps.zip(cvModel.avgMetrics)

// COMMAND ----------

//val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
//val parms = bestModel.stages.last.asInstanceOf[DecisionTreeRegressionModel].explainParams

// COMMAND ----------

// Make predictions on test documents. cvModel uses the best model found (cvModel).
val cvPredictionDT = cvModel.transform(test)

// COMMAND ----------

// Select (prediction, true label) and compute test error.
val rmse  = evaluator.evaluate(cvPredictionDT)
println("Root Mean Squared Error (RMSE) on test data = "  + rmse )
// Select (prediction, true label) and compute test error.
val evaluator_r2 = new RegressionEvaluator().setLabelCol("price").setPredictionCol("prediction")
  .setMetricName("r2")
val r2 = evaluator_r2.evaluate(cvPredictionDT)
println("R-squared (r^2) on test data = " + r2)
// Select (prediction, true label) and compute test error.
val evaluator_mae = new RegressionEvaluator().setLabelCol("price").setPredictionCol("prediction")
  .setMetricName("mae")
val mae = evaluator_mae.evaluate(cvPredictionDT)
println("Mean Absolute Error (MAE) on test data = " + mae)
// Select (prediction, true label) and compute test error.
val evaluator_mse = new RegressionEvaluator().setLabelCol("price").setPredictionCol("prediction")
  .setMetricName("mse")
val mse = evaluator_mse.evaluate(cvPredictionDT)
println("Mean Squared Error (MSE) on test data = " + mse)

// COMMAND ----------

val modeltest_dt = cvPredictionDT.withColumn("per_error", (($"prediction"-$"price" )/$"price") * lit(100))
modeltest_dt.select(avg(abs($"per_error")),min(abs($"per_error")), max(abs($"per_error"))).show()

// COMMAND ----------

display(modeltest_dt.select("price","prediction","per_error"))

// COMMAND ----------

// MAGIC %md
// MAGIC 
// MAGIC ## Conclusion
// MAGIC 
// MAGIC Here is the summary of the 3 algorithms with the use of K-fold =5
// MAGIC 
// MAGIC -----
// MAGIC 
// MAGIC * Random Forest  Regressor:     
// MAGIC 
// MAGIC RMSE train data= 2764  
// MAGIC 
// MAGIC RMSE test data= 2813    
// MAGIC 
// MAGIC r2=0.90   
// MAGIC 
// MAGIC MAE=2043  
// MAGIC 
// MAGIC MSE=7914746
// MAGIC 
// MAGIC 
// MAGIC 
// MAGIC ---
// MAGIC 
// MAGIC * Decison Tree Regression:      
// MAGIC 
// MAGIC RMSE train data= 3134 
// MAGIC 
// MAGIC RMSE test data= 2922  
// MAGIC 
// MAGIC r2=0.895  
// MAGIC 
// MAGIC MAE= 2111 
// MAGIC 
// MAGIC MSE=8538118
// MAGIC 
// MAGIC 
// MAGIC ----
// MAGIC * Linear Regression: 
// MAGIC 
// MAGIC RMSE train data= 3626 
// MAGIC 
// MAGIC RMSE test data= 3989  
// MAGIC 
// MAGIC R2=0.804  
// MAGIC 
// MAGIC MAE=2688  
// MAGIC 
// MAGIC MSE=1591376
// MAGIC 
// MAGIC ----
// MAGIC 
// MAGIC 
// MAGIC Overall, Random Forest is mostly fast, simple and flexible, but not without some limitations because Root-mean-square deviation, Mean Square Error and Mean absolute error have the lowest error values of the 3 models. The R2 value tells us that the predictor variables in the model ("fuelType","transmission","engineSize") are able to explain 90% of the variation in the car prices. This is not bad. However the Decision tree is very close to the performance of the Random Forest. I can improve my model with more k folds and more hyperparameters but also with more information. I also tested the xgboost and logitical regression models but the databricks community version did not have enough memory to run them.
