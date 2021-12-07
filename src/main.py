import findspark
findspark.init()
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.mllib.evaluation import RegressionMetrics



class Main():
	def run(self):
		sc = SparkContext('local')
		spark = SparkSession(sc)
		data = spark.read.format("csv").option("header", "true").option("delimiter",";").load("winequality-white.csv")
		for i in data.columns:
			data=data.withColumn(i,data[i].cast(DoubleType()))

		vectorAssembler = VectorAssembler(inputCols = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol'], outputCol = 'features')
		vdata = vectorAssembler.transform(data)
		vdata = vdata.select(['features', 'quality'])

		scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
		vdata = scaler.fit(vdata).transform(vdata)
		splits = vdata.randomSplit([0.80, 0.20])
		testing = splits[1]
		loadedModel = LinearRegressionModel.load("lrModel")
		lrPredictions = loadedModel.evaluate(testing)
		print('RMSE for Linear Regression: ' + str(lrPredictions.rootMeanSquaredError))

		loadedModel = RandomForestRegressionModel.load("rfModel")
		rfPredictions = loadedModel.transform(testing)
		valuesAndPreds = rfPredictions.select("prediction","quality").rdd
		metrics = RegressionMetrics(valuesAndPreds)	
		print('RMSE for Random Forest Regression: ' + str(metrics.meanSquaredError))

		loadedModel = RandomForestClassificationModel.load("rfcModel")
		rfcPredictions = loadedModel.transform(testing)
		evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName='f1')
		f1 = evaluator.evaluate(rfcPredictions)
		evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName='weightedPrecision')
		precision = evaluator.evaluate(rfcPredictions)
		evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName='weightedRecall')
		recall = evaluator.evaluate(rfcPredictions)
		print('F1, Recall, Precision for Random Forest Classification: ' + str(f1) + " " + str(recall) + " " + str(precision))

if __name__ == "__main__":	
	Main().run()
