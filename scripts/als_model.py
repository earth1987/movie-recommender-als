import sys
from math import sqrt
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

INPUT_PATH = sys.argv[1] # parse file path from command line

if __name__ == '__main__':

    # Create SparkContext and SparkSession
    sc = SparkContext(appName="spark-submit demo")
    spark = SparkSession.builder.getOrCreate()

    # 1. Training/test split
    ratings_DF = spark.read.csv("gs://arctic-crawler-403811-storage/ratings.csv", header=True, inferSchema=True)
    (training_DF, test_DF) = ratings_DF.randomSplit([0.8, 0.2], seed=10)
    training_DF.createOrReplaceTempView('training_DF')
    test_DF.createOrReplaceTempView('test_DF')
    training_DF.write.csv("gs://arctic-crawler-403811-storage/movie_rec_171123_training_DF/", mode="overwrite", header=True)
    test_DF.write.csv("gs://arctic-crawler-403811-storage/movie_rec_171123_test_DF/", mode="overwrite", header=True)

    # 2. Baseline RMSE
    row_baseline = spark.sql("SELECT AVG(rating) FROM training_DF").first()
    baseline = row_baseline['avg(rating)']
    se_RDD = test_DF.rdd.map(lambda row: Row(se=pow(row['rating']-baseline, 2)))
    se_DF = spark.createDataFrame(se_RDD)
    se_DF.createOrReplaceTempView('se_DF')
    row_avg = spark.sql("SELECT AVG(se) FROM se_DF").first()
    baseline_rmse = sqrt(row_avg['avg(se)'])
    print(f"Baseline RMSE: {baseline_rmse}")

    # 3. Hyperparameter tuning
    als = ALS(maxIter=10, rank=10, regParam=0.05, userCol="userId", itemCol="movieId", ratingCol="rating", nonnegative=True, coldStartStrategy='drop', seed=1)
    param_grid = ParamGridBuilder().addGrid(als.rank, [5, 10, 15, 20]).addGrid(als.regParam, [0.01, 0.05, 0.1, 0.15, 0.2]).addGrid(als.maxIter, [5, 10, 15, 20]).build()
    reg_eval = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    kfold_cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=reg_eval, numFolds=5)
    cv_model = kfold_cv.fit(training_DF)

    # 4. Save best model
    param_map = zip(cv_model.getEstimatorParamMaps(), cv_model.avgMetrics)
    param_min = min(param_map, key=lambda x: x[1])
    print(f"Optimal configuration: {param_min}")

    best_model = cv_model.bestModel
    best_model.write().overwrite().save("gs://arctic-crawler-403811-storage/movie_rec_171123")

    # 5. Evaluate best model
    print("Training set performance")
    training_pred = best_model.transform(training_DF)
    for metric in ['rmse', 'mae', 'r2']:
      reg_eval = RegressionEvaluator(metricName=metric, labelCol="rating", predictionCol="prediction")
      reg_score = reg_eval.evaluate(training_pred)
      print(f"{metric}: {reg_score}")
    
    print("Test set performance")
    test_pred = best_model.transform(test_DF)
    for metric in ['rmse', 'mae', 'r2']:
      reg_eval = RegressionEvaluator(metricName=metric, labelCol="rating", predictionCol="prediction")
      reg_score = reg_eval.evaluate(test_pred)
      print(f"{metric}: {reg_score}")

    # 6. Shutdown Spark context
    sc.stop()
