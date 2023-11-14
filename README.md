# Movie recommendation using ALS in Spark

The goal of this project is to use Alternating Least Squares (ALS) in Spark to recommend movies.

## File structure

<pre>
|- data/
   |- raw/
|- notebooks/
   |- spark_als_movie_recommendation.ipynb
   |- figures/
|- movie-recommender-als/
   |- custom_funcs.py
   |- config.py
|- README.md
</pre>

## Implementation

An ALS model was succesfully created in two stages. Firstly, hyperparameters were tuned using a local implementation of the model with a subset of the training data. Secondly, the final model was trained using all available data using the Google Cloud platform.