# Thesis
Model for one Recommender System

# Benchmarks

|  [Movielens 1M](http://grouplens.org/datasets/movielens/1m) |  RMSE  |   MAE  |
|-----------------|:------:|:------:|
| [NormalPredictor](http://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.random_pred.NormalPredictor) | 1.5037 | 1.2051 |
| [BaselineOnly](http://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly)    |  .9086 | .7194 |
| [KNNBasic](http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBasic)        |  .9207 |  .7250 |
| [KNNWithMeans](http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithMeans)    |  .9292 |  .7386 |
| [KNNBaseline](http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline)     |  .8949 | .7063 |
| [SVD](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD)             |  .8738 |  .6858 |
| [NMF](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF)             |  .9155 |  .7232 |
| [Slope One](http://surprise.readthedocs.io/en/stable/slope_one.html#surprise.prediction_algorithms.slope_one.SlopeOne)             |  .9065 |  .7144 |
| [Co clustering](http://surprise.readthedocs.io/en/stable/co_clustering.html#surprise.prediction_algorithms.co_clustering.CoClustering)             |  .9155 |  .7174 |
| [Word2Vec](https://code.google.com/archive/p/word2vec/) |  .8761  |  .7168  |


*table is modified from origin

*Original: https://github.com/NicolasHug/Surprise


# Environment setup
1. Get the data:
`$ wget http://files.grouplens.org/datasets/movielens/ml-1m.zip && mkdir -p ~/data/movielens && unzip -d ~/data/movielens/ ml-1m.zip && rm ml-1m.zip`
2. Download and install Elasticsearch (tested on 5.2 and 6.3) https://www.elastic.co/guide/en/elasticsearch/reference/current/_installation.html 

3. Install dependencies
```
$ pip3 install gensim sklearn tqdm jupyter pandas numpy
```

Vualaá! Now You are ready to run the code.
