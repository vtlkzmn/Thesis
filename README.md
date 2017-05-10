# Thesis
Model for STACC's Recommender System

P.S. Everything tested on Linux Mint, hope this is working on most UNIX systems, for other OS modify script commands:

`$ wget http://files.grouplens.org/datasets/movielens/ml-1m.zip && mkdir -p ~/data/movielens && unzip -d ~/data/movielens/ ml-1m.zip && rm ml-1m.zip`

For smaller dataset use this script (much faster, if you don't use any powerful server):

`$ wget http://files.grouplens.org/datasets/movielens/ml-100k.zip && mkdir -p ~/data/movielens && unzip -d ~/data/movielens/ ml-100k.zip && rm ml-100k.zip`

(with last option you need to modify base folder (and some filenames as well) inside the script)

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

If you need more instructions how to prepare environment not to spend time redirecting to different pages and installing too many tools-packages, please let me know (vtlkzmn@gmail.com)

Download conda from https://conda.io/miniconda.html and install it (https://conda.io/docs/install/quick.html#linux-miniconda-install).

Download and install Elasticsearch 5.2+.X https://www.elastic.co/guide/en/elasticsearch/reference/current/_installation.html (Extremely fast search engine, to provide online recommendations when there is no time to wait while model will train). 

Make sure that Elasticsearch is running as a service, for that download .deb package and it'll do the job itselt.
```
$ conda create --name virtual_environment_name python=3
$ source activate venv_name
$ pip install --upgrade gensim
$ pip install jupyter
$ pip install pandas
```

Vualaá! Now You are ready to test script as well
