import os
import json
import math
import requests
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from scipy import sparse
from random import shuffle
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

def get_similars(map_items, item_sims, map_items_reverse, item, truncate_model=200):
    item_index = map_items[item]
    sims = [(i, k) for i, k in enumerate(item_sims[item_index]) if i != item_index]
    top_similar = sorted(sims, key=lambda x: x[1], reverse=True)[:truncate_model]

    top_similar_items = [map_items_reverse[i[0]] for i in top_similar]
    top_similar_scores = [i[1] for i in top_similar]

    return top_similar_items, top_similar_scores


def add_to_elasticsearch(data_str):
    base_url = r'{0}/_bulk'.format(elastic_url)
    response = requests.put(base_url, data=data_str)
    if response.status_code == 400:
        print('-- Not good... ')
        print(response.text)


def bulk_data(data, bulk_size=100):
    data = list(data)
    bulk_index = 0
    while bulk_index < len(data):
        bulk = data[bulk_index:bulk_index+bulk_size]
        bulk_index += bulk_size
        yield bulk


class NumpyDecoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyDecoder, self).default(obj)
        
def score_item_to_user_w2v(item_id, user_scored_items, w2v_model, w2v_score_time):
    score = 0.0
    w_acc = 0.0
    for item in user_scored_items:
        start = time()
        item_score = w2v_model.wv.similarity(item, item_id)
        w2v_score_time += time() - start
        score += user_scored_items[item] * item_score
        w_acc += abs(item_score)
    if w_acc:
        score = (score / w_acc) + movies_mean_rating[item_id]
    else:
        score = round(movies_mean_rating[item_id] * 2) / 2
    return score

def score_item_to_user_cos_sim(item_id, user_scored_items, cos_sim_matrix, cs_score_time):
    score = 0.0
    w_acc = 0.0
    for item in user_scored_items:
        start = time()
        item1 = map_movies_reverse[item_id]
        item2 = map_movies_reverse[item]
        item_score = cos_sim_matrix[item1][item2]
        cs_score_time += time() - start
        score += user_scored_items[item] * item_score
        w_acc += abs(item_score)
    if w_acc:
        score = (score / w_acc) + movies_mean_rating[item_id]
    else:
        score = round(movies_mean_rating[item_id] * 2) / 2
    return score

def process_user(user, user_id, test_sets, train_sets, item_user_matrices, i_u_mat_time):
    shuffle(user)
    len_user = len(user)
    for test_set in test_sets:
        test_set.append({})
    for train_set in train_sets:
        train_set.append([{}, {}])
    k = 0
    switchers = [int(len_user * x / 5) for x in range(1, 5)]
    for i in range(len_user):
        if i in switchers:
            k += 1
        test_sets[k][user_id][str(user[i][1])] = user[i][2]
        temp_train_sets = [temp_set for h, temp_set in enumerate(train_sets) if h != k]
        start = time()
        temp_item_user_matrices = [mat for h, mat in enumerate(item_user_matrices) if h != k]
        i_u_mat_time += time() - start
        for train_set in temp_train_sets:
            if user[i][2] >= 4:
                train_set[user_id][0][str(user[i][1])] = user[i][2] - movies_mean_rating[str(user[i][1])]
            elif user[i][2] <= 3:
                train_set[user_id][1][str(user[i][1])] = user[i][2] - movies_mean_rating[str(user[i][1])]
        start = time()
        for i_u_mat in temp_item_user_matrices:
            i_u_mat[map_movies_reverse[item], user_id-1] = user[i][2] - movies_mean_rating[str(user[i][1])]
        i_u_mat_time += time() - start
    user = []
    
    return user, user_id, test_sets, train_sets, item_user_matrices, i_u_mat_time


def add_model_to_elastic(w2v_model):
    data_list = []
    for i, movie_id in enumerate(tqdm(movie_ids)):
        try:
            top = w2v_model.wv.most_similar(str(movie_id), topn=200)
            top_similar_items, top_similar_scores = [], []
            for each in top:
                top_similar_items.append(each[0])
                top_similar_scores.append(each[1])
            movie_data = {}
            movie_data['item_id'] = movie_id
            movie_data['label'] = movie_titles[i]
            movie_data['similars'] = top_similar_items
            movie_data['score_similars'] = top_similar_scores
            movie_data['metadata'] = {}
            movie_data['metadata']['genres'] = movie_genres[i].split('|')
            data_list.append(movie_data) 
        except KeyError as e:
            pass
    _index = 'b_thesis'
    model_version = 'item2vec'
    _type = 'item_model_{0}'.format(model_version)

    print('- Add items and metadata to elastic ...')
    n_total = len(data_list)
    n_count = 0
    print('... total elements to add: {0}'.format(n_total))
    for bulk in bulk_data(data_list):
        data = []
        for doc in bulk:
            n_count += 1
            if n_count % (n_total / 10) == 0:
                print('Total: {0} %'.format(int((100.0 * n_count) / n_total)))
            index = {"index": {"_index": _index, "_type": _type, "_id": n_count}}
            index = json.dumps(index)
            document = json.dumps(doc, cls=NumpyDecoder)
            data.extend([index, document])
        # Add bulk to elastic search
        data_str = '\n'.join(data)
        data_str += '\n'
        add_to_elasticsearch(data_str)
    return True
        
def optimize_param():
    tested = []
    for i in tqdm(range(5)):
        for epochss in tqdm([1,20,50, 100]):  # or any other param
            W2VRMSE = W2VMAE =  0.0
            w2v_model = Word2Vec(item_sentences, size=6, window=5, min_count=1, hs=1, workers=6, sg=1)
            w2v_model.build_vocab(vocab, update=True)
            w2v_model.train(item_sentences, total_examples=w2v_model.corpus_count, epochs=epochss)
            w2v_model.init_sims()

            w2v_final_rmse = 0.0
            w2v_final_mae = 0.0
            k = 0
            for i in range(1,len(test)):
                if i % 100 == 1: # using only 1% of test_set, otherwise it takes too long
                    w2v_ratings_sum = 0.0
                    w2v_abs_sum = 0.0
                    test_user = test_1[i]
                    for item in test_user:
                        actual = test_user[item]
                        predicted_w2v = score_item_to_user_w2v(item, {**train_1[i][0], **train_1[i][1]}, w2v_model)
                        w2v_abs_sum += abs(predicted_w2v - actual)
                        w2v_ratings_sum += (predicted_w2v - actual)**2
                    w2v_user_rmse = math.sqrt(w2v_ratings_sum / len(test_user))
                    w2v_user_mae = w2v_abs_sum/len(test_user)
                    w2v_final_rmse += w2v_user_rmse
                    w2v_final_mae += w2v_user_mae
                    k+=1
            W2VRMSE = w2v_final_rmse/k
            W2VMAE = w2v_final_mae/k
            tested.append((epochss,(W2VRMSE, W2VMAE)))
    return tested


def main():
    base_dir = '~/data/movielens/ml-1m'

    df_movies = pd.read_csv(os.path.join(base_dir, 'movies.dat'), delimiter='::', header=None,
                            names=['movieId', 'title', 'genres'], engine='python')
    df_ratings = pd.read_csv(os.path.join(base_dir, 'ratings.dat'), delimiter='::', header=None,
                             names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')

    users = df_ratings['userId'].values
    items = df_ratings['movieId'].values
    ratings = df_ratings['rating'].values

    movie_ids = df_movies['movieId'].values
    movie_titles = df_movies['title'].values
    movie_genres = df_movies['genres'].values


    map_movies_str = {}
    for i, movie_id in enumerate(tqdm(movie_ids)):
        map_movies_str[i] = str(movie_id)

    map_movies_reverse = {}
    for i, movie_id in enumerate(tqdm(movie_ids)):
        map_movies_reverse[str(movie_id)] = i

    vocab = []
    movies_mean_rating = {}

    for row in tqdm(list(zip(df_ratings['userId'], df_ratings['movieId'], df_ratings['rating']))):
        item = str(row[1])
        rating = row[2]
        if item not in movies_mean_rating:
            movies_mean_rating[item] = (1, rating)
        else:
            movies_mean_rating[item] = (movies_mean_rating[item][0] + 1, movies_mean_rating[item][1] + rating)

    for movie_id in movie_ids:
        movie_id = str(movie_id)
        vocab.append([movie_id])
        try:
            movies_mean_rating[movie_id] = movies_mean_rating[movie_id][1] / movies_mean_rating[movie_id][0]
        except KeyError:
            movies_mean_rating[movie_id] = 2.5
    unique_genres = set([g for i in movie_genres for g in i.split('|')])


    user = []
    user_id = 1
    # Test_sets are lists of dicts (user items)
    test_1, test_2, test_3, test_4, test_5 = [{}], [{}], [{}], [{}], [{}]

    # Train_sets are lists of lists(users) of two dicts (positive and negative items respectively)
    train_1, train_2, train_3, train_4, train_5 = [[{}, {}]], [[{}, {}]], [[{}, {}]], [[{}, {}]], [[{}, {}]]
    test_sets = [test_1, test_2, test_3, test_4, test_5]
    train_sets = [train_1, train_2, train_3, train_4, train_5]

    item_user_matrix_1 = sparse.lil_matrix((len(movie_ids), len(users)), dtype=np.float32)
    item_user_matrix_5 = item_user_matrix_4 = item_user_matrix_3 = item_user_matrix_2 = item_user_matrix_1.copy()
    item_user_matrices = [item_user_matrix_1, item_user_matrix_2, item_user_matrix_3, item_user_matrix_4, item_user_matrix_5]
    i_u_mat_time = 0.0
    for row in tqdm(list(zip(df_ratings['userId'], df_ratings['movieId'], df_ratings['rating']))):
        new_user_id = row[0]
        item = str(row[1])
        rating = row[2]
        if new_user_id != user_id:
            user, user_id, test_sets, train_sets, item_user_matrices, i_u_mat_time = process_user(user, user_id, test_sets, train_sets, item_user_matrices, i_u_mat_time)
        user.append(row)
        user_id = new_user_id

    # Last user
    user, user_id, test_sets, train_sets, item_user_matrices, i_u_mat_time = process_user(user, user_id, test_sets, train_sets, item_user_matrices, i_u_mat_time)

    start = time()
    item_similarities_1 = cosine_similarity(item_user_matrix_1, item_user_matrix_1, dense_output=True)
    item_similarities_2 = cosine_similarity(item_user_matrix_2, item_user_matrix_2, dense_output=True)
    item_similarities_3 = cosine_similarity(item_user_matrix_3, item_user_matrix_3, dense_output=True)
    item_similarities_4 = cosine_similarity(item_user_matrix_4, item_user_matrix_4, dense_output=True)
    item_similarities_5 = cosine_similarity(item_user_matrix_5, item_user_matrix_5, dense_output=True)
    item_sims = [item_similarities_1, item_similarities_2, item_similarities_3, item_similarities_4, item_similarities_5]
    i_u_mat_time += time() - start



    w2v_time, cs_time , w2v_score_time, cs_score_time, w2v_model_time, W2VRMSE, W2VMAE, COSSIMRMSE, COSSIMMAE = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for k in range(5):
        train = train_sets[k]
        test = test_sets[k]
        cos_sim_matrix = item_sims[k]
        item_sentences = [list(item_ids.keys()) for user in train for item_ids in user][2:] #skip first two empty
        start = time()
        w2v_model = Word2Vec(item_sentences, size=len(unique_genres), window=7, min_count=1, hs=1, workers=6, sg=1)
        w2v_model.build_vocab(vocab, update=True)
        w2v_model.train(item_sentences, total_examples=w2v_model.corpus_count, epochs=22)
        w2v_model.init_sims()
        w2v_model_time += time() - start

        w2v_final_rmse, w2v_final_mae, cos_sim_final_rmse, cos_sim_final_mae = 0.0, 0.0, 0.0, 0.0
        m = 0
        for i in tqdm(range(1,len(test))):
            #if i % 100 == 1: # using only 1% of test_set, otherwise it takes too long
            w2v_ratings_sum, w2v_abs_sum, cos_sim_ratings_sum, cos_sim_abs_sum = 0.0, 0.0, 0.0, 0.0
            test_user = test[i]
            for item in test_user:
                actual = test_user[item]
                startwv = time()
                predicted_w2v = score_item_to_user_w2v(item, {**train[i][0], **train[i][1]}, w2v_model, w2v_score_time)
                w2v_time += time() - startwv
                startcs = time()
                predicted_cos_sim = score_item_to_user_cos_sim(item, {**train[i][0], **train[i][1]}, cos_sim_matrix, cs_score_time)
                cs_time += time() - startcs
                w2v_abs_sum += abs(predicted_w2v - actual)
                w2v_ratings_sum += (predicted_w2v - actual)**2
                cos_sim_abs_sum += abs(predicted_cos_sim - actual)
                cos_sim_ratings_sum += (predicted_cos_sim - actual)**2
            w2v_user_rmse = math.sqrt(w2v_ratings_sum / len(test_user))
            w2v_user_mae = w2v_abs_sum/len(test_user)
            w2v_final_rmse += w2v_user_rmse
            w2v_final_mae += w2v_user_mae

            cos_sim_user_rmse = math.sqrt(cos_sim_ratings_sum / len(test_user))
            cos_sim_user_mae = cos_sim_abs_sum/len(test_user)
            cos_sim_final_rmse += cos_sim_user_rmse
            cos_sim_final_mae += cos_sim_user_mae
            m+=1
        W2VRMSE += w2v_final_rmse/m
        W2VMAE += w2v_final_mae/m
        COSSIMRMSE += cos_sim_final_rmse/m
        COSSIMMAE += cos_sim_final_mae/m
    print('W2VRMSE =', W2VRMSE)
    print('W2VMAE =', W2VMAE)
    print('COSSIMRMSE =', COSSIMRMSE)
    print('COSSIMMAE =', COSSIMMAE)
    print('W2V overall time =', w2v_time)
    print('COSSIM overall time =', cs_time)
    print('sim_mat calculations took', i_u_mat_time)
    print('w2v model creating time took', w2v_model_time)

if __name__ == "__main__":
    main()
