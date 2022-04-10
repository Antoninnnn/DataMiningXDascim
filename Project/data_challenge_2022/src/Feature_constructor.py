import csv
import networkx as nx
import numpy as np
import pandas as pd
import pickle

from collections import OrderedDict
from gensim.models import Word2Vec

from Text_authors_preprocessing import authors
from Text_abstracts_preprocessing import abstracts

from Text_abstracts_preprocessing import TextRankForKeyword

G = nx.read_edgelist('../data/edgelist.txt', delimiter=',',
                     create_using=nx.Graph(), nodetype=int)  # import the graph from the edgelist file.
nodes = list(G.nodes())
n = G.number_of_nodes()
m = G.number_of_edges()
print('Number of nodes:', n)
print('Number of edges:', m)

# open the testranked
with open('../data/generated_data/tr4w.pkl', 'rb') as abstracts_read:
    tr4w_input_abstracts = pickle.load(abstracts_read)

tr4w_abstracts = dict()
for node in tr4w_input_abstracts:
    tr4w_abstracts[node] = pickle.loads(tr4w_input_abstracts[node])


# open the abstract word2vec model
with open('../data/generated_data/abs_w2v_model_withmincount1.pkl', 'rb') as abs_w2v_model_file:
    abs_w2v_model = pickle.load(abs_w2v_model_file)
    abs_w2v_model = pickle.loads(abs_w2v_model)
    abs_w2v_model_file.close()


# introduce the distinct authors set
with open("../data/generated_data/authors.pkl", 'rb') as authorfile:
    distin_authors = pickle.load(authorfile)
    authorfile.close()

# introduce the model in .pkl file
with open('../data/generated_data/athrs_w2v_model.pkl', 'rb') as athrs_w2v_model_file:
    athrs_w2v_model = pickle.load(athrs_w2v_model_file)
    athrs_w2v_model = pickle.loads(athrs_w2v_model)
    athrs_w2v_model_file.close()


def node_abstract_feature(G, Word2Vec=abs_w2v_model, key_generator=tr4w_abstracts, f_reshape=(10, 10), num_keyword=10):
    feature = np.zeros((len(G.nodes), f_reshape[0], f_reshape[1], num_keyword))
    for n in G.nodes():
        # print(n)
        # generate list of all keyword vectors (the order is depend on the keyword's significance)
        keyws = list([i.lower() for i in OrderedDict(sorted(
            key_generator[n].node_weight.items(), key=lambda t: t[1], reverse=True))])

        # pick num_keyword vectors to construct the ndarray of shape (f_reshape,num_keyword): typically ((10,10),10)
        if len(keyws) < num_keyword:
            if len(keyws) > 0:
                origi_feature = np.zeros(
                    (num_keyword, f_reshape[0], f_reshape[1]))
                origi_feature[:len(keyws)] = np.array(
                    [Word2Vec.wv.get_vector(i).reshape(f_reshape) for i in keyws[:]])
                origi_feature[len(keyws):] = np.array(np.broadcast_to(np.mean([abs_w2v_model.wv.get_vector(i).reshape(
                    f_reshape) for i in keyws[:len(keyws)]], axis=0), (num_keyword-len(keyws), f_reshape[0], f_reshape[1])))
                feature[n] = origi_feature.transpose((1, -1, 0))
            else:
                feature[n] = np.zeros(
                    (f_reshape[0], f_reshape[1], num_keyword))
        else:
            feature[n] = np.array([Word2Vec.wv.get_vector(i).reshape(
                f_reshape) for i in keyws[:num_keyword]]).transpose((1, -1, 0))

    return feature


G_abs_feature = node_abstract_feature(G, num_keyword=8)


authors_word_data = [list(authors[node]) for node in authors]


def node_author_feature(G, Word2Vec=athrs_w2v_model, author_data=authors_word_data, f_reshape=(10, 10), num_author=2):
    feature = np.zeros((len(G.nodes), f_reshape[0], f_reshape[1], num_author))
    for n in G.nodes():
        # print(n)

        # pick num_keyword vectors to construct the ndarray of shape (f_reshape,num_keyword): typically ((10,10),10)
        if len(author_data[n]) < num_author:
            if len(author_data[n]) > 0:
                origi_feature = np.zeros(
                    (num_author, f_reshape[0], f_reshape[1]))
                origi_feature[:len(author_data[n])] = np.array(
                    [Word2Vec.wv.get_vector(i).reshape(f_reshape) for i in author_data[n][:]])
                if len(author_data[n]) == 1:
                    origi_feature[len(author_data[n]):] = np.array(np.broadcast_to(Word2Vec.wv.get_vector(
                        author_data[n][0]).reshape(f_reshape), (num_author-len(author_data[n]), f_reshape[0], f_reshape[1])))
                else:
                    origi_feature[len(author_data[n]):] = np.array(np.broadcast_to(np.mean([Word2Vec.wv.get_vector(i).reshape(
                        f_reshape) for i in author_data[n]], axis=0), (num_author-len(author_data[n]), f_reshape[0], f_reshape[1])))
                feature[n] = origi_feature.transpose((1, -1, 0))
            else:
                feature[n] = np.zeros((f_reshape[0], f_reshape[1], num_author))
        else:
            feature[n] = np.array([Word2Vec.wv.get_vector(i).reshape(
                f_reshape) for i in author_data[n][:num_author]]).transpose((1, -1, 0))

    return feature


G_athrs_feature = node_author_feature(G, num_author=2)


def node_feature(features=(G_abs_feature, G_athrs_feature), ax=3):
    return np.concatenate(features, axis=ax)


X = node_feature()
#np.save('../data/generated_data/X.npy', X)
