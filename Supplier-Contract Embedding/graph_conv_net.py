import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import csv
import ContentEmbedding
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk.tokenize
import Helper
import networkx

# Load adj mat dict style
adj_dict = {}
with open('out_train/filtered_adjacency_matrix_0.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    reader.next()
    for row in reader:
        if int(row[0]) not in adj_dict.keys():
            adj_dict[int(row[0])] = []
        adj_dict[int(row[0])].append(int(row[1]))

# Convert dictionary style to numpy style
n_suppliers = 235
n_contracts = 1364

adj_mat = np.zeros([235,1364])
for i in range(adj_mat.shape[0]):
    for j in range(adj_mat.shape[1]):
        if i in adj_dict and (j+n_suppliers) in adj_dict[i]:
            adj_mat[i][j] = 1
