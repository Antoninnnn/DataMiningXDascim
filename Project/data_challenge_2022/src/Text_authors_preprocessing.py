import csv
import numpy as np

data_path = '../data/'
# Read the abstract of each paper
authors = dict()
with open(data_path+'authors.txt', 'r') as f:
    for line in f:
        node, abstract = line.split('|--|')
        authors[int(node)] = abstract

# Map text to set of terms
for node in authors:
    authors[node] = set(authors[node][:-1].split(','))

# print([authors[node] for node in range(3)])
