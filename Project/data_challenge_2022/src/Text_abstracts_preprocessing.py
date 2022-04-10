import numpy as np
import time
import pickle

import nltk
from nltk.corpus import stopwords
# if nltk languages packages not installed, use the command
# nltk.download()
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer, WordNetLemmatizer

from collections import OrderedDict
import spacy
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm')


class TextRankForKeyword():
    """Extract keywords from text"""

    def __init__(self):
        self.d = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 10  # iteration steps
        self.node_weight = None  # save keywords and its weight
        #
        # self.lemmatizer = WordNetLemmatizer()

    def set_stopwords(self, stopwords):
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]  # set the stopword for the nlp model
            lexeme.is_stop = True

    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:

                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.lemma_.lower())
                    else:
                        selected_words.append(token.lemma_)

            sentences.append(selected_words)
        return sentences

    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs

    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())

    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1

        # Get Symmeric matrix
        g = self.symmetrize(g)

        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        # this is ignore the 0 element in norm
        g_norm = np.divide(g, norm, where=norm != 0)

        return g_norm

    def get_keywords(self, number=10):
        """Print top number keywords"""
        node_weight = OrderedDict(
            sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            print(key + ' - ' + str(value))
            if i > number:
                break

    def analyze(self, text,
                candidate_pos=['NOUN', 'PROPN'],
                window_size=4, lower=False, stopwords=list()):
        """Main function to analyze text"""

        # Set stop words
        self.set_stopwords(stopwords)

        # Pare text by spaCy
        doc = nlp(text)

        # visualize the doc sentences
        # for sent in doc.sents:
        #     displacy.render(sent, style="dep")
        # for sent in doc.sents:
        #     displacy.render(sent, style="ent")

        # Filter sentences
        sentences = self.sentence_segment(
            doc, candidate_pos, lower)  # list of list of words

        # Build vocabulary
        vocab = self.get_vocab(sentences)

        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)
        # print(token_pairs)
        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)

        # Initialization for weight(pagerank value)
        pr = np.array([1] * len(vocab))

        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]

        self.node_weight = node_weight


'''
print(lemmatizer.lemmatize('cooking'))
print(lemmatizer.lemmatize('cooking', pos='v'))
print(lemmatizer.lemmatize('cookbooks'))'''

'''
EXAMPLE_TEXT = "Hello World! Isn't it good to see you? Thanks for buying this book."

tokens = nltk.word_tokenize(EXAMPLE_TEXT)
tagged = nltk.pos_tag(tokens)
print(tagged[0:10])  # 词性标注
'''

data_path = '../data/'

# Read the abstract of each paper
abstracts = dict()
with open(data_path+'abstracts.txt', 'r') as f:
    for line in f:
        node, abstract = line.split('|--|')
        abstracts[int(node)] = abstract


# # Generate and register the textrank classes for 130000+ abstracts (correspondingly 130000+ classes)
# tr4w = dict()
# t = time.time()
# for node in abstracts:

#     tr4w[node] = TextRankForKeyword()
#     tr4w[node].analyze(abstracts[node], candidate_pos=[
#         'NOUN', 'PROPN', 'ADJ'], window_size=5, lower=False)
#     # tr4w[node].get_keywords(20)
#     # if int(node) == 4:
#     #     break
#     if int(node) % 1000 == 0 and int(node) != 0:
#         t_n = time.time()
#         print("the %d th absttract has been processed!" %
#               int(node), " [time cost:%d s]" % (t_n-t))
#         t = time.time()

# # for i in range(3):
# #     print("The %d th article's abstract: " % i, abstracts[i])
# #     print("The %d th article's abstract keywords: " % i, OrderedDict(
# #         sorted(tr4w[i].node_weight.items(), key=lambda t: t[1], reverse=True)))

# OutStr_tr4w = dict()
# for node in tr4w:
#     OutStr_tr4w[node] = pickle.dumps(tr4w[node])

# with open('../data/generated_data/tr4w.pkl', 'wb') as outputfile:
#     pickle.dump(OutStr_tr4w, outputfile)
#     outputfile.close()
