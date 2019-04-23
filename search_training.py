from collections import defaultdict
import pandas as pd
import pickle
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.scripts.glove2word2vec import glove2word2vec
# Load Google's pre-trained Word2Vec model.
from gensim.models.keyedvectors import KeyedVectors

pkl_file = open('inverted_index.pkl', 'rb')
inv_indx = pickle.load(pkl_file)
pkl_file.close()
glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)
stemmer = SnowballStemmer('english')


class search():

    def __init__(self):
        self.inv_indx = inv_indx
        self.glove_model = glove_model

    def strtolist(self, doc):
        return set(doc.split(' '))

    def lookup_query(self,query):
        words = word_tokenize(query)
        extented_words = []
        for word in words:
            if word not in STOP_WORDS:
                word = WordNetLemmatizer().lemmatize(word, pos='v')
                tmp = self.glove_model.most_similar(positive=word, topn=5)
                tmp = list(set([stemmer.stem(WordNetLemmatizer().lemmatize(item[0], pos='v')) for item in tmp if
                                item[0] not in STOP_WORDS]))
                tmp.append(word)
            else:
                continue
            extented_words.append(tmp)
        return extented_words

    def search_indexes(self,keywords,sent):
        extented_query = self.lookup_query(keywords)
        index_list = []

        if sent == 'Positive':
            sentList = ['positive']
        elif sent == 'Negative':
            sentList = ['negative']
        else:
            sentList = ['negative', 'neutral', 'positive']

        for word_list in extented_query:
            index_set_ = set()
            for word in word_list:
                index_set_ = index_set_.union(set(self.inv_indx[word]))
            index_list.append(index_set_)

        index_set = set(index_list[0])
        if len(index_list) > 1:
            for i in range(1, len(index_list)):
                index_set = index_set.intersection(set(index_list[i]))
            return index_set,sentList
        else:
            return index_set,sentList




