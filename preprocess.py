import gensim
import re
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
import numpy as np
import nltk
from collections import OrderedDict
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix
from nltk.stem.porter import *
from textblob import TextBlob

nlp = spacy.load('en')
stemmer = SnowballStemmer('english')
np.random.seed(1)


class LDA4Topic():

    def __init__(self):
        self.stopwords =gensim.parsing.preprocessing.STOPWORDS.union(set(
            ['verse','chorus','need','yuh','hey','ooh','oooh','yeah','aah','whoo']))
        self.d = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-3  # convergence threshold
        self.steps = 50  # iteration steps
        self.node_weight = None  # save keywords and its weight

    def lemmatize_stemming(self, token):
        return stemmer.stem(WordNetLemmatizer().lemmatize(token, pos='v'))

    def preprocess(self, text):
        text = text.lower()
        text = re.sub("</?.*?>", " <> ", text)
        text = re.sub("(\\d|\\W)+", " ", text)
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in self.stopwords and len(token) > 2 and token.isalpha() == True:
                result.append(self.lemmatize_stemming(token))
        return result

    def bow_corpus(self, processed_docs, no_below, no_above, keep_n):
        dictionary = gensim.corpora.Dictionary(processed_docs)
        dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        return dictionary, bow_corpus

    def lda_bow(self, documents, dictionary, bow_corpus, num_topics):
        lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=1)  # 100
        topics_bow = {}
        for idx, topic in lda_model.print_topics(-1):
            topics_bow[idx] = self.preprocess(topic)
        lda_bow = []
        for item in documents:
            topic_class = sorted(lda_model.get_document_topics(bow_corpus[item]), key=lambda x: x[1], reverse=True)
            if len(topic_class) == 0:
                lda_bow.append('null')
            else:
                lda_bow.append(topics_bow[topic_class[0][0]])
        return lda_bow

    def lda_tfidf(self, documents, dictionary, bow_corpus, num_topics):
        tfidf = models.TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]
        lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=num_topics, id2word=dictionary, passes=1)
        topics_tfidf = {}
        for idx, topic in lda_model_tfidf.print_topics(-1):
            topics_tfidf[idx] = self.preprocess(topic)
        lda_tfidf = []
        for item in documents:
            topic_class = sorted(lda_model_tfidf.get_document_topics(bow_corpus[item]), key=lambda x: x[1],
                                 reverse=True)
            if len(topic_class) == 0:
                lda_tfidf.append('null')
            else:
                lda_tfidf.append(topics_tfidf[topic_class[0][0]])
        return lda_tfidf


class TextRank4Keyword(LDA4Topic):

    def sentence_segment(self, doc):
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                token_text = re.sub("</?.*?>", " <> ", token.text)
                token_text = re.sub("(\\d|\\W)+", "", token_text)
                token_text = token_text.lower()
                if token_text not in self.stopwords and len(token_text) > 2 and token_text.isalpha()==True:
                    selected_words.append(stemmer.stem(WordNetLemmatizer().lemmatize(token_text, pos='v')))
            sentences.append(selected_words)
        # print(sentences)
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
                for j in range(i + 1, i + window_size):
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
        g_norm = np.divide(g, norm, where=norm != 0)  # this is ignore the 0 element in norm

        return g_norm

    def get_keywords(self, number):
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        keywords = []
        for i, (key, value) in enumerate(node_weight.items()):
            keywords.append(key)
            if i > number:
                break
        return keywords

    def analyze(self,text,window_size):
        doc = nlp(text)
        sentences = self.sentence_segment(doc)  # list of list of words
        vocab = self.get_vocab(sentences)
        token_pairs = self.get_token_pairs(window_size, sentences)
        g = self.get_matrix(vocab, token_pairs)
        pr = np.array([1] * len(vocab))
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1 - self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]

        self.node_weight = node_weight



class tfidf4keywords(LDA4Topic):

    def listtostr(self,text):
        text = " ".join(text)
        return text

    def tfidf_keywords(self,corpus):
        cv = CountVectorizer(max_df=0.5, stop_words=self.stopwords, max_features=20000, ngram_range=(1, 1))
        X = cv.fit_transform(corpus)
        tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
        tfidf_transformer.fit(X)
        feature_names = cv.get_feature_names()
        return tfidf_transformer,feature_names,cv

    def sort_coo(self,coo_matrix):
        tuples = zip(coo_matrix.col, coo_matrix.data)
        return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

    def extract_topn_from_vector(self,feature_names, sorted_items, topn):
        # use only topn items from vector
        sorted_items = sorted_items[:topn]

        score_vals = []
        feature_vals = []

        # word index and corresponding tf-idf score
        for idx, score in sorted_items:
            # keep track of feature name and its corresponding score
            score_vals.append(round(score, 3))
            feature_vals.append(feature_names[idx])

        results = {}
        for idx in range(len(feature_vals)):
            results[feature_vals[idx]] = score_vals[idx]

        return results

class Sentiment_Analysis(LDA4Topic):

    def preprocess(self, text):
        text = text.lower()
        text = re.sub("</?.*?>", " <> ", text)
        text = re.sub("(\\d|\\W)+", " ", text)
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in self.stopwords and len(token) > 2 and token.isalpha() == True:
                result.append(WordNetLemmatizer().lemmatize(token))
        return result

    def corpus(self,text):
        text = " ".join(text)
        return text

    def sentiment(self,text):
        return TextBlob(text).sentiment.polarity

    def pos_neg(self,polarity):
        if polarity >= 0.05:
            return 'positive'
        elif polarity < 0.05 and polarity > -0.05:
            return 'neutral'
        else:
            return 'negative'

def merge_keywords(doc):
    return list(set(doc))


if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv('/Users/mengdili/downloads/lyrics.csv')
    data = data.drop_duplicates(subset=['lyrics'], keep='first')
    data['lyrics'] = data['lyrics'].fillna('delete')
    hiphop = data[(data['genre'] == 'Hip-Hop') & (data['lyrics'] != 'delete') & (data['lyrics'].map(len) > 300)]
    hiphop = hiphop.reset_index(drop=True)
    del hiphop['index']
    lyrics = hiphop[['lyrics','song']]
    lyrics['index'] = lyrics.index
    documents = lyrics

    st_an = Sentiment_Analysis()
    sent_processed_docs = documents['lyrics'].map(st_an.preprocess)
    sent_corpus = sent_processed_docs.map(st_an.corpus)
    polarity= sent_corpus.map(st_an.sentiment)
    documents['sentiment'] = polarity.map(st_an.pos_neg)
    # documents['polarity'] = np.abs(polarity)

    lda4t = LDA4Topic()
    lda_processed_docs = documents['lyrics'].map(lda4t.preprocess)
    documents['title'] = documents['song'].map(lda4t.preprocess)
    dictionary, bow_corpus = lda4t.bow_corpus(lda_processed_docs, 15, 0.5, 20000)
    documents['lda_bow'] = lda4t.lda_bow(documents['index'], dictionary, bow_corpus, 50)
    documents['lda_tfidf'] = lda4t.lda_tfidf(documents['index'], dictionary, bow_corpus, 50)
    documents = documents[(documents['lda_bow'] != 'null') & (documents['lda_tfidf'] != 'null')]

    tfidf_doc = []
    ti4w = tfidf4keywords()
    tfidf_processed_docs = documents['lyrics'].map(ti4w.preprocess)
    tfidf_corpus = tfidf_processed_docs.map(ti4w.listtostr)
    tfidf_transformer, feature_names, cv = ti4w.tfidf_keywords(tfidf_corpus)
    for text in tfidf_corpus:
        tf_idf_vector = tfidf_transformer.transform(cv.transform([text]))
        sorted_items = ti4w.sort_coo(tf_idf_vector.tocoo())
        keywords_value = ti4w.extract_topn_from_vector(feature_names, sorted_items,topn=10)
        keywords = list(keywords_value.keys())
        tfidf_doc.append(keywords)
    documents['tfidf'] = tfidf_doc

    textrank_doc = []
    tr4w = TextRank4Keyword()
    for text in documents['lyrics']:
        tr4w.analyze(text, window_size=3)
        textrank_doc.append(tr4w.get_keywords(10))
    documents['textRank'] = textrank_doc

    documents['kw'] = documents['lda_bow']+documents['lda_tfidf']+documents['tfidf']+documents['textRank']+ documents['title']
    documents['keywords'] = documents['kw'].map(merge_keywords)
    documents['keywords'] = documents['keywords'].map(ti4w.listtostr)
    del documents['kw']
    del documents['title']
    del documents['textRank']
    del documents['lda_bow']
    del documents['lda_tfidf']
    del documents['tfidf']

    documents.to_csv('processed_lyrics.csv')

