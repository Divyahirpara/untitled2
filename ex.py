from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from gensim import corpora, models, similarities
from itertools import chain
from pandas import DataFrame
import csv
import gensim
from gensim import corpora
import numpy as np
import lda
import lda.datasets
import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.WARNING)
logging.root.level = logging.WARNING
data = pd.read_csv('/Users/hirpara/Desktop/Divya project/Dataset.csv',sep=',',index_col=False, encoding='latin-1')
documents = pd.DataFrame(data)
documents = documents['Unnamed: 1'][1:]
documents = documents.values
dcosCount = documents.size
#print(documents)

no_features = 1000


# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=1, min_df=1, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 5


# Run LDA
# Use tf (raw term count) features for LDA.
model = lda.LDA(n_topics=no_topics, n_iter=1000, random_state=0)
model.fit(tf)
#lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]
                    for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10
#Lda = gensim.models.ldamodel.LdaModel
print("\nTopics in LDA model:\n")
display_topics(model, tf_feature_names, no_top_words)
print("\n\n")


doc_topic = model.doc_topic_
trendData = np.array([[], [],[]])
for n in range(dcosCount):
    topic_most_pr = doc_topic[n].argmax()
    print("doc: {} topic: {}\n{}...".format(n,topic_most_pr,
documents[n]))
#trendData = np.append(trendData,[[topic_most_pr],[n],[documents.values[1][n]]],
#axis=1)
#trendData = trendData.transpose()
#trendData


