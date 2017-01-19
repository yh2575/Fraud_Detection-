from __future__ import division

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

from sklearn.feature_extraction import text
from numpy.linalg import lstsq
import re

# PRINT FUNCTION
# Un-comment this if you want to print the topics, otherwise it it unnecessary.
# def print_top_words(model, feature_names, n_top_words):
#     for topic_idx, topic in enumerate(model.components_):
#         print(" ".join([feature_names[i]
#                         for i in topic.argsort()[:-n_top_words - 1:-1]]))


def topic_dummies(df):

    #CLEAN HTML FUNCTION
    def get_text(cell):
        return BeautifulSoup(cell, 'html.parser').get_text()

    #Parse descriptions using html function above:
    df['description'] = df['description'].apply(get_text)
    df['org_desc'] = df['org_desc'].apply(get_text)
    clean = df['description']

    #All the parameters for the topic modeling.
    n_samples = len(clean)
    n_features = 500
    n_topics = 9
    n_top_words = 30

    my_additional_stopwords = ["la", "et", "en", "le", "les", "des", 'january', 'february',
                           'march', 'april', 'may', 'june', 'july', 'august', 'september',
                           'october', 'november', 'december', 'friday', 'thursday', 'saturday']
    stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stopwords)


    # Use tf-idf features for NMF.
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                       max_features=n_features,
                                       stop_words=stop_words)
    tfidf = tfidf_vectorizer.fit_transform(clean)

    # Fit the NMF model
    nmf = NMF(n_components=n_topics, random_state=1,
              alpha=.1, l1_ratio=.5).fit(tfidf)

    #Leave this turned off unless you want to print.
    #tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    #print_top_words(nmf, tfidf_feature_names, n_top_words)

    '''
    #Assign topics to descriptions:
    #These are from the full data.  Do NOT use these descriptions on any subset, as they will not match.
    topic_dict = {0:'dinner_party', 1:'educational', 2:'social_networks', 3:'logistics', 4: 'business', 5:'university',
                  6:'club_logistics', 7:'workshop', 8:'club_content'}
    '''
    topic_dict = {0:'topic1', 1:'topic2', 2:'topic3', 3:'topic4', 4: 'topic5', 5:'topic6',
                  6:'topic7', 7:'topic8', 8:'topic9'}


    W = nmf.transform(tfidf)
    df['topic_index'] = np.argmax(W, axis=1)
    df['topic_index'] = df['topic_index'].replace(topic_dict)

    ###Create dummy variables to insert into model
    topic_dummies = pd.get_dummies(df['topic_index']).rename(columns = lambda x: 'topic_'+str(x))
    df = pd.concat([df,topic_dummies],axis=1)
    return df
