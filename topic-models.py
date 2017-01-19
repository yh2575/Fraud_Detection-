from __future__ import division, print_function

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from numpy.linalg import lstsq
import re

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF

df = pd.read_json('data/data.json')

df['fraud'] = 0
df.loc[df['acct_type'] == 'fraudster_event', 'fraud'] = 1
df.loc[df['acct_type'] == 'fraudster', 'fraud'] = 1
df.loc[df['acct_type'] == 'fraudster_att', 'fraud'] = 1

# previous payouts -- binary variable

df['has_previous_payouts'] = df.previous_payouts.apply(lambda x: int(x == []))

# gts values -- binned

df['gts_is_0'] = df.gts.apply(lambda x: int(x == 0))
df['gts_less_10'] = df.gts.apply(lambda x: int(0 < x < 10))
df['gts_less_25'] = df.gts.apply(lambda x: int(10 < x < 25))

# user country != venue country

country_mismatch = df.venue_country != df.country
df['venue_outside_user_country'] = country_mismatch.astype(int)

#country dummies

#num of tix for sale
# num of tix sold
# previous tix sold

#emails:
df['email_gmail'] = (df.email_domain == "gmail.com").astype(int)
df['email_yahoo'] = (df.email_domain == "yahoo.com").astype(int)
df['email_hotmail'] = (df.email_domain == "hotmail.com").astype(int)
df['email_aol'] = (df.email_domain == "aol.com").astype(int)
df['email_com'] = (df.email_domain[-3:] == "com").astype(int)
df['email_org'] = (df.email_domain[-3:] == "org").astype(int)
df['email_edu'] = (df.email_domain[-3:] == "edu").astype(int)

###Begin topic models###

#CLEAN HTML
def get_text(cell):
    return BeautifulSoup(cell, 'html.parser').get_text()

df['description'] = df['description'].apply(get_text)
df['org_desc'] = df['org_desc'].apply(get_text)
clean = df['description']

n_samples = len(clean)
n_features = 500
n_topics = 9
n_top_words = 30

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

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

print("Topics in NMF model:")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

###Assign topics to descriptions:
topic_dict = {0:'dinner_party', 1:'educational', 2:'social_networks', 3:'logistics', 4: 'business', 5:'university',
              6:'club_logistics', 7:'workshop', 8:'club_content'}

W = nmf.transform(tfidf)
topic_index = np.argmax(W, axis=1)
df['topic_index'] = topic_index
df['topic_index'] = df['topic_index'].replace(topic_dict)

###Create dummy variables to insert into model
topic_dummies = pd.get_dummies(df['topic_index']).rename(columns = lambda x: 'topic_'+str(x))
df = pd.concat([df,topic_dummies],axis=1)
