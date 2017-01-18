# models!
from __future__ import division
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
# from sklearn.m import KFold
from sklearn.cross_validation import KFold
from statsmodels.api import Logit
from sklearn.metrics import accuracy_score, precision_score, recall_score

from fraud_utilities import fraudColumn_and_split

df = pd.read_json('../fraud-detection-case-study/data/data.json')
X_train, X_test, y_train, y_test = fraudColumn_and_split(df)

# further division
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=.3)

# simplest model 'delivery_method',
X_train = X_train[['fb_published','num_order']]
X_val = X_val[['fb_published','num_order']]

model = LogisticRegression()  # from sklearn
model.fit(X_train, y_train)
threshold = 0.15
predictions = model.predict_proba(X_val)[:, 1] > threshold

# plt.plot(predictions,'o')
# plt.show()

acc = accuracy_score(y_val,predictions)
presc = precision_score(y_val,predictions)
rec = recall_score(y_val,predictions)

print "Accuracy = {}, precision = {}, recall = {}".format(acc,presc,rec)