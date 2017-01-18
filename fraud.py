import numpy as np
import pandas as pd
from __future__ import division

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_json('data/data.json')

#Create binary fraud column
df['fraud'] = 0
df.loc[df['acct_type'] == 'fraudster_event', 'fraud'] = 1
df.loc[df['acct_type'] == 'fraudster', 'fraud'] = 1
df.loc[df['acct_type'] == 'fraudster_att', 'fraud'] = 1
#Check
print "Should be 1293, it is... "+ sum(df['fraud'])

#Train, test, split
y = df['fraud']
X = df[[         u'approx_payout_date',        u'body_length',
                 u'channels',            u'country',           u'currency',
          u'delivery_method',        u'description',       u'email_domain',
            u'event_created',          u'event_end',    u'event_published',
              u'event_start',       u'fb_published',                u'gts',
            u'has_analytics',         u'has_header',           u'has_logo',
                   u'listed',               u'name',        u'name_length',
                u'num_order',        u'num_payouts',          u'object_id',
                 u'org_desc',       u'org_facebook',           u'org_name',
              u'org_twitter',         u'payee_name',        u'payout_type',
         u'previous_payouts',      u'sale_duration',     u'sale_duration2',
                 u'show_map',       u'ticket_types',           u'user_age',
             u'user_created',          u'user_type',      u'venue_address',
            u'venue_country',     u'venue_latitude',    u'venue_longitude',
               u'venue_name',        u'venue_state',              ]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
