import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.linear_model import LogisticRegression


def get_data():
    df = pd.read_json('data/data.json')

    #Create binary fraud column
    df['fraud'] = 0
    df.loc[df['acct_type'] == 'fraudster_event', 'fraud'] = 1
    df.loc[df['acct_type'] == 'fraudster', 'fraud'] = 1
    df.loc[df['acct_type'] == 'fraudster_att', 'fraud'] = 1
    #Check
    # print "Should be 1293, it is... "+ sum(df['fraud'])

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
    return X_train, X_test, y_train, y_test


def random_forrest():
    X1_train, x1_test, y1_train, y1_test = train_test_split(X_train, y_train, test_size=.2)

    columns = ['approx_payout_date', 'gts', 'previous_payouts', 'sale_duration2', 'num_order', 'body_length', 'venue_country!=country']

    #for trainning feature transformation
    X1_train['gts']= X1_train['gts'].apply(lambda x: 1 if x==0 else 0)
    X1_train['previous_payouts'] = X1_train['previous_payouts'].apply(lambda x: 1 if len(x)==0 else 0)
    X1_train['venue_country!=country']= (X1_train['country']!=X1_train['venue_country']).astype(int)
    X1_train_select = X1_train[columns]
    scaler = preprocessing.StandardScaler()
    scaler.fit(X1_train_select)
    scaler_train = scaler.transform(X1_train_select)


    #for test feature transformation
    x1_test['gts']= x1_test['gts'].apply(lambda x: 1 if x==0 else 0)
    x1_test['previous_payouts'] = x1_test['previous_payouts'].apply(lambda x: 1 if len(x)==0 else 0)
    x1_test['venue_country!=country']= (x1_test['country']!=x1_test['venue_country']).astype(int)
    x1_test_select = x1_test[columns]


    scaler_test = scaler.transform(x1_test_select)


    #random_forrest
    rf =  RandomForestClassifier(n_estimators=3, oob_score=True)
    rf.fit(scaler_train, y1_train)
    scalar_test_predict = rf.predict(scaler_test)

    import pdb; pdb.set_trace()

    rs = recall_score(scalar_test_predict, y1_test)
    #recall_score 0.861445783133

    ps = precision_score(scalar_test_predict, y1_test) #87%
    #precision_score 0.803370786517
    print "recall_score", rs
    print 'precision_score', ps

    print " accuracy score:", rf.score(scaler_test, y1_test)
    feature_importances = np.argsort(rf.feature_importances_)
    print " top five features:", list(X1_train_select.columns[feature_importances[-1:-6:-1]])

    #accuracy score: 0.971101145989

    #[u'previous_payouts', u'sale_duration2', u'num_order', u'body_length', u'approx_payout_date']



if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()
    random_forrest()
