import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble.partial_dependence import plot_partial_dependence



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


def feature_engineering(df):

	# are there any previous payouts?

	df['has_previous_payouts'] = df.previous_payouts.apply(lambda x: int(x == []))

	# gts values -- binned

	df['gts_is_0'] = df.gts.apply(lambda x: int(x == 0))
	df['gts_less_10'] = df.gts.apply(lambda x: int(0 < x < 10))
	df['gts_less_25'] = df.gts.apply(lambda x: int(10 < x < 25))

	# user country != venue country

	country_mismatch = df.venue_country != df.country
	df['venue_outside_user_country'] = country_mismatch.astype(int)

	#country dummies

	#num of tix for sale (from ticket types)
	df['num_tix_total'] = df.ticket_types.apply(get_tix, args=("quantity_total",))
	# num of tix sold (from ticket types)
	df['num_tix_sold_by_event'] = df.ticket_types.apply(get_tix, args=("quantity_sold",))
	# previous tix sold (from previous_payouts)
	df['num_payouts'] = df.previous_payouts.apply(lambda x: len(x))

	#emails:
	df['email_gmail'] = (df.email_domain == "gmail.com").astype(int)
	df['email_yahoo'] = (df.email_domain == "yahoo.com").astype(int)
	df['email_hotmail'] = (df.email_domain == "hotmail.com").astype(int)
	df['email_aol'] = (df.email_domain == "aol.com").astype(int)
	df['email_com'] = (df.email_domain.apply(lambda x: x[-3:]) == "com").astype(int)
	df['email_org'] = (df.email_domain.apply(lambda x: x[-3:]) == "org").astype(int)
	df['email_edu'] = (df.email_domain.apply(lambda x: x[-3:]) == "edu").astype(int)

	return df

def get_tix(ticket_types, value):
	total = 0
	for ticket in ticket_types:
		total += ticket[value]
	return total

def scale_data(x_train, x_test):
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train)
    scaler_train = scaler.transform(x_train)
    scaler_test = scaler.transform(x_test)

    return scaler_train, scaler_test

def random_forrest(scaler_train, scaler_test):

    #random_forrest
    rf =  RandomForestClassifier(n_estimators=3, oob_score=True)
    rf.fit(scaler_train, y1_train)
    scaler_test_predict = rf.predict(scaler_test)

    rs = recall_score(scaler_test_predict, y1_test)
    #recall_score 0.861445783133
    ps = precision_score(scaler_test_predict, y1_test) #87%
    #precision_score 0.803370786517
    print "recall_score", rs
    print 'precision_score', ps
    print " accuracy score:", rf.score(scaler_test, y1_test)
    feature_importances = np.argsort(rf.feature_importances_)
    print " top 10 features:", list(x_feature_train.columns[feature_importances[-1:-10:-1]])
    return rf
    #recall_score 0.909090909091
    #precision_score 0.925925925926
    #accuracy score: 0.986547085202
    #['has_previous_payouts', 'num_tix_sold_by_event', u'num_payouts', u'approx_payout_date', 'num_tix_total', u'sale_duration2', u'body_length', u'num_order', 'email_com']
def random_forrest_class_balence(scaler_train, scaler_test):

    #random_forrest
    rf =  RandomForestClassifier(n_estimators=3, oob_score=True, class_weight={1:9})
    rf.fit(scaler_train, y1_train)
    scaler_test_predict = rf.predict(scaler_test)

    rs = recall_score(scaler_test_predict, y1_test)
    #recall_score 0.861445783133
    ps = precision_score(scaler_test_predict, y1_test) #87%
    #precision_score 0.803370786517
    print 'random forrest'
    print "recall_score", rs
    print 'precision_score', ps
    print " accuracy score:", rf.score(scaler_test, y1_test)
    feature_importances = np.argsort(rf.feature_importances_)
    print " top 10 features:", list(x_feature_train.columns[feature_importances[-1:-10:-1]])
    print "confusion_matrix:", confusion_matrix(scaler_test_predict, y1_test)
    print '------------------------------------------------------'

    return rf
    #recall_score 0.944134078212
    #precision_score 0.871134020619
    #accuracy score: 0.982561036373
    #top 10 features: ['has_previous_payouts', u'sale_duration2', 'num_tix_sold_by_event', u'body_length', u'num_payouts', u'approx_payout_date', 'venue_outside_user_country', u'num_order', 'num_tix_total']
def gdbr(scaler_train, scaler_test):
    gdbr = GradientBoostingClassifier(learning_rate=0.03,
                                 n_estimators=150, random_state=1)
    # weights = y1_train
    # weights.loc[weights==1] = 20
    # weights.loc[weights==0] = 80

    gdbr.fit(scaler_train, y1_train)
    scaler_test_predict = gdbr.predict(scaler_test)

    rs = recall_score(scaler_test_predict, y1_test)
    #recall_score 0.861445783133
    ps = precision_score(scaler_test_predict, y1_test) #87%
    #precision_score 0.803370786517
    print 'gdbr'
    print "recall_score", rs
    print 'precision_score', ps
    print " accuracy score:", gdbr.score(scaler_test, y1_test)
    print "confusion_matrix:", confusion_matrix(scaler_test_predict, y1_test)
    #recall_score 0.958333333333
    #precision_score 0.879781420765
    #accuracy score: 0.985550572995
    return gdbr

def cross_val(estimator, train_x, train_y):
    # n_jobs=-1 uses all the cores on your machine
    mse = cross_val_score(estimator, train_x, train_y,
                           scoring='mean_squared_error',
                           cv=5) * -1
    r2 = cross_val_score(estimator, train_x, train_y,
                           scoring='r2', cv=5)

    mean_mse = mse.mean()
    mean_r2 = r2.mean()
    params = estimator.get_params()
    name = estimator.__class__.__name__
    print '%s Train CV | MSE: %.3f | R2: %.3f' % (name, mean_mse, mean_r2)
    return mean_mse, mean_r2

def plot_feature_importance():
    indices = np.argsort(gdbr.feature_importances_)
    # plot as bar chart
    figure = plt.figure(figsize=(10,7))
    plt.barh(np.arange(len(columns)), gdbr.feature_importances_[indices],
             align='center', alpha=.5)
    plt.yticks(np.arange(len(columns)), np.array(columns)[indices], fontsize=14)
    plt.xticks(fontsize=14)
    _ = plt.xlabel('Relative importance', fontsize=18)
    plt.show()


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()
    X1_train, x1_test, y1_train, y1_test = train_test_split(X_train, y_train, test_size=.2)
    # columns = ['approx_payout_date', 'gts', 'previous_payouts', 'sale_duration2', 'num_order', 'body_length', 'venue_country!=country']
    columns = ['has_previous_payouts', 'gts_is_0', 'gts_less_10', 'gts_less_25', 'venue_outside_user_country', 'num_tix_total', 'num_tix_sold_by_event', 'num_payouts', 'email_gmail', 'email_yahoo', 'email_hotmail','email_aol','email_com', 'email_org', 'email_edu','approx_payout_date', 'sale_duration2', 'num_order', 'body_length']

    x_feature_train = feature_engineering(X1_train)[columns]
    x_feature_test = feature_engineering(x1_test)[columns]

    scaler_train, scaler_test = scale_data(x_feature_train, x_feature_test)

    # rf = random_forrest(scaler_train, scaler_test)

    rf2 = random_forrest_class_balence(scaler_train, scaler_test)
    gdbr = gdbr(scaler_train, scaler_test)

    cross_val(rf2, scaler_train, y1_train)
    cross_val(gdbr, scaler_train, y1_train)

    plot_feature_importance()
