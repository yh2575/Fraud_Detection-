
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

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
	# Make a single binary column for fraud
	df['fraud'] = 0
	df.loc[df['acct_type'] == 'fraudster_event', 'fraud'] = 1
	df.loc[df['acct_type'] == 'fraudster', 'fraud'] = 1
	df.loc[df['acct_type'] == 'fraudster_att', 'fraud'] = 1
	#Check
	#print "Should be 1293, it is... "+ sum(df['fraud']).  It is! :)

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

	#fraudy countries
	fraud_one_sd_above = np.mean(df['fraud']) + np.std(df['fraud'])
	fraud_bools = df.groupby('country').mean()['fraud'] > fraud_one_sd_above
	high_fraud=df.groupby('country').mean()[fraud_bools]
	high_fraud_countries = high_fraud.index
	df['high_fraud_country'] = df.country.apply(lambda x: x in high_fraud_countries).astype(int)
	return df

def create_X_and_y(df):
	df = feature_engineering(df)
	y = df['fraud']
	X = df.drop('fraud')
	return X, y

def test_train_split(X, y)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
	return X_train, X_test, y_train, y_test

def smote(X, y, target, k=None):
    """
    INPUT:
    X, y - your data
    target - the percentage of positive class 
             observations in the output
    k - k in k nearest neighbors
    OUTPUT:
    X_oversampled, y_oversampled - oversampled data
    `smote` generates new observations from the positive (minority) class:
    For details, see: https://www.jair.org/media/953/live-953-2037-jair.pdf
    """
    if target <= sum(y)/float(len(y)):
        return X, y
    if k is None:
        k = len(X)**.5
    # fit kNN model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X[y==1], y[y==1])
    neighbors = knn.kneighbors()[0]
    positive_observations = X[y==1]
    # determine how many new positive observations to generate
    positive_count = sum(y)
    negative_count = len(y) - positive_count
    target_positive_count = target*negative_count / (1. - target)
    target_positive_count = int(round(target_positive_count))
    number_of_new_observations = target_positive_count - positive_count
    # generate synthetic observations
    synthetic_observations = np.empty((0, X.shape[1]))
    while len(synthetic_observations) < number_of_new_observations:
        obs_index = np.random.randint(len(positive_observations))
        observation = positive_observations[obs_index]
        neighbor_index = np.random.choice(neighbors[obs_index])
        neighbor = X[neighbor_index]
        obs_weights = np.random.random(len(neighbor))
        neighbor_weights = 1 - obs_weights
        new_observation = obs_weights*observation + neighbor_weights*neighbor
        synthetic_observations = np.vstack((synthetic_observations, new_observation))

    X_smoted = np.vstack((X, synthetic_observations))
    y_smoted = np.concatenate((y, [1]*len(synthetic_observations)))

    return X_smoted, y_smoted

def get_tix(ticket_types, value):
	total = 0
	for ticket in ticket_types:
		total += ticket[value]
	return total
