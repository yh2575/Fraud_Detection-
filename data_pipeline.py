
import pandas as pd
import numpy as np

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

def get_tix(ticket_types, value):
	total = 0
	for ticket in ticket_types:
		total += ticket[value]
	return total
