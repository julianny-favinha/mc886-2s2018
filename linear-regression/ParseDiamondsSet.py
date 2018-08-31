import pandas as pd

"""
	45849 -> training (35849 training and 10000 validation)
	8091 -> test
"""
def parse_diamonds_set():
	file = pd.read_csv("diamonds.csv")

	feat_carat = file["carat"]
	feat_price = file["price"]

	# carat is x
	feat_carat_train = feat_carat[:-8091]
	feat_carat_validation = feat_carat_train[-10000:]
	feat_carat_train = feat_carat_train[:-10000]
	feat_carat_test = feat_carat[-8091:]

	# price is y
	feat_price_train = feat_price[:-8091]
	feat_price_validation = feat_price_train[-10000:]
	feat_price_train = feat_price_train[:-10000]
	feat_price_test = feat_price[-8091:]

	return feat_carat_train, feat_carat_validation, feat_price_train, feat_price_validation