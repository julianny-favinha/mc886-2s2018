import numpy as np
import pandas as pd

def get_data():
	# load dataset
	file = pd.read_csv("diamonds.csv")

	# get variables carat and price
	feature_carat = file["carat"]
	feature_price = file["price"]

	# carat is x
	feat_carat_train = feat_carat[:-8091]
	feat_carat_validation = feat_carat[-8091:]

	# price is y
	feat_price_train = feat_price[:-8091]
	feat_price_validation = feat_price[-8091:]

	return feat_carat_train, feat_carat_validation, feat_price_train, feat_price_validation