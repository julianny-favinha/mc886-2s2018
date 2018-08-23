import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# load dataset
file = pd.read_csv("diamonds.csv")

# get variables carat and price
feature_carat = file["carat"]
feature_price = file["price"]

# carat is x
feature_carat_train = feature_carat[:-10000]
feature_carat_validation = feature_carat[-10000:]

# price is y
feature_price_train = feature_price[:-10000]
feature_price_validation = feature_price[-10000:]

regr = linear_model.SGDRegressor(max_iter=100000, eta0=0.1)

regr.fit((feature_carat_train.values).reshape(-1, 1), feature_price_train.values)

feature_price_pred = regr.predict((feature_carat_validation.values).reshape(-1, 1))

# The coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

print("Mean squared error: {}".format(mean_squared_error(feature_price_validation, feature_price_pred)))

plt.scatter(feature_carat_train, feature_price_train, color='gray')
plt.scatter(feature_carat_validation, feature_price_validation, color='black')
plt.plot(feature_carat_validation, feature_price_pred, color='blue', linewidth=3)

plt.xticks()
plt.yticks()

plt.show()