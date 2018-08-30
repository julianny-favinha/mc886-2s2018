import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

from parsediamondsset import get_data

# get variables carat and price
feat_carat_train, feat_carat_validation, feat_price_train, feat_price_validation = get_data()

print("Applying SGDRegressor...")
regr = linear_model.SGDRegressor(max_iter=100000, eta0=0.1)

print("Applying fit...")
regr.fit((feat_carat_train.values).reshape(-1, 1), feat_price_train.values)

print("Applying prediction...")
feat_price_pred = regr.predict((feat_carat_validation.values).reshape(-1, 1))

# The coefficients
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

print("Mean squared error: {}".format(mean_squared_error(feat_price_validation, feat_price_pred)))

plt.scatter(feat_carat_train, feat_price_train, color='gray')
plt.scatter(feat_carat_validation, feat_price_validation, color='black')
plt.plot(feat_carat_validation, feat_price_pred, color='blue', linewidth=3)

plt.xticks()
plt.yticks()

plt.show()