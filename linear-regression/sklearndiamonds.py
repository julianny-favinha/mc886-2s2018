import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

from ParseDiamondsSet import parse_diamonds_set

start_time = time.time()

# get variables carat and price
feat_carat_train, feat_carat_validation, feat_price_train, feat_price_validation = parse_diamonds_set()

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

elapsed_time = time.time() - start_time
print("Elapsed time: %1f s" %(elapsed_time))

plt.show()