import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

from ParseDiamondsSet import numberVariables

start_time = time.time()

xs_train, xs_validation, xs_test, y_train, y_validation, y_test = numberVariables()

print("Applying SGDRegressor...")
regr = linear_model.SGDRegressor(max_iter=100000, eta0=0.0001)

print("Applying fit...")
regr.fit((xs_train.values).reshape(xs_train.shape[0], xs_train.shape[1]), y_train.values)

print("Applying prediction...")
y_pred = regr.predict((xs_test.values).reshape(xs_test.shape[0], xs_test.shape[1]))

print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)
print("Mean squared error: {}".format(mean_squared_error(y_test, y_pred)))

# plt.scatter(feat_carat_train, feat_price_train, color='gray')
# plt.scatter(feat_carat_validation, feat_price_validation, color='black')
# plt.plot(feat_carat_validation, feat_price_pred, color='blue', linewidth=3)

# plt.xticks()
# plt.yticks()

elapsed_time = time.time() - start_time
print("Elapsed time: %1f s" %(elapsed_time))

# plt.show()