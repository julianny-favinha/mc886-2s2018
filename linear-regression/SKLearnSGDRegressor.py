import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
import pickle

from ParseDiamondsSet import getTrainingSet

start_time = time.time()

xs_train, ys_train, xs_validation, ys_validation = getTrainingSet()

print("Applying SGDRegressor...")
regr = linear_model.SGDRegressor(max_iter=100000, eta0=0.1)

print("Applying fit...")
regr.fit((xs_train.values).reshape(xs_train.shape[0], xs_train.shape[1]), ys_train.values)

print("Applying prediction...")
y_pred = regr.predict((xs_validation.values).reshape(xs_validation.shape[0], xs_validation.shape[1]))

print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)
print("Mean squared error: {}".format(mean_squared_error(ys_validation, y_pred)))

# TODO: SALVAR INTERCEPT
coeffs = np.insert(regr.coef_, 0, regr.intercept_)
with open('SGDRegressorModel', 'wb') as f:
	pickle.dump(coeffs, f)

elapsed_time = time.time() - start_time
print("Elapsed time: %1f s" %(elapsed_time))