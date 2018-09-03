import numpy as np
import pickle
import pandas as pd
import time

from ParseDiamondsSet import getTrainingSet

"""
	Normal Equation method to find coeffiecients of hypothesis
	theta = inverse(Xt * X) * Xt * y
"""

start_time = time.time()

xs_train, ys_train, xs_validation, ys_validation = getTrainingSet()
xs_train.insert(loc=0, column="x0", value=np.ones(xs_train.shape[0]))

print("Applying Normal Equation...")
xst = xs_train.transpose()
inverse = np.linalg.inv(xst.dot(xs_train))
coeffs = (inverse.dot(xst)).dot(ys_train)
print("Coefficients:", coeffs)

# save normal equation model to a file
with open('NormalEquationModel', 'wb') as f:
	pickle.dump(coeffs, f)

elapsed_time = time.time() - start_time
print("Elapsed time: %1f s" %(elapsed_time))
