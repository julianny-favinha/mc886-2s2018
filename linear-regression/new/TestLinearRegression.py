import pickle
import numpy as np
import pandas as pd

from ParseDiamondsSet import getTestSet
from Cost import cost

file_name = input("Input file name of model: ")

with open (file_name, 'rb') as fp:
	coeffs = pickle.load(fp)
    
	xs_test, y_test = getTestSet()

	# insert column x0 of 1s
	xs_test.insert(loc=0, column="x0", value=np.ones(xs_test.shape[0]))

	cost_test = cost(coeffs, xs_test.values, y_test.values)

	print("MSE =", cost_test)