# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import pickle

from BatchGradientDescent import BatchGradientDescent
from ParseDiamondsSet import getTrainingSet
from Cost import cost

if __name__ == "__main__":
	learning_rate = 0.1
	xs_train, ys_train, xs_validation, ys_validation = getTrainingSet()
	
	# insert column x0 of 1s
	xs_train.insert(loc=0, column="x0", value=np.ones(xs_train.shape[0]))
	xs_validation.insert(loc=0, column="x0", value=np.ones(xs_validation.shape[0]))

	thetas = np.ones(xs_train.shape[1])

	# apply BGD to some number of iterations
	iterations = range(5000, 100005, 5000)
	cost_list = []
	coeffs = []

	for it in iterations:
		start_time = time.time()
		print("Applying BGD for", it, "iterations...")
		bgd = BatchGradientDescent(it, learning_rate)
		bgd.fit(xs_train.values, ys_train, thetas)
		batch_coef = bgd.coefficients
		print("Coefficients:", batch_coef)
		batch_cost = cost(batch_coef, xs_validation.values, ys_validation.values)
		print("Cost:", batch_cost)
		cost_list.append(batch_cost)
		coeffs.append(batch_coef)
		elapsed_time = time.time() - start_time
		print("Elapsed time: %1f s" %(elapsed_time))
		print()

	# save bgd model to a file
	with open('BGDModel', 'wb') as f:
		pickle.dump(coeffs[-1], f)

	# plot cost x number of iterations graph
	plt.title("BGD using Learning Rate = %f" %(learning_rate))
	plt.xticks()
	plt.xlabel("Number of iterations")
	plt.yticks()
	plt.ylabel("Cost")
	plt.plot(iterations, cost_list, color="blue")
	plt.savefig("CostGraphByNumberOfIterations")