# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time

from BatchGradientDescent import BatchGradientDescent
from ParseDiamondsSet import numberVariables

if __name__ == "__main__":
	xs_train, xs_validation, xs_test, ys_train, ys_validation, ys_test = numberVariables()

	# start with some thetas
	thetas = np.array([0.5, 2, 3, 4, 1.5, 6, 6])
	
	# insert column x0 of 1s
	xs_train.insert(loc=0, column="x0", value=np.ones(xs_train.shape[0]))
	xs_validation.insert(loc=0, column="x0", value=np.ones(xs_validation.shape[0]))
	xs_test.insert(loc=0, column="x0", value=np.ones(xs_test.shape[0]))

	# normal equation
	print("Applying Normal Equation...")
	bgd_normalEq = BatchGradientDescent()
	normalEq_coef = bgd_normalEq.normalEq(xs_train, ys_train)
	print("Coefficients:", normalEq_coef)

	# apply BGD to some number of iterations
	iterations = range(100, 100000, 1000)
	cost = []

	for it in iterations:
		start_time = time.time()
		print("Applying BGD for", it, "iterations...")
		bgd = BatchGradientDescent(it, 0.0001)
		bgd.fit(xs_train.values, ys_train, thetas)
		batch_coef = bgd.coefficients
		print("Coefficients:", batch_coef)
		batch_cost = bgd.cost(batch_coef, xs_validation.values, ys_validation.values)
		print("Cost:", batch_cost)
		cost.append(batch_cost)
		elapsed_time = time.time() - start_time
		print("Elapsed time: %1f s" %(elapsed_time))
		print()

	# plot cost x number of iterations graph
	plt.plot(iterations, cost, color="blue")
	plt.xticks()
	plt.xlabel("Number of iterations")
	plt.yticks()
	plt.ylabel("Cost")

	plt.show()
