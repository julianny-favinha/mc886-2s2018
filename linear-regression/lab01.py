# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time

from BatchGradientDescent import BatchGradientDescent
from ParseDiamondsSet import parse_diamonds_set

def appendX0(xs):
	xs_aux = np.empty((0,2), int)
	for x in xs:
		new_x = np.insert(x, 0, 1)
		xs_aux = np.append(xs_aux, np.array([new_x]), axis=0)

	return xs_aux

if __name__ == "__main__":
	feat_carat_train, feat_carat_validation, feat_price_train, feat_price_validation = parse_diamonds_set()

	# start with some thetas
	thetas = np.array([0.5, 2])
	
	# TODO: xs possui uma feature. adicionamos os x0 = 1 para todas as linhas
	xs = appendX0(feat_carat_train)

	ys = feat_price_train

	thetas = np.array([0.5, 2])
	xs = np.array([[1, 3], [1, 3.5], [1, 2]])
	ys = np.array([6, 8, 5])

	# apply BGD to some number of iterations
	iterations = np.array([10000, 1000000, 100000000])
	cost = []

	for it in iterations:
		start_time = time.time()
		print("Applying BGD for", it, "iterations...")
		bgd = BatchGradientDescent(it, 0.01)
		bgd.fit(xs, ys, thetas)
		batch_coef = bgd.coefficients
		print("Coefficients:", batch_coef)
		batch_cost = bgd.cost(batch_coef, xs, ys)
		print("Cost:", batch_cost)
		cost.append(batch_cost)
		elapsed_time = time.time() - start_time
		print("Elapsed time: %1f s" %(elapsed_time))
		print()

	# plot cost x number of iterations graph
	plt.plot(iterations, cost, color="blue")
	plt.xticks(iterations)
	plt.xlabel("Number of iterations")
	plt.yticks()
	plt.ylabel("Cost")

	# normal equation
	print("Applying Normal Equation...")
	bgd_normalEq = BatchGradientDescent()
	normalEq_coef = bgd_normalEq.normalEq(xs, ys)
	print("Coefficients:", normalEq_coef)

	plt.show()
