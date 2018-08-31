# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time

from BatchGradientDescent import BatchGradientDescent

"""
	45849 -> training (35849 training and 10000 validation)
	8091 -> test
"""
def parse_diamonds_set():
	file = pd.read_csv("diamonds.csv")

	feat_carat = file["carat"]
	feat_price = file["price"]

	# carat is x
	feat_carat_train = feat_carat[:-8091]
	feat_carat_validation = feat_carat_train[-10000:]
	feat_carat_train = feat_carat_train[:-10000]
	feat_carat_test = feat_carat[-8091:]

	# price is y
	feat_price_train = feat_price[:-8091]
	feat_price_validation = feat_price_train[-10000:]
	feat_price_train = feat_price_train[:-10000]
	feat_price_test = feat_price[-8091:]

	return feat_carat_train, feat_carat_validation, feat_price_train, feat_price_validation

def append_x0(xs):
	xs_aux = np.empty((0,2), int)
	for x in xs:
		new_x = np.insert(x, 0, 1)
		xs_aux = np.append(xs_aux, np.array([new_x]), axis=0)

	return xs_aux

if __name__ == "__main__":
	feat_carat_train, feat_carat_validation, feat_price_train, feat_price_validation = parse_diamonds_set()

	# starting with random thetas
	thetas = np.array([0.5, 2])
	
	# TODO: xs possui uma feature. adicionamos os x0 = 1 para todas as linhas
	xs = append_x0(feat_carat_train)
	#xs = append_x0(np.array([[3], [3.5], [2], [7]]))

	ys = feat_price_train
	#ys = np.array([6, 8, 5, 9])

	# apply BGD to some number of iterations
	iterations = np.array([100000])
	cost = []

	for it in iterations:
		start_time = time.time()
		print("Applying BGD for", it, "iterations...")
		bgd = BatchGradientDescent(it, 0.1)
		bgd.fit(xs, ys, thetas)
		batch_coef = bgd.coefficients
		print("Coefficients:", batch_coef)
		cost.append(bgd.cost(batch_coef, xs, ys))
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
