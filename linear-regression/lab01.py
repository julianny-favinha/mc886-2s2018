# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math

from ParseDiamondsSet import get_data
from BatchGradientDescent import BatchGradientDescent

def append_x0(xs):
	xs_aux = np.empty((0,2), int)
	for x in xs:
		new_x = np.insert(x, 0, 1)
		xs_aux = np.append(xs_aux, np.array([new_x]), axis=0)

	return xs_aux

if __name__ == "__main__":
	thetas = np.array([0.5, 2])
	
	# TODO: xs possui uma feature. adicionamos os x0 = 1 para todas as linhas
	xs = append_x0(np.array([[3], [3.5], [2], [7]]))

	ys = np.array([6, 8, 5, 9])

	# apply BGD to some number of iterations
	iterations = np.array([1, 1000, 100000, 100000])
	cost = []

	# TODO: como usar o conjunto de validacao (teste)?
	for it in iterations:
		print("Applying BGD for", it, "iterations...")
		bgd = BatchGradientDescent(it, 0.1)
		bgd.fit(xs, ys, thetas)
		batch_coef = bgd.coefficients
		print("Coefficients:", batch_coef)
		cost.append(bgd.cost(batch_coef, xs, ys))
		print()

	# plot cost x number of iterations graph
	plt.plot(iterations, cost, color="blue")
	plt.xticks(iterations)
	plt.xlabel("Number of iterations")
	plt.yticks()
	plt.ylabel("Cost")
	plt.show()

	# normal equation
	print("Applying Normal Equation...")
	bgd_normalEq = BatchGradientDescent()
	normalEq_coef = bgd_normalEq.normalEq(xs, ys)
	print("Coefficients:", normalEq_coef)