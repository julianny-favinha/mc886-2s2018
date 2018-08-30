# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math

from parsediamondsset import get_data

class BatchGradientDescent:
	"""
		Attributes
	"""
	coefficients = None
	intercept = None
	
	"""
		Constructor, initializes number of iterations and learning rate 
		Will default do 100 iterations and learning Rate 0.5
	"""
	def __init__(self, numIt=100, lr=0.5):
		self.maxIteration = numIt
		self.learningRate = lr

	"""
		Calculates the hypotesis based on coefficients
	"""
	def hypothesis(self, thetas, xs):
		return np.sum([t*x for t, x in zip(thetas, xs)])

	"""
		Function to calculate mean squared error
	"""
	def cost(self, thetas, xs, ys):
		m = len(xs)
		sum = 0
		
		for i in range(m):
			sum += (self.hypothesis(thetas, xs[i]) - ys[i]) ** 2
		
		return (1/(2*m)) * sum

	"""
		Literal cost to view equation
	"""
	def verbose_cost(self, thetas, xs, ys):
		m = len(xs)
		print("cost = (1/(2*{}))(".format(m), end="")
		for i in range(m):
			if i < m-1:
				print("({} - {})^2 + ".format(self.hypothesis(thetas, xs[i]), ys[i]), end="")
			else:
				print("({} - {})^2)".format(self.hypothesis(thetas, xs[i]), ys[i]))

	"""
		Find coefficients of hypothesis using Gradient Descent
	"""
	def fit(self, xs, ys, initialGuess):
		thetas = initialGuess
		for _ in range(self.maxIteration):
			h = [self.hypothesis(thetas, x) for x in xs]
			n = len(thetas)
			currentThetas = np.empty(n, dtype=np.float64)
			for j in range(n):
				currentThetas[j] = 0
				somatoria = 0
				m = len(xs)
				for i in range(m):
					somatoria += (h[i] - ys[i]) * xs[i][j]
				if math.isnan(somatoria) or math.isinf(somatoria):
					raise ValueError("somatória se tornou infinita!")
				currentThetas[j] += thetas[j] - (self.learningRate * (1/m) * somatoria)
	
			thetas = currentThetas
		self.coefficients = thetas

	"""
		Normal Equation method to find coeffiecients of hypothesis
		theta = inverse(Xt * X) * Xt * y
	"""
	def normalEq(self, xs, ys):
		xst = xs.transpose()
		inverse = np.linalg.inv(xst.dot(xs))
		return (inverse.dot(xst)).dot(ys)
	
	#----------------------------------------------------------------------------------------------

def appendX0(xs):
	xs_aux = np.empty((0,2), int)
	for x in xs:
		new_x = np.insert(x, 0, 1)
		xs_aux = np.append(xs_aux, np.array([new_x]), axis=0)

	return xs_aux

if __name__ == "__main__":
	thetas = np.array([0.5, 2])
	
	# TODO: xs possui uma feature. adicionamos os x0 = 1 para todas as linhas
	xs = appendX0(np.array([[3], [3.5], [2], [7]]))

	ys = np.array([6, 8, 5, 9])

	# apply BGD to some number of iterations
	iterations = np.array([1, 1000, 100000, 1000000])
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